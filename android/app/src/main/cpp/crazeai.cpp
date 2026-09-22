#include <jni.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor { std::vector<uint32_t> s; std::vector<float> x; };
static std::unordered_map<std::string,Tensor> M;
static uint32_t V=0,B=0,D=0,H=0,L=0;
static bool ready=false;
static constexpr uint32_t MAX_CTX=256;

static uint32_t u32(std::ifstream& f){uint32_t x=0;f.read((char*)&x,4);return x;}
static uint64_t u64(std::ifstream& f){uint64_t x=0;f.read((char*)&x,8);return x;}
static const Tensor& t(const std::string& n){return M.at(n);}

static void lin(const Tensor&w,const Tensor&b,const float*a,float*z,uint32_t o,uint32_t i){
    for(uint32_t r=0;r<o;r++){
        float q=b.x[r]; const float*p=w.x.data()+size_t(r)*i;
        for(uint32_t k=0;k<i;k++)q+=p[k]*a[k];
        z[r]=q;
    }
}

static void ln(const Tensor&w,const Tensor&b,const float*a,float*z){
    double m=0,v=0;
    for(uint32_t i=0;i<D;i++)m+=a[i];
    m/=D;
    for(uint32_t i=0;i<D;i++){double q=a[i]-m;v+=q*q;}
    v/=D;
    float inv=1.f/std::sqrt(float(v)+1e-5f);
    for(uint32_t i=0;i<D;i++)z[i]=(a[i]-float(m))*inv*w.x[i]+b.x[i];
}

static float gelu(float x){
    return .5f*x*(1.f+std::tanh(.79788456f*(x+.044715f*x*x*x)));
}

static bool load(const std::string&p){
    std::ifstream f(p,std::ios::binary); if(!f)return false;
    char magic[8]; f.read(magic,8);
    if(std::string(magic,8)!="CRAZEAI1"||u32(f)!=1)return false;
    V=u32(f);B=u32(f);D=u32(f);L=u32(f);H=u32(f);uint32_t n=u32(f);
    if(V!=258||!B||!D||!H||D%H||B>4096)return false;
    M.clear();
    for(uint32_t k=0;k<n;k++){
        uint32_t nl=u32(f);std::string name(nl,'\0');f.read(name.data(),nl);
        uint32_t nd=u32(f);Tensor q;q.s.resize(nd);uint64_t ne=1;
        for(uint32_t i=0;i<nd;i++){q.s[i]=u32(f);ne*=q.s[i];}
        uint64_t bytes=u64(f);if(bytes!=ne*4)return false;
        q.x.resize((size_t)ne);f.read((char*)q.x.data(),(std::streamsize)bytes);
        if(!f)return false;M.emplace(name,std::move(q));
    }
    ready=true;return true;
}

static std::vector<float> step(uint32_t token,uint32_t pos,
                               std::vector<std::vector<float>>&kc,
                               std::vector<std::vector<float>>&vc){
    const uint32_t HD=D/H;
    std::vector<float>x(D),norm(D),qkv(3*D),q(D),k(D),v(D);
    const auto&tw=t("tok.weight");const auto&pw=t("pos.weight");
    for(uint32_t d=0;d<D;d++)x[d]=tw.x[(size_t)token*D+d]+pw.x[(size_t)pos*D+d];

    for(uint32_t li=0;li<L;li++){
        const std::string p="blocks."+std::to_string(li)+".";
        const auto&ln1w=t(p+"ln1.weight");const auto&ln1b=t(p+"ln1.bias");
        const auto&qkvw=t(p+"attn.qkv.weight");const auto&qkvb=t(p+"attn.qkv.bias");
        const auto&ow=t(p+"attn.proj.weight");const auto&ob=t(p+"attn.proj.bias");

        ln(ln1w,ln1b,x.data(),norm.data());
        lin(qkvw,qkvb,norm.data(),qkv.data(),3*D,D);
        std::copy(qkv.begin(),qkv.begin()+D,q.begin());
        std::copy(qkv.begin()+D,qkv.begin()+2*D,k.begin());
        std::copy(qkv.begin()+2*D,qkv.end(),v.begin());

        const size_t base=(size_t)pos*D;
        for(uint32_t d=0;d<D;d++){kc[li][base+d]=k[d];vc[li][base+d]=v[d];}

        std::vector<float>att(D);const float scale=1.f/std::sqrt((float)HD);
        for(uint32_t h=0;h<H;h++){
            uint32_t qb=h*HD;std::vector<float>scores(pos+1);float mx=-1e30f;
            for(uint32_t j=0;j<=pos;j++){
                size_t jb=(size_t)j*D+qb;float s=0.f;
                for(uint32_t d=0;d<HD;d++)s+=q[qb+d]*kc[li][jb+d];
                scores[j]=s*scale;mx=std::max(mx,scores[j]);
            }
            float sum=0.f;for(float&s:scores){s=std::exp(s-mx);sum+=s;}
            float inv=1.f/sum;
            for(uint32_t d=0;d<HD;d++){
                float z=0.f;
                for(uint32_t j=0;j<=pos;j++)z+=scores[j]*inv*vc[li][(size_t)j*D+qb+d];
                att[qb+d]=z;
            }
        }

        std::vector<float>proj(D);lin(ow,ob,att.data(),proj.data(),D,D);
        for(uint32_t d=0;d<D;d++)x[d]+=proj[d];

        const auto&ln2w=t(p+"ln2.weight");const auto&ln2b=t(p+"ln2.bias");
        const auto&w1=t(p+"mlp.0.weight");const auto&b1=t(p+"mlp.0.bias");
        const auto&w2=t(p+"mlp.2.weight");const auto&b2=t(p+"mlp.2.bias");
        std::vector<float>hid(4*D),out(D);
        ln(ln2w,ln2b,x.data(),norm.data());lin(w1,b1,norm.data(),hid.data(),4*D,D);
        for(float&z:hid)z=gelu(z);lin(w2,b2,hid.data(),out.data(),D,4*D);
        for(uint32_t d=0;d<D;d++)x[d]+=out[d];
    }

    std::vector<float>last(D),logits(V);
    ln(t("ln.weight"),t("ln.bias"),x.data(),last.data());
    for(uint32_t id=0;id<V;id++){
        float z=0.f;const float*r=tw.x.data()+(size_t)id*D;
        for(uint32_t d=0;d<D;d++)z+=r[d]*last[d];
        logits[id]=z;
    }
    return logits;
}

static bool repeated_single_token(const std::string& s, int window = 12) {
    if ((int)s.size() < window) return false;
    char last = s.back();
    for (int i = 1; i < window; ++i)
        if (s[s.size() - 1 - i] != last) return false;
    return true;
}

static bool repeated_short_pattern(const std::string& s) {
    if (s.size() < 24) return false;
    size_t n = s.size();
    for (int i = 0; i < 8; ++i)
        if (s[n - 24 + i] != s[n - 16 + i] ||
            s[n - 24 + i] != s[n - 8 + i]) return false;
    return true;
}

static std::string run(const std::string&userPrompt,int max,float){
    if(!ready)return "Load a .craze model first.";

    // Match the exact instruction format used by the training dataset.
    std::string prompt="### Instruction:\

"+userPrompt+
                       "\n### Input:\n\n### Output:\n";

    std::vector<uint32_t>ids(1,256);
    for(unsigned char c:prompt)ids.push_back((uint32_t)c);
    if(ids.size()>MAX_CTX-1)
        ids.erase(ids.begin()+1,ids.end()-(MAX_CTX-1));

    std::vector<std::vector<float>>kc(L,std::vector<float>((size_t)MAX_CTX*D,0.f));
    std::vector<std::vector<float>>vc(L,std::vector<float>((size_t)MAX_CTX*D,0.f));
    std::vector<float>logits;

    for(size_t i=0;i<ids.size();i++)logits=step(ids[i],(uint32_t)i,kc,vc);

    std::string out;
    const int limit=std::min(max,48);

    // Deterministic decoding is much safer for this small, lightly trained model.
    for(int n=0;n<limit;n++){
        int pick=0;
        float best=-1e30f;
        for(int i=0;i<(int)V;i++){
            if(logits[i]>best){best=logits[i];pick=i;}
        }

        if(pick==257)break;
        if(pick>=256)break;

        out.push_back((char)pick);

        if (repeated_single_token(out) || repeated_short_pattern(out)) break;

        uint32_t pos=(uint32_t)ids.size();
        if(pos>=MAX_CTX)break;
        ids.push_back((uint32_t)pick);
        logits=step((uint32_t)pick,pos,kc,vc);
    }

    // Remove obvious control bytes from malformed generations.
    std::string clean;
    for(unsigned char c:out){
        if(c=='\n'||c=='\r'||c=='\t'||(c>=32&&c<127))clean.push_back((char)c);
    }
    if (clean.empty()) return "No useful response generated yet.";
    return clean;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_craze_runtime_MainActivity_loadModel(JNIEnv*e,jobject,jstring path){
    const char*p=e->GetStringUTFChars(path,nullptr);
    bool ok=p&&load(p);if(p)e->ReleaseStringUTFChars(path,p);
    return ok?JNI_TRUE:JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_craze_runtime_MainActivity_generate(JNIEnv*e,jobject,jstring prompt,jint max,jfloat temp){
    if(!ready)return e->NewStringUTF("Load a .craze model first.");
    const char*p=e->GetStringUTFChars(prompt,nullptr);
    std::string r=run(p?p:"",(int)max,(float)temp);
    if(p)e->ReleaseStringUTFChars(prompt,p);
    return e->NewStringUTF(r.c_str());
}
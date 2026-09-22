#include <jni.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor { std::vector<uint32_t> s; std::vector<float> x; };
static std::unordered_map<std::string,Tensor> M;
static uint32_t V,B,D,H,L; static bool ready=false;
static uint32_t u32(std::ifstream&f){uint32_t x;f.read((char*)&x,4);return x;}
static uint64_t u64(std::ifstream&f){uint64_t x;f.read((char*)&x,8);return x;}
static const Tensor& t(const std::string&n){return M.at(n);}
static void lin(const Tensor&w,const Tensor&b,const float*a,float*z,uint32_t o,uint32_t i){
 for(uint32_t r=0;r<o;r++){float q=b.x[r];const float*p=w.x.data()+size_t(r)*i;for(uint32_t k=0;k<i;k++)q+=p[k]*a[k];z[r]=q;}
}
static void ln(const Tensor&w,const Tensor&b,const float*a,float*z){
 double m=0,v=0;for(uint32_t i=0;i<D;i++)m+=a[i];m/=D;
 for(uint32_t i=0;i<D;i++){double q=a[i]-m;v+=q*q;}v/=D;float q=1.f/std::sqrt(float(v)+1e-5f);
 for(uint32_t i=0;i<D;i++)z[i]=(a[i]-float(m))*q*w.x[i]+b.x[i];
}
static float gelu(float x){return .5f*x*(1.f+std::tanh(.79788456f*(x+.044715f*x*x*x)));}

static bool load(const std::string&p){
 std::ifstream f(p,std::ios::binary);if(!f)return false;char magic[8];f.read(magic,8);
 if(std::string(magic,8)!="CRAZEAI1"||u32(f)!=1)return false;
 V=u32(f);B=u32(f);D=u32(f);L=u32(f);H=u32(f);uint32_t n=u32(f);
 if(V!=258||!B||!D||!H||D%H)return false;M.clear();
 for(uint32_t k=0;k<n;k++){
  uint32_t nl=u32(f);std::string name(nl,'\0');f.read(name.data(),nl);uint32_t nd=u32(f);Tensor q;q.s.resize(nd);uint64_t ne=1;
  for(uint32_t i=0;i<nd;i++){q.s[i]=u32(f);ne*=q.s[i];}uint64_t bytes=u64(f);
  if(bytes!=ne*4)return false;q.x.resize((size_t)ne);f.read((char*)q.x.data(),(std::streamsize)bytes);if(!f)return false;M.emplace(name,std::move(q));
 }
 ready=true;return true;
}

static std::vector<float> forward(const std::vector<uint32_t>&ids){
 const uint32_t T=(uint32_t)ids.size(),HD=D/H;std::vector<float>x((size_t)T*D),nrm((size_t)T*D),tmp((size_t)T*D);
 const auto&tw=t("tok.weight");const auto&pw=t("pos.weight");
 for(uint32_t q=0;q<T;q++)for(uint32_t d=0;d<D;d++)x[(size_t)q*D+d]=tw.x[(size_t)ids[q]*D+d]+pw.x[(size_t)q*D+d];
 for(uint32_t li=0;li<L;li++){
  std::string p="blocks."+std::to_string(li)+".";
  const auto&a=t(p+"ln1.weight"),&ab=t(p+"ln1.bias"),&qw=t(p+"attn.qkv.weight"),&qb=t(p+"attn.qkv.bias");
  const auto&ow=t(p+"attn.proj.weight"),&ob=t(p+"attn.proj.bias");
  std::vector<float>q((size_t)T*D),k(q.size()),v(q.size()),qkv(3*D);
  for(uint32_t pos=0;pos<T;pos++){ln(a,ab,&x[(size_t)pos*D],nrm.data()+(size_t)pos*D);lin(qw,qb,nrm.data()+(size_t)pos*D,qkv.data(),3*D,D);
   std::copy(qkv.begin(),qkv.begin()+D,q.begin()+(size_t)pos*D);std::copy(qkv.begin()+D,qkv.begin()+2*D,k.begin()+(size_t)pos*D);std::copy(qkv.begin()+2*D,qkv.end(),v.begin()+(size_t)pos*D);}
  std::vector<float>att((size_t)T*D),score(T);float scale=1.f/std::sqrt((float)HD);
  for(uint32_t pos=0;pos<T;pos++)for(uint32_t h=0;h<H;h++){
   float mx=-1e30f;for(uint32_t j=0;j<=pos;j++){float s=0;for(uint32_t d=0;d<HD;d++)s+=q[(size_t)pos*D+h*HD+d]*k[(size_t)j*D+h*HD+d];score[j]=s*scale;mx=std::max(mx,score[j]);}
   float sum=0;for(uint32_t j=0;j<=pos;j++){score[j]=std::exp(score[j]-mx);sum+=score[j];}
   for(uint32_t d=0;d<HD;d++){float z=0;for(uint32_t j=0;j<=pos;j++)z+=score[j]/sum*v[(size_t)j*D+h*HD+d];att[(size_t)pos*D+h*HD+d]=z;}
  }
  for(uint32_t pos=0;pos<T;pos++){lin(ow,ob,att.data()+(size_t)pos*D,tmp.data()+(size_t)pos*D,D,D);for(uint32_t d=0;d<D;d++)x[(size_t)pos*D+d]+=tmp[(size_t)pos*D+d];}
  const auto&c=t(p+"ln2.weight"),&cb=t(p+"ln2.bias"),&w1=t(p+"mlp.0.weight"),&b1=t(p+"mlp.0.bias"),&w2=t(p+"mlp.2.weight"),&b2=t(p+"mlp.2.bias");
  std::vector<float>hid(4*D),out(D);
  for(uint32_t pos=0;pos<T;pos++){ln(c,cb,x.data()+(size_t)pos*D,nrm.data()+(size_t)pos*D);lin(w1,b1,nrm.data()+(size_t)pos*D,hid.data(),4*D,D);for(float&z:hid)z=gelu(z);lin(w2,b2,hid.data(),out.data(),D,4*D);for(uint32_t d=0;d<D;d++)x[(size_t)pos*D+d]+=out[d];}
 }
 std::vector<float>last(D),logits(V);ln(t("ln.weight"),t("ln.bias"),x.data()+(size_t)(T-1)*D,last.data());
 for(uint32_t id=0;id<V;id++){float z=0;const float*r=tw.x.data()+(size_t)id*D;for(uint32_t d=0;d<D;d++)z+=r[d]*last[d];logits[id]=z;}return logits;
}

static std::string run(const std::string&s,int max,float temp){
 std::vector<uint32_t>ids(1,256);for(unsigned char c:s)ids.push_back(c);std::mt19937 rng(std::random_device{}());std::string out;
 for(int step=0;step<max;step++){
  size_t st=ids.size()>B?ids.size()-B:0;std::vector<uint32_t>ctx(ids.begin()+st,ids.end());auto z=forward(ctx);float q=std::max(temp,.01f);for(float&v:z)v/=q;
  int K=std::min(40,(int)V);std::vector<int>ix(V);for(int i=0;i<(int)V;i++)ix[i]=i;
  std::partial_sort(ix.begin(),ix.begin()+K,ix.end(),[&](int a,int b){return z[a]>z[b];});float mx=-1e30f;for(int i=0;i<K;i++)mx=std::max(mx,z[ix[i]]);
  std::vector<float>pr(K);float sum=0;for(int i=0;i<K;i++){pr[i]=std::exp(z[ix[i]]-mx);sum+=pr[i];}std::uniform_real_distribution<float>rd(0,sum);float r=rd(rng),acc=0;int pick=ix[K-1];
  for(int i=0;i<K;i++){acc+=pr[i];if(r<=acc){pick=ix[i];break;}}ids.push_back((uint32_t)pick);if(pick==257)break;if(pick<256)out.push_back((char)pick);
 }return out;
}

extern "C" JNIEXPORT jboolean JNICALL Java_ai_craze_runtime_MainActivity_loadModel(JNIEnv*e,jobject,jstring path){
 const char*p=e->GetStringUTFChars(path,nullptr);bool ok=p&&load(p);if(p)e->ReleaseStringUTFChars(path,p);return ok?JNI_TRUE:JNI_FALSE;
}
extern "C" JNIEXPORT jstring JNICALL Java_ai_craze_runtime_MainActivity_generate(JNIEnv*e,jobject,jstring prompt,jint max,jfloat temp){
 if(!ready)return e->NewStringUTF("Load a .craze model first.");const char*p=e->GetStringUTFChars(prompt,nullptr);std::string r=run(p?p:"",(int)max,temp);if(p)e->ReleaseStringUTFChars(prompt,p);return e->NewStringUTF(r.c_str());
}

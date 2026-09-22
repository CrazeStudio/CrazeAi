package ai.craze.runtime;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.*;
import java.io.*;
import java.util.concurrent.*;

public class MainActivity extends Activity {
 static { System.loadLibrary("crazeai"); }
 private native boolean loadModel(String path);
 private native String generate(String prompt,int maxTokens,float temperature);
 private final ExecutorService executor=Executors.newSingleThreadExecutor();
 private TextView status,output; private EditText input; private Button gen;
 private static final int PICK=42;

 @Override public void onCreate(Bundle b){
  super.onCreate(b); LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(24,24,24,24);
  TextView title=new TextView(this);title.setText("CrazeAI • Local Native");title.setTextSize(24);root.addView(title);
  status=new TextView(this);status.setText("No model loaded.");root.addView(status);
  Button load=new Button(this);load.setText("Load .craze Model");root.addView(load);
  input=new EditText(this);input.setHint("Ask CrazeAI...");input.setMinLines(3);root.addView(input);
  gen=new Button(this);gen.setText("Generate");gen.setEnabled(false);root.addView(gen);
  output=new TextView(this);output.setTextIsSelectable(true);root.addView(output);setContentView(root);
  load.setOnClickListener(v->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("*/*");startActivityForResult(i,PICK);});
  gen.setOnClickListener(v->generateAsync());
 }
 private void generateAsync(){
  String p=input.getText().toString();if(p.trim().isEmpty())return;gen.setEnabled(false);status.setText("Generating...");
  executor.execute(()->{String a=generate(p,128,.8f);runOnUiThread(()->{output.setText(a);status.setText("Ready");gen.setEnabled(true);});});
 }
 @Override protected void onActivityResult(int r,int result,Intent data){
  super.onActivityResult(r,result,data);if(r!=PICK||result!=RESULT_OK||data==null)return;Uri u=data.getData();status.setText("Copying model...");
  executor.execute(()->{try{
   File f=new File(getFilesDir(),"crazeai.craze");try(InputStream in=getContentResolver().openInputStream(u);OutputStream out=new FileOutputStream(f)){
    byte[]buf=new byte[1024*1024];int n;while((n=in.read(buf))!=-1)out.write(buf,0,n);
   }
   boolean ok=loadModel(f.getAbsolutePath());runOnUiThread(()->{status.setText(ok?"Model loaded • Ready":"Model load failed");gen.setEnabled(ok);});
  }catch(Exception e){runOnUiThread(()->status.setText("Load error: "+e.getMessage()));}});
 }
 @Override protected void onDestroy(){executor.shutdownNow();super.onDestroy();}
}

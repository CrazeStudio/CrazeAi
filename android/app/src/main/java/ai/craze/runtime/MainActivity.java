package ai.craze.runtime;

import android.app.Activity;
import android.os.Bundle;
import android.widget.*;

public class MainActivity extends Activity {
    static { System.loadLibrary("crazeai"); }
    private native String generate(String prompt, int maxTokens, float temperature);

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24,24,24,24);
        TextView title = new TextView(this);
        title.setText("CrazeAI • Local");
        title.setTextSize(24);
        EditText input = new EditText(this);
        input.setHint("Ask CrazeAI...");
        Button run = new Button(this);
        run.setText("Generate");
        TextView output = new TextView(this);
        output.setTextIsSelectable(true);
        root.addView(title); root.addView(input); root.addView(run); root.addView(output);
        setContentView(root);
        run.setOnClickListener(v -> output.setText(generate(input.getText().toString(), 256, 0.8f)));
    }
}

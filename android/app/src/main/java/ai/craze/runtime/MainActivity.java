package ai.craze.runtime;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import java.io.*;
import java.util.concurrent.*;

public class MainActivity extends Activity {
    static { System.loadLibrary("crazeai"); }

    private native boolean loadModel(String path);
    private native String generate(String prompt, int maxTokens, float temperature);

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView status, chat;
    private EditText input;
    private ImageButton send, menu;
    private static final int PICK = 42;

    private int dp(float v) { return (int)(v * getResources().getDisplayMetrics().density + 0.5f); }

    private GradientDrawable bg(String color, float radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(Color.parseColor(color));
        d.setCornerRadius(dp(radius));
        return d;
    }

    private TextView text(String value, float size, int color) {
        TextView t = new TextView(this);
        t.setText(value);
        t.setTextSize(size);
        t.setTextColor(color);
        return t;
    }

    @Override public void onCreate(Bundle b) {
        super.onCreate(b);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0B0B0F"));

        // Header
        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(14), dp(10), dp(10), dp(10));

        ImageView logo = new ImageView(this);
        logo.setImageResource(ai.craze.runtime.R.drawable.ic_crazeai_logo);
        LinearLayout.LayoutParams lpLogo = new LinearLayout.LayoutParams(dp(42), dp(42));
        header.addView(logo, lpLogo);

        LinearLayout brand = new LinearLayout(this);
        brand.setOrientation(LinearLayout.VERTICAL);
        brand.setPadding(dp(10), 0, 0, 0);
        TextView title = text("CrazeAI", 20, Color.WHITE);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        brand.addView(title);
        status = text("No model loaded", 12, Color.parseColor("#9CA3AF"));
        brand.addView(status);
        header.addView(brand, new LinearLayout.LayoutParams(0, -2, 1));

        menu = new ImageButton(this);
        menu.setImageResource(android.R.drawable.ic_menu_more);
        menu.setColorFilter(Color.WHITE);
        menu.setBackgroundColor(Color.TRANSPARENT);
        menu.setContentDescription("Menu");
        header.addView(menu, new LinearLayout.LayoutParams(dp(48), dp(48)));

        root.addView(header);

        // Chat area
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        chat = text("", 16, Color.parseColor("#E5E7EB"));
        chat.setPadding(dp(20), dp(18), dp(20), dp(18));

        LinearLayout chatWrap = new LinearLayout(this);
        chatWrap.setOrientation(LinearLayout.VERTICAL);
        chatWrap.setGravity(Gravity.CENTER_HORIZONTAL);
        chatWrap.addView(chat, new LinearLayout.LayoutParams(-1, -2));

        TextView welcome = text("What can I help with?", 28, Color.WHITE);
        welcome.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        welcome.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams wp = new LinearLayout.LayoutParams(-1, -2);
        wp.setMargins(dp(20), dp(80), dp(20), dp(16));
        chatWrap.addView(welcome, 0, wp);

        TextView hint = text("Ask CrazeAI anything", 15, Color.parseColor("#9CA3AF"));
        hint.setGravity(Gravity.CENTER);
        chatWrap.addView(hint, 1, new LinearLayout.LayoutParams(-1, -2));

        scroll.addView(chatWrap);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));

        // Composer
        LinearLayout composer = new LinearLayout(this);
        composer.setGravity(Gravity.CENTER_VERTICAL);
        composer.setPadding(dp(10), dp(8), dp(10), dp(12));

        LinearLayout box = new LinearLayout(this);
        box.setGravity(Gravity.CENTER_VERTICAL);
        box.setPadding(dp(14), dp(2), dp(6), dp(2));
        box.setBackground(bg("#19191F", 28));

        input = new EditText(this);
        input.setHint("Message CrazeAI...");
        input.setHintTextColor(Color.parseColor("#777784"));
        input.setTextColor(Color.WHITE);
        input.setTextSize(16);
        input.setSingleLine(false);
        input.setMaxLines(4);
        input.setBackgroundColor(Color.TRANSPARENT);
        box.addView(input, new LinearLayout.LayoutParams(0, -2, 1));

        send = new ImageButton(this);
        send.setImageResource(android.R.drawable.ic_media_play);
        send.setColorFilter(Color.WHITE);
        send.setBackground(bg("#8B5CF6", 30));
        send.setContentDescription("Send");
        box.addView(send, new LinearLayout.LayoutParams(dp(48), dp(48)));

        composer.addView(box, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(composer);

        setContentView(root);

        View.OnClickListener loadAction = v -> {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("*/*");
            startActivityForResult(i, PICK);
        };
        menu.setOnClickListener(loadAction);
        send.setOnClickListener(v -> generateAsync());
    }

    private void generateAsync() {
        String p = input.getText().toString();
        if (p.trim().isEmpty()) return;
        send.setEnabled(false);
        status.setText("Thinking...");
        String old = chat.getText().toString();
        chat.setText(old + (old.isEmpty() ? "" : "\n\n") + "You\n" + p + "\n\nCrazeAI\n");
        input.setText("");

        executor.execute(() -> {
            String answer = generate(p, 256, 0.8f);
            runOnUiThread(() -> {
                String current = chat.getText().toString();
                chat.setText(current + answer);
                status.setText("Ready");
                send.setEnabled(true);
            });
        });
    }

    @Override protected void onActivityResult(int r, int result, Intent data) {
        super.onActivityResult(r, result, data);
        if (r != PICK || result != RESULT_OK || data == null) return;
        Uri u = data.getData();
        status.setText("Loading model...");
        executor.execute(() -> {
            try {
                File f = new File(getFilesDir(), "crazeai.craze");
                try (InputStream in = getContentResolver().openInputStream(u);
                     OutputStream out = new FileOutputStream(f)) {
                    byte[] buf = new byte[1024 * 1024];
                    int n;
                    while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                }
                boolean ok = loadModel(f.getAbsolutePath());
                runOnUiThread(() -> status.setText(ok ? "Model ready" : "Model load failed"));
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Load error"));
            }
        });
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
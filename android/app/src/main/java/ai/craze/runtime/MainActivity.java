package ai.craze.runtime;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.concurrent.*;

public class MainActivity extends Activity {
    static { System.loadLibrary("crazeai"); }

    private native boolean loadModel(String path);
    private native String generate(String prompt, int maxTokens, float temperature);

    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView status;
    private ImageButton send;
    private TextView welcome;
    private TextView webToggle;
    private TextView thinkToggle;
    private boolean webSearch = false;
    private boolean deepThink = false;
    private static final int PICK = 42;

    private int dp(float v) { return (int)(v * getResources().getDisplayMetrics().density + 0.5f); }

    private GradientDrawable rounded(String color, float radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(Color.parseColor(color));
        d.setCornerRadius(dp(radius));
        return d;
    }

    private TextView label(String value, float size, int color) {
        TextView t = new TextView(this);
        t.setText(value);
        t.setTextSize(size);
        t.setTextColor(color);
        return t;
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.parseColor("#08090D"));
        getWindow().setNavigationBarColor(Color.parseColor("#08090D"));
        buildUi();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#08090D"));

        LinearLayout bar = new LinearLayout(this);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), dp(10), dp(10), dp(10));

        ImageView logo = new ImageView(this);
        logo.setImageResource(R.drawable.ic_crazeai_logo);
        bar.addView(logo, new LinearLayout.LayoutParams(dp(42), dp(42)));

        LinearLayout identity = new LinearLayout(this);
        identity.setOrientation(LinearLayout.VERTICAL);
        identity.setPadding(dp(11), 0, 0, 0);
        TextView title = label("CrazeAI", 19, Color.WHITE);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        identity.addView(title);
        status = label("Model not loaded", 12, Color.parseColor("#8F96A3"));
        identity.addView(status);
        bar.addView(identity, new LinearLayout.LayoutParams(0, -2, 1));

        TextView load = label("Load GGUF", 13, Color.WHITE);
        load.setGravity(Gravity.CENTER);
        load.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        load.setBackground(rounded("#20232B", 18));
        load.setPadding(dp(10), 0, dp(10), 0);
        load.setOnClickListener(v -> pickModel());
        bar.addView(load, new LinearLayout.LayoutParams(dp(92), dp(40)));
        root.addView(bar);

        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(16), dp(8), dp(16), dp(24));

        LinearLayout empty = new LinearLayout(this);
        empty.setOrientation(LinearLayout.VERTICAL);
        empty.setGravity(Gravity.CENTER_HORIZONTAL);
        welcome = label("How can I help?", 28, Color.WHITE);
        welcome.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        welcome.setGravity(Gravity.CENTER);
        empty.addView(welcome, new LinearLayout.LayoutParams(-1, -2));
        TextView sub = label("Local AI • Web search • Deep Think", 15, Color.parseColor("#8F96A3"));
        sub.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(-1, -2);
        sp.setMargins(0, dp(8), 0, dp(22));
        empty.addView(sub, sp);

        LinearLayout suggestions = new LinearLayout(this);
        suggestions.setOrientation(LinearLayout.VERTICAL);
        String[] prompts = {"Explain HTML simply", "Write a Python function", "Search the web for today's news"};
        for (String p : prompts) {
            TextView chip = label(p, 14, Color.parseColor("#D8DCE5"));
            chip.setGravity(Gravity.CENTER_VERTICAL);
            chip.setPadding(dp(16), 0, dp(16), 0);
            chip.setBackground(rounded("#151820", 16));
            chip.setOnClickListener(v -> { input.setText(((TextView)v).getText()); input.requestFocus(); });
            LinearLayout.LayoutParams cp = new LinearLayout.LayoutParams(-1, dp(48));
            cp.setMargins(0, 0, 0, dp(9));
            suggestions.addView(chip, cp);
        }
        empty.addView(suggestions, new LinearLayout.LayoutParams(-1, -2));
        messages.addView(empty, new LinearLayout.LayoutParams(-1, -2));
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));

        LinearLayout composer = new LinearLayout(this);
        composer.setOrientation(LinearLayout.VERTICAL);
        composer.setPadding(dp(12), dp(6), dp(12), dp(12));

        LinearLayout tools = new LinearLayout(this);
        tools.setGravity(Gravity.CENTER_VERTICAL);
        webToggle = toolButton("◎ Web Search: OFF");
        thinkToggle = toolButton("✦ Deep Think: OFF");
        tools.addView(webToggle, new LinearLayout.LayoutParams(0, dp(38), 1));
        tools.addView(thinkToggle, new LinearLayout.LayoutParams(0, dp(38), 1));
        composer.addView(tools);

        LinearLayout box = new LinearLayout(this);
        box.setGravity(Gravity.CENTER_VERTICAL);
        box.setPadding(dp(15), dp(2), dp(6), dp(2));
        box.setBackground(rounded("#171A21", 25));

        input = new EditText(this);
        input.setHint("Message CrazeAI");
        input.setHintTextColor(Color.parseColor("#707783"));
        input.setTextColor(Color.WHITE);
        input.setTextSize(16);
        input.setSingleLine(false);
        input.setMaxLines(4);
        input.setPadding(0, dp(7), 0, dp(7));
        input.setBackgroundColor(Color.TRANSPARENT);
        box.addView(input, new LinearLayout.LayoutParams(0, -2, 1));

        send = new ImageButton(this);
        send.setImageResource(android.R.drawable.ic_media_play);
        send.setColorFilter(Color.WHITE);
        send.setBackground(rounded("#8B5CF6", 22));
        send.setContentDescription("Send message");
        send.setOnClickListener(v -> generateAsync());
        box.addView(send, new LinearLayout.LayoutParams(dp(44), dp(44)));
        composer.addView(box);
        root.addView(composer);

        setContentView(root);
    }

    private TextView toolButton(String text) {
        TextView t = label(text, 12, Color.parseColor("#AEB5C2"));
        t.setGravity(Gravity.CENTER);
        t.setBackground(rounded("#12151B", 16));
        t.setOnClickListener(v -> {
            if (v == webToggle) {
                webSearch = !webSearch;
                webToggle.setText(webSearch ? "● Web Search: ON" : "◎ Web Search: OFF");
                webToggle.setTextColor(webSearch ? Color.WHITE : Color.parseColor("#AEB5C2"));
            } else {
                deepThink = !deepThink;
                thinkToggle.setText(deepThink ? "● Deep Think: ON" : "✦ Deep Think: OFF");
                thinkToggle.setTextColor(deepThink ? Color.WHITE : Color.parseColor("#AEB5C2"));
            }
        });
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, dp(38), 1);
        p.setMargins(0, 0, dp(6), 0);
        t.setLayoutParams(p);
        return t;
    }

    private void addMessage(String who, String text, boolean user) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.VERTICAL);
        row.setPadding(user ? dp(42) : 0, dp(5), user ? 0 : dp(42), dp(5));
        TextView name = label(who, 12, Color.parseColor("#8F96A3"));
        name.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        row.addView(name);
        TextView body = label(text, 16, Color.parseColor("#E8EAF0"));
        body.setTextIsSelectable(true);
        body.setPadding(dp(14), dp(11), dp(14), dp(11));
        body.setBackground(rounded(user ? "#252936" : "#11141A", 16));
        row.addView(body, new LinearLayout.LayoutParams(-1, -2));
        messages.addView(row);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void generateAsync() {
        final String prompt = input.getText().toString().trim();
        if (prompt.isEmpty() || !send.isEnabled()) return;
        final boolean useWeb = webSearch || looksLikeWebQuestion(prompt);
        final boolean useThink = deepThink;
        send.setEnabled(false);
        status.setText(useWeb ? "Searching…" : (useThink ? "Thinking deeply…" : "Thinking…"));
        welcome.setVisibility(View.GONE);
        input.setText("");
        ((InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE)).hideSoftInputFromWindow(input.getWindowToken(), 0);
        addMessage("You", prompt, true);
        addMessage("CrazeAI", useWeb ? "Searching the web…" : "Thinking…", false);
        final int answerIndex = messages.getChildCount() - 1;

        executor.execute(() -> {
            String answer;
            try {
                String webContext = useWeb ? webSearch(prompt) : "";
                String system = "You are CrazeAI, a helpful AI assistant created by CrazeStudio. " +
                        "Your underlying model is Qwen3-1.7B. Never claim to be trained from scratch. " +
                        "Be accurate and never invent facts. If web sources are supplied, use them and distinguish sourced facts from your own knowledge. " +
                        (useThink ? "Reason carefully before answering, check your assumptions, and provide only the useful final reasoning summary; never reveal hidden chain-of-thought. " : "") +
                        (useWeb ? "Web search was requested. Answer using the retrieved web context when relevant and include source titles/URLs in a Sources section. " : "") +
                        "\n\n" + webContext + "\n\nUser: " + prompt + "\nCrazeAI:";
                answer = generate(system, useThink ? 384 : 192, useThink ? 0.45f : 0.65f);
                if (answer == null || answer.trim().isEmpty()) answer = "I couldn't generate a response.";
            } catch (Throwable t) {
                answer = "Generation failed: " + (t.getMessage() == null ? "unknown error" : t.getMessage());
            }
            final String result = answer;
            runOnUiThread(() -> {
                LinearLayout row = (LinearLayout) messages.getChildAt(answerIndex);
                TextView body = (TextView) row.getChildAt(1);
                body.setText(result);
                status.setText("Ready");
                send.setEnabled(true);
            });
        });
    }

    private boolean looksLikeWebQuestion(String q) {
        String s = q.toLowerCase();
        return s.contains("today") || s.contains("latest") || s.contains("current") || s.contains("news") ||
               s.contains("right now") || s.contains("this week") || s.contains("search the web") || s.contains("look up");
    }

    private String webSearch(String query) throws Exception {
        String encoded = URLEncoder.encode(query, "UTF-8");
        URL url = new URL("https://api.duckduckgo.com/?q=" + encoded + "&format=json&no_html=1&skip_disambig=1");
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setConnectTimeout(8000);
        c.setReadTimeout(10000);
        c.setRequestMethod("GET");
        c.setRequestProperty("User-Agent", "CrazeAI/1.0");
        try (InputStream in = c.getInputStream(); BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
            StringBuilder b = new StringBuilder();
            String line;
            while ((line = r.readLine()) != null) b.append(line);
            return parseSearchResults(b.toString());
        } finally {
            c.disconnect();
        }
    }

    private String parseSearchResults(String json) {
        try {
            JSONObject root = new JSONObject(json);
            StringBuilder out = new StringBuilder("WEB SEARCH RESULTS (DuckDuckGo):\n");
            String abstractText = root.optString("AbstractText", "");
            String abstractUrl = root.optString("AbstractURL", "");
            String heading = root.optString("Heading", "");
            if (!abstractText.isEmpty()) {
                out.append("- ").append(heading).append(": ").append(abstractText).append("\nURL: ").append(abstractUrl).append("\n");
            }
            JSONArray topics = root.optJSONArray("RelatedTopics");
            int count = 0;
            if (topics != null) {
                for (int i = 0; i < topics.length() && count < 6; i++) {
                    JSONObject item = topics.optJSONObject(i);
                    if (item == null) continue;
                    String text = item.optString("Text", "");
                    String firstUrl = item.optString("FirstURL", "");
                    if (!text.isEmpty() && !firstUrl.isEmpty()) {
                        out.append("- ").append(text).append("\nURL: ").append(firstUrl).append("\n");
                        count++;
                    }
                }
            }
            if (out.toString().equals("WEB SEARCH RESULTS (DuckDuckGo):\n")) return "WEB SEARCH RESULTS: No useful results were returned. State uncertainty rather than inventing current facts.";
            return out.toString();
        } catch (Exception e) {
            return "WEB SEARCH RESULTS: Search parsing failed. Do not invent current information.";
        }
    }

    private void pickModel() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        startActivityForResult(i, PICK);
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        status.setText("Loading GGUF…");
        executor.execute(() -> {
            try {
                File f = new File(getFilesDir(), "crazeai.gguf");
                try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(f)) {
                    byte[] buf = new byte[1024 * 1024];
                    int n;
                    while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
                }
                boolean ok = loadModel(f.getAbsolutePath());
                runOnUiThread(() -> {
                    status.setText(ok ? "GGUF model ready" : "Model load failed");
                    if (ok) Toast.makeText(this, "CrazeAI is ready", Toast.LENGTH_SHORT).show();
                });
            } catch (Throwable e) {
                runOnUiThread(() -> status.setText("Model load failed"));
            }
        });
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}

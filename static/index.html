<!DOCTYPE html>
<html>
<head>
    <title>CrazeAI Web</title>
    <style>
        body { background:#0d0d0d; color:white; font-family:Arial; padding:20px; }
        input { width:70%; padding:12px; border-radius:8px; border:none; }
        button { padding:12px 20px; border:none; background:#00b7ff; color:white; border-radius:8px; }
        .box { background:#1a1a1a; padding:12px; margin:10px 0; border-radius:8px; }
        a { color:#00b7ff; }
    </style>
</head>
<body>
    <h1>CrazeAI</h1>
    <input id="msg" placeholder="Ask anything..." />
    <button onclick="send()">Ask</button>

    <h2>Answer:</h2>
    <pre id="answer"></pre>

    <h2>Web Results:</h2>
    <div id="results"></div>

<script>
async function send() {
    const msg = document.getElementById("msg").value;

    const res = await fetch("/api/crazeai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    });

    const data = await res.json();

    document.getElementById("answer").innerText = data.answer;

    const resultsBox = document.getElementById("results");
    resultsBox.innerHTML = "";

    data.results.forEach(r => {
        const div = document.createElement("div");
        div.className = "box";
        div.innerHTML = `
            <b>${r.title}</b><br>
            <p>${r.snippet}</p>
            <a href="${r.url}" target="_blank">Open</a> — ${r.source}
        `;
        resultsBox.appendChild(div);
    });
}
</script>
</body>
</html>

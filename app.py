"""
ML AI App — Flask backend
"""
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from brain import brain, init_brain
from collections import deque

app = Flask(__name__)
CORS(app)

# Session memory
_sessions: dict[str, deque] = {}
MEMORY = 8

def get_session(sid):
    if sid not in _sessions:
        _sessions[sid] = deque(maxlen=MEMORY)
    return _sessions[sid]

# Math solver
def try_math(text):
    expr = re.sub(r"[^\d\s\+\-\*\/\.\(\)]", "", text).strip()
    expr = expr.replace("x","*").replace("×","*")
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expr) or len(expr) < 3:
        return None
    try:
        result = eval(expr)
        result = int(result) if isinstance(result, float) and result == int(result) else round(result, 6)
        return f"= {result}"
    except:
        return None

# Web search
def web_search(query):
    try:
        import requests, urllib.parse
        url  = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        data = requests.get(url, timeout=3, headers={"User-Agent":"Mozilla/5.0"}).json()
        if data.get("AbstractText"):
            return data["AbstractText"], data.get("AbstractSource","Web"), data.get("AbstractURL","")
        if data.get("Answer"):
            return data["Answer"], "DuckDuckGo", ""
    except:
        pass
    try:
        import requests, urllib.parse, re as _re
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        res = requests.get(url, timeout=4, headers={"User-Agent":"Mozilla/5.0"})
        snippets = _re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', res.text, _re.S)
        snippets = [_re.sub(r'<[^>]+>','',s).strip() for s in snippets[:1] if len(s)>20]
        if snippets:
            return snippets[0], "DuckDuckGo", ""
    except:
        pass
    return None, None, None

# Teach parser
def parse_teach(text):
    patterns = [
        r"remember (?:that )?(.+?) (?:is|=|means) (.+)",
        r"learn[:\s]+(.+?) (?:is|=|means) (.+)",
        r"the answer to (.+?) is (.+)",
        r"know that (.+?) (?:is|=|means) (.+)",
    ]
    for p in patterns:
        m = re.match(p, text.strip(), re.I)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None

def is_question(text):
    t = text.lower().strip()
    return t.endswith("?") or any(t.startswith(w) for w in
        ["what","who","when","where","why","how","can you","is ","does ","did ","tell me"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "trained": brain.trained,
        "patterns": brain.knowledge_count(),
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json() or {}
    user  = (data.get("message") or "").strip()
    sid   = data.get("session_id") or "default"
    if not user:
        return jsonify({"error": "Empty"}), 400

    mem = get_session(sid)
    used_web = False
    source   = None

    # ── Teach ──
    teach_result = parse_teach(user)
    if teach_result:
        q, a = teach_result
        reply = brain.learn(q, a)
        mem.append({"role":"user","text":user})
        mem.append({"role":"ai","text":reply})
        return jsonify({"reply": reply, "confidence": 1.0, "tag": "learned", "web": False})

    # ── Clear ──
    if user.lower().strip() in ("clear memory","reset","start over","forget everything"):
        _sessions.pop(sid, None)
        return jsonify({"reply": "🧹 Memory cleared!", "confidence": 1.0, "tag": "system", "web": False})

    # ── Math ──
    math_result = try_math(user)
    if math_result:
        return jsonify({"reply": math_result, "confidence": 1.0, "tag": "math", "web": False})

    # ── ML Brain predict ──
    response, confidence, tag = brain.predict(user)

    # ── Web search fallback for questions ──
    if response is None and is_question(user):
        text, src, url = web_search(user)
        if text:
            response  = text
            used_web  = True
            source    = f"{src}||{url}" if url else src

    # ── Final fallback ──
    if response is None:
        if confidence < 0.2:
            response = "I'm not sure about that yet. You can teach me by saying: \"remember that [question] is [answer]\""
        else:
            response = "Hmm, I didn't quite get that. Could you rephrase?"

    mem.append({"role":"user","text":user})
    mem.append({"role":"ai","text":response})

    return jsonify({
        "reply":      response,
        "confidence": round(confidence, 3),
        "tag":        tag,
        "web":        used_web,
        "source":     source,
    })

@app.route("/api/retrain", methods=["POST"])
def retrain():
    brain.train()
    brain.save()
    return jsonify({"ok": True, "patterns": brain.knowledge_count()})

if __name__ == "__main__":
    init_brain()
    print("🚀 ML AI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

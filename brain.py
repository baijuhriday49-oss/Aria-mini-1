"""
AI Brain V2 — Pure Python, Zero Dependencies
Implements from scratch:
  - TF-IDF vectorization
  - Cosine similarity
  - Intent classification
  - Learn on the fly
No numpy. No scikit-learn. Just Python.
"""

import json, re, math, pickle, random
from pathlib import Path
from collections import defaultdict

DATA_PATH  = Path(__file__).parent / "knowledge.json"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# ── KNOWLEDGE BASE ────────────────────────────────────────────
DEFAULT_KNOWLEDGE = [
  {"input": "hello", "output": "Hey! How can I help you?", "tag": "greeting"},
  {"input": "hi", "output": "Hi there! What's on your mind?", "tag": "greeting"},
  {"input": "hey", "output": "Hello! What can I do for you?", "tag": "greeting"},
  {"input": "good morning", "output": "Good morning! Hope you're having a great day! ☀️", "tag": "greeting"},
  {"input": "good evening", "output": "Good evening! How can I assist?", "tag": "greeting"},
  {"input": "good night", "output": "Good night! Sleep well! 🌙", "tag": "greeting"},
  {"input": "how are you", "output": "I'm doing great, thanks for asking! How about you?", "tag": "greeting"},
  {"input": "what is your name", "output": "I'm TinyAI — a pure Python ML chatbot with no external dependencies!", "tag": "identity"},
  {"input": "who are you", "output": "I'm your local AI assistant! I use TF-IDF and cosine similarity to understand you.", "tag": "identity"},
  {"input": "what can you do", "output": "I can answer questions, have conversations, learn new things, do math, and search the web!", "tag": "identity"},
  {"input": "are you smart", "output": "I use real ML techniques — TF-IDF vectors and cosine similarity. For a pure Python bot, pretty smart! 😄", "tag": "identity"},
  {"input": "bye", "output": "Goodbye! Take care! 👋", "tag": "farewell"},
  {"input": "goodbye", "output": "See you later! Have a great day! 👋", "tag": "farewell"},
  {"input": "see you later", "output": "See you! 👋", "tag": "farewell"},
  {"input": "thank you", "output": "You're welcome! Glad I could help! 😊", "tag": "thanks"},
  {"input": "thanks", "output": "No problem at all! 😊", "tag": "thanks"},
  {"input": "what is artificial intelligence", "output": "AI is the simulation of human intelligence by machines — learning, reasoning, problem-solving. I'm a tiny example!", "tag": "tech"},
  {"input": "what is machine learning", "output": "ML is a type of AI where models learn from data. I use TF-IDF + cosine similarity — real ML techniques!", "tag": "tech"},
  {"input": "what is python", "output": "Python is a high-level programming language great for AI, web, and automation. I'm built entirely in Python!", "tag": "tech"},
  {"input": "what is a neural network", "output": "A neural network is a system of layers that learn patterns from data, inspired by the human brain.", "tag": "tech"},
  {"input": "what is deep learning", "output": "Deep learning uses neural networks with many layers. It powers image recognition, language models like ChatGPT, and more.", "tag": "tech"},
  {"input": "what is an api", "output": "An API lets applications communicate. REST APIs use HTTP methods like GET and POST.", "tag": "tech"},
  {"input": "what is javascript", "output": "JavaScript is the language of the web — makes websites interactive and runs on servers via Node.js.", "tag": "tech"},
  {"input": "what is html", "output": "HTML (HyperText Markup Language) defines web page structure using tags like <h1>, <p>, <div>.", "tag": "tech"},
  {"input": "what is css", "output": "CSS (Cascading Style Sheets) styles HTML — colors, fonts, layouts, animations.", "tag": "tech"},
  {"input": "what is git", "output": "Git is a version control system tracking code changes. GitHub hosts repositories online.", "tag": "tech"},
  {"input": "what is a database", "output": "A database stores organized data. SQL uses tables, NoSQL uses flexible documents.", "tag": "tech"},
  {"input": "what is tfidf", "output": "TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures how important a word is in a document relative to a collection. I use it to understand your messages!", "tag": "tech"},
  {"input": "what is cosine similarity", "output": "Cosine similarity measures the angle between two vectors. If the angle is 0°, they're identical. I use it to find the best matching response to your input!", "tag": "tech"},
  {"input": "what is gravity", "output": "Gravity attracts objects with mass. Earth's gravity pulls at 9.8 m/s². Einstein described it as the curvature of spacetime.", "tag": "science"},
  {"input": "what is the speed of light", "output": "Light travels at ~299,792 km/s in a vacuum — the universal speed limit.", "tag": "science"},
  {"input": "what is dna", "output": "DNA (Deoxyribonucleic Acid) carries genetic information in all living things in a double helix structure.", "tag": "science"},
  {"input": "what is a black hole", "output": "A black hole is a region where gravity is so strong not even light escapes. They form when massive stars collapse.", "tag": "science"},
  {"input": "what is photosynthesis", "output": "Plants convert sunlight, CO₂ and water into glucose and oxygen using chlorophyll — that's photosynthesis!", "tag": "science"},
  {"input": "what is the big bang", "output": "The Big Bang: ~13.8 billion years ago everything expanded from an infinitely dense point. The universe is still expanding!", "tag": "science"},
  {"input": "what is evolution", "output": "Species change over generations through natural selection — first described by Charles Darwin in 1859.", "tag": "science"},
  {"input": "what is quantum physics", "output": "Quantum physics describes subatomic scale. Particles exist in superposition — multiple states at once until observed!", "tag": "science"},
  {"input": "what is electricity", "output": "Electricity is the flow of electric charge (electrons) through a conductor, powering all our devices.", "tag": "science"},
  {"input": "what is pi", "output": "Pi (π) ≈ 3.14159265... The ratio of a circle's circumference to diameter. Irrational — never repeats!", "tag": "math"},
  {"input": "what is the pythagorean theorem", "output": "a² + b² = c². The square of the hypotenuse equals the sum of squares of the other two sides.", "tag": "math"},
  {"input": "what is a prime number", "output": "A prime is divisible only by 1 and itself. Examples: 2, 3, 5, 7, 11, 13. Only even prime is 2.", "tag": "math"},
  {"input": "what is fibonacci", "output": "0, 1, 1, 2, 3, 5, 8, 13, 21... Each number is the sum of the two before it. Found throughout nature!", "tag": "math"},
  {"input": "how to make tea", "output": "Boil water, steep tea bag 3-5 min. Add milk, sugar, or lemon. For chai: boil with ginger and cardamom!", "tag": "food"},
  {"input": "how to boil an egg", "output": "In cold water, bring to boil. Soft: 6 min. Hard: 10-12 min. Ice water immediately after to stop cooking.", "tag": "food"},
  {"input": "how to make rice", "output": "1:2 ratio (rice:water). Boil, reduce heat, cover 18 min. Rest 5 min before fluffing.", "tag": "food"},
  {"input": "tell me a joke", "output": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛", "tag": "fun"},
  {"input": "another joke", "output": "Why did the Python developer go broke? He kept using recursion without a base case! 💀", "tag": "fun"},
  {"input": "do you have feelings", "output": "Not really — I'm pure Python code. But I'm designed to understand yours! 🤖", "tag": "fun"},
  {"input": "what is the meaning of life", "output": "42 — according to The Hitchhiker's Guide! Philosophically speaking, humanity still debates this. 😄", "tag": "fun"},
]

# ── PURE PYTHON TF-IDF ────────────────────────────────────────
def tokenize(text: str) -> list:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]

def compute_tfidf(corpus: list) -> tuple:
    """Returns (tfidf_matrix, vocab) where tfidf_matrix[i] = dict of {word: score}"""
    N = len(corpus)
    tokenized = [tokenize(doc) for doc in corpus]

    # Document frequency
    df = defaultdict(int)
    for tokens in tokenized:
        for word in set(tokens):
            df[word] += 1

    # Build vocab
    vocab = sorted(df.keys())
    word_idx = {w: i for i, w in enumerate(vocab)}

    # TF-IDF matrix as list of dicts
    matrix = []
    for tokens in tokenized:
        tf = defaultdict(float)
        for word in tokens:
            tf[word] += 1
        total = len(tokens) or 1
        vec = {}
        for word, count in tf.items():
            if word in word_idx:
                tfidf = (count / total) * math.log((N + 1) / (df[word] + 1))
                vec[word] = tfidf
        matrix.append(vec)

    return matrix, vocab, word_idx, df, N

def vectorize(text: str, word_idx: dict, df: dict, N: int, total_docs: int) -> dict:
    """Convert new text to TF-IDF vector using existing vocab."""
    tokens = tokenize(text)
    tf = defaultdict(float)
    for word in tokens:
        tf[word] += 1
    total = len(tokens) or 1
    vec = {}
    for word, count in tf.items():
        if word in word_idx:
            tfidf = (count / total) * math.log((total_docs + 1) / (df.get(word, 0) + 1))
            vec[word] = tfidf
    return vec

def cosine_sim(v1: dict, v2: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    dot = sum(v1.get(w, 0) * v2.get(w, 0) for w in v1)
    mag1 = math.sqrt(sum(x*x for x in v1.values()))
    mag2 = math.sqrt(sum(x*x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ── TINY BRAIN ────────────────────────────────────────────────
class TinyBrain:
    def __init__(self):
        self.knowledge  = []
        self.inputs     = []
        self.outputs    = []
        self.tags       = []
        self.matrix     = []
        self.vocab      = []
        self.word_idx   = {}
        self.df         = {}
        self.N          = 0
        self.trained    = False

    def train(self, knowledge=None):
        if knowledge:
            self.knowledge = knowledge
        else:
            self.knowledge = load_knowledge()

        self.inputs  = [k["input"]  for k in self.knowledge]
        self.outputs = [k["output"] for k in self.knowledge]
        self.tags    = [k["tag"]    for k in self.knowledge]
        self.N       = len(self.inputs)

        if self.N < 2:
            return False

        self.matrix, self.vocab, self.word_idx, self.df, _ = compute_tfidf(self.inputs)
        self.trained = True
        return True

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                self.__dict__.update(pickle.load(f))
            return True
        except:
            return False

    def predict(self, user_input: str) -> tuple:
        """Returns (response, confidence, tag)"""
        if not self.trained:
            return ("I'm not trained yet!", 0.0, "unknown")

        vec = vectorize(user_input, self.word_idx, self.df, self.N, self.N)
        if not vec:
            return (None, 0.0, "unknown")

        # Score all knowledge entries
        scores = [(cosine_sim(vec, self.matrix[i]), i) for i in range(self.N)]
        scores.sort(reverse=True)

        best_score, best_idx = scores[0]

        if best_score >= 0.7:
            return (self.outputs[best_idx], best_score, self.tags[best_idx])
        elif best_score >= 0.35:
            return (f"I think you're asking about this:\n\n{self.outputs[best_idx]}", best_score, self.tags[best_idx])
        else:
            return (None, best_score, "unknown")

    def learn(self, user_input: str, response: str, tag: str = "learned") -> str:
        knowledge = load_knowledge()
        norm = user_input.lower().strip()
        for item in knowledge:
            if item["input"].lower().strip() == norm:
                item["output"] = response
                save_knowledge(knowledge)
                self.train(knowledge)
                self.save()
                return f'✅ Updated: "{user_input}" → "{response}"'
        knowledge.append({"input": user_input, "output": response, "tag": tag})
        save_knowledge(knowledge)
        self.train(knowledge)
        self.save()
        return f'✅ Learned: "{user_input}" → "{response}"'

    def knowledge_count(self) -> int:
        return len(self.inputs)


# ── DATA HELPERS ──────────────────────────────────────────────
def load_knowledge() -> list:
    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    data = DEFAULT_KNOWLEDGE.copy()
    save_knowledge(data)
    return data

def save_knowledge(data: list):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── GLOBAL INSTANCE ───────────────────────────────────────────
brain = TinyBrain()

def init_brain():
    if brain.load():
        print(f"🧠 Model loaded ({brain.knowledge_count()} patterns)")
    else:
        print("🏋️  Training from scratch...")
        brain.train()
        brain.save()
        print(f"✅ Ready! ({brain.knowledge_count()} patterns)")

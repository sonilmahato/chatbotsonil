from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load CSV data ===
df = pd.read_csv("university_data.csv")
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()

questions = df['question'].tolist()
answers = df['answer'].tolist()

# === Embedding & FAISS Setup ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embed_model.encode(questions).astype('float32')
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# === Load FLAN-T5 for RAQG ===
qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# === Normalize ===
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# === Exact Match ===
def is_exact_match(query):
    nq = normalize(query)
    for i, q in enumerate(questions):
        if normalize(q) == nq:
            return True, answers[i]
    return False, None

# === Semantic Suggestions via FAISS ===
def get_semantic_matches(query, top_k=5):
    query_vec = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, top_k)
    suggested = [questions[i] for i in indices[0] if i < len(questions)]
    return suggested

# === Generate Follow-up Questions (RAQG) ===
def generate_followups(query):
    prompt = f"Suggest 3 follow-up university questions for: {query}"
    out = qg_pipeline(prompt, max_length=60, num_return_sequences=1)[0]['generated_text']
    return out.strip().split("\n")

# === Flask App ===
app = Flask(__name__)

# === Frontend Template ===
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>University Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #eef2f7;
            font-family: 'Segoe UI', sans-serif;
        }
        .chat-container {
            display: flex;
            justify-content: center;
            padding: 60px 20px;
        }
        .chat-card {
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            padding: 40px;
            width: 100%;
            max-width: 900px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
        }
        .chat-title {
            font-size: 28px;
            font-weight: 600;
            color: #004085;
            text-align: center;
        }
        .form-control {
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
        }
        .btn-primary {
            border-radius: 10px;
            font-weight: 500;
            padding: 10px 20px;
        }
        .response-section, .suggestion-section, .followup-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
        }
        .section-title {
            font-weight: bold;
            color: #343a40;
            margin-bottom: 12px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-card">
            <div class="chat-title">🎓 UEL University Chatbot</div>
            <form method="post">
                <div class="mb-3">
                    <input name="query" class="form-control" placeholder="Type your question here..." required>
                </div>
                <button type="submit" class="btn btn-primary w-100">Submit</button>
            </form>

            {% if response %}
            <div class="response-section">
                <div class="section-title">🤖 Response:</div>
                <div>{{ response }}</div>
            </div>
            {% endif %}

            {% if suggestions %}
            <div class="suggestion-section">
                <div class="section-title">🔎 Similar Questions:</div>
                <ul>
                    {% for q in suggestions %}
                    <li>{{ q }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}

            {% if followups %}
            <div class="followup-section">
                <div class="section-title">💡 Follow-up Suggestions:</div>
                <ul>
                    {% for q in followups %}
                    <li>{{ q }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
    </div>

    <div class="footer">
        &copy; 2025 C P Budha Magar | <a href="mailto:budhacpmagar2@gmail.com">Contact Developer</a>
    </div>
</body>
</html>
'''

# === Route ===
@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    suggestions = []
    followups = []

    if request.method == "POST":
        query = request.form["query"]
        match, answer = is_exact_match(query)

        if match:
            response = answer + " Would you like to ask another question?"
        else:
            response = "I couldn’t find an exact match. Here are related suggestions."
            suggestions = get_semantic_matches(query)
            followups = generate_followups(query)

    return render_template_string(HTML, response=response, suggestions=suggestions, followups=followups)

# === Run App ===
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

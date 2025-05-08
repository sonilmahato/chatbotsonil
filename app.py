from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load CSV ===
df = pd.read_csv("university_chatbot_data.csv")
df.columns = [col.lower().strip() for col in df.columns]
if 'question' not in df.columns or 'answer' not in df.columns:
    raise ValueError("CSV must contain 'question' and 'answer' columns.")

df = df.dropna(subset=['question', 'answer'])
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()

questions = df['question'].tolist()
answers = df['answer'].tolist()

# === Load Embeddings & Index ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embed_model.encode(questions)
question_embeddings = np.array(question_embeddings).astype('float32')

index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# === Load RAQG Generator ===
qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# === Utility Functions ===
def normalize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

def is_exact_match(query):
    clean_query = query.strip().lower()
    for i, q in enumerate(questions):
        if clean_query == q.strip().lower() or normalize(clean_query) == normalize(q):
            return True, q, answers[i]
    return False, None, None

def get_related_questions(query, limit=5):
    query_keywords = set(normalize(query).split())
    scored_questions = []

    for q in questions:
        q_keywords = set(normalize(q).split())
        overlap = query_keywords.intersection(q_keywords)
        if overlap:
            score = len(overlap) / len(q_keywords)
            scored_questions.append((score, q))

    sorted_qs = sorted(scored_questions, key=lambda x: x[0], reverse=True)
    seen = set()
    unique_qs = []
    for _, q in sorted_qs:
        if q not in seen:
            unique_qs.append(q)
            seen.add(q)
        if len(unique_qs) >= limit:
            break
    return unique_qs

def generate_followups(query, num_return_sequences=3):
    prompt = f"Suggest follow-up questions for: '{query}'"
    outputs = qg_pipeline(prompt, max_length=60, num_return_sequences=num_return_sequences)
    return [o['generated_text'].strip() for o in outputs]

# === Flask App ===
app = Flask(__name__)

# === UI Template ===
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>University Chatbot</title>
</head>
<body>
    <h2>🎓 University Chatbot</h2>
    <form method="post">
        <input name="query" placeholder="Ask a question..." style="width:300px;" required>
        <button type="submit">Submit</button>
    </form>
    {% if response %}
    <p><strong>Bot:</strong> {{ response }}</p>
    {% endif %}
    {% if suggestions %}
    <p><strong>Related questions:</strong></p>
    <ul>{% for q in suggestions %}<li>{{ q }}</li>{% endfor %}</ul>
    {% endif %}
    {% if followups %}
    <p><strong>Follow-up ideas:</strong></p>
    <ul>{% for f in followups %}<li>{{ f }}</li>{% endfor %}</ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    suggestions = []
    followups = []
    if request.method == "POST":
        query = request.form["query"]
        matched, matched_q, matched_a = is_exact_match(query)
        if matched:
            response = matched_a
        else:
            suggestions = get_related_questions(query)
            if not suggestions:
                followups = generate_followups(query)
            response = "I couldn't find an exact answer."
    return render_template_string(HTML, response=response, suggestions=suggestions, followups=followups)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

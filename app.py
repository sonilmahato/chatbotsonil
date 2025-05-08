from flask import Flask, request, render_template_string
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

app = Flask(__name__)

# Load CSV
df = pd.read_csv("university_data.csv")
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Load Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
question_embeddings = embed_model.encode(questions)

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

def is_exact_match(query):
    q_norm = normalize(query)
    for i, q in enumerate(questions):
        if normalize(q) == q_norm:
            return True, answers[i]
    return False, None

def get_top_matches(query, top_k=5):
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, question_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [questions[i] for i in top_indices if sims[i] > 0.5]

def generate_followups(query, n=3):
    prompt = f"Suggest follow-up questions for: '{query}'"
    result = qg_pipeline(prompt, max_length=60, num_return_sequences=n)
    return [r['generated_text'] for r in result]

# Minimal HTML template
HTML = """
<!DOCTYPE html>
<html>
<head><title>University Chatbot</title></head>
<body>
    <h2>🎓 University Chatbot</h2>
    <form method="post">
        <input name="query" required style="width:300px;" placeholder="Ask a question...">
        <button type="submit">Ask</button>
    </form>
    {% if response %}
        <p><b>Bot:</b> {{ response }}</p>
    {% endif %}
    {% if suggestions %}
        <p><b>Related:</b></p><ul>{% for s in suggestions %}<li>{{ s }}</li>{% endfor %}</ul>
    {% endif %}
    {% if followups %}
        <p><b>Follow-up ideas:</b></p><ul>{% for f in followups %}<li>{{ f }}</li>{% endfor %}</ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chatbot():
    response = ""
    suggestions = []
    followups = []
    if request.method == "POST":
        query = request.form["query"]
        matched, answer = is_exact_match(query)
        if matched:
            response = answer
        else:
            suggestions = get_top_matches(query)
            if not suggestions:
                followups = generate_followups(query)
            response = "I couldn't find an exact answer."
    return render_template_string(HTML, response=response, suggestions=suggestions, followups=followups)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, render_template_string
import pandas as pd
import re
import os

app = Flask(__name__)

# === Load CSV and Build QA Dictionary ===
df = pd.read_csv(os.path.join("data", "university_data.csv")
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()
qa_pairs = dict(zip(df['question'], df['answer']))
all_questions = list(qa_pairs.keys())

# === Preprocess: Normalize + Tokenize ===
def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    return set(text.split())

# === Jaccard Similarity for Match & Ranking ===
def jaccard_similarity(a, b):
    set_a = preprocess(a)
    set_b = preprocess(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# === Matching Function ===
def find_answer(user_query):
    for question, answer in qa_pairs.items():
        if preprocess(user_query) == preprocess(question):
            return True, answer
    return False, None

# === Rank Related Questions by Similarity Score ===
def get_ranked_suggestions(user_query, top_n=5):
    scores = [(q, jaccard_similarity(user_query, q)) for q in all_questions]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [q for q, score in scores if score > 0][:top_n]

# === HTML Template ===
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>University Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f2f4f8;
            font-family: 'Segoe UI', sans-serif;
            padding-top: 60px;
        }
        .container {
            max-width: 680px;
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .chat-title {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            color: #0056b3;
            margin-bottom: 25px;
        }
        .form-control {
            border-radius: 12px;
            padding: 14px;
            font-size: 18px;
        }
        .btn-primary {
            border-radius: 12px;
            padding: 12px;
            font-size: 16px;
        }
        .response, .suggestions {
            margin-top: 25px;
        }
        .suggestions ul {
            padding-left: 20px;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="chat-title">University Chatbot</div>
    <form method="post">
        <div class="mb-3">
            <input name="query" class="form-control" placeholder="Type your question..." required>
        </div>
        <button type="submit" class="btn btn-primary w-100">Ask</button>
    </form>

    {% if response %}
    <div class="response alert alert-success">
        <strong>Bot:</strong> {{ response }}
    </div>
    {% endif %}

    {% if suggestions %}
    <div class="suggestions alert alert-info">
        <strong>Suggested questions:</strong>
        <ul>
        {% for q in suggestions %}
            <li>{{ q }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
</div>

<div class="footer">
    Designed by C P Budha Magar – AI/ML Developer
</div>
</body>
</html>
'''

# === Route ===
@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    suggestions = []
    if request.method == "POST":
        query = request.form.get("query", "")
        match, answer = find_answer(query)
        if match:
            response = answer + " Would you like to ask another question?"
        else:
            response = "Sorry, I couldn't find an exact answer."
            suggestions = get_ranked_suggestions(query)
    return render_template_string(HTML, response=response, suggestions=suggestions)

# === Render-Compatible Entry ===
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

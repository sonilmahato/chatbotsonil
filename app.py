from flask import Flask, request, render_template_string
import pandas as pd
import re

# === Load CSV ===
df = pd.read_csv("university_data.csv")
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()
questions = df['question'].tolist()
answers = df['answer'].tolist()

# === Initialize App ===
app = Flask(__name__)

# === Text Normalizer ===
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# === Exact Match Function ===
def is_exact_match(query):
    nq = normalize(query)
    for i, q in enumerate(questions):
        if normalize(q) == nq:
            return True, answers[i]
    return False, None

# === Related Questions (Substring Matching) ===
def get_related_questions(query, limit=10):
    keywords = normalize(query).split()
    related = []
    seen = set()
    for q in questions:
        q_norm = normalize(q)
        if any(k in q_norm for k in keywords) and q not in seen:
            related.append(q)
            seen.add(q)
    return related[:limit]

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

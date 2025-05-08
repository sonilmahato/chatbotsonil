from flask import Flask, request, render_template_string
import pandas as pd
import re

app = Flask(__name__)

# Load CSV
df = pd.read_csv("university_data.csv")
questions = df['question'].astype(str).str.strip().tolist()
answers = df['answer'].astype(str).str.strip().tolist()

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def is_exact_match(query):
    nq = normalize(query)
    for i, q in enumerate(questions):
        if normalize(q) == nq:
            return True, answers[i]
    return False, None

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

HTML = '''
<!DOCTYPE html>
<html>
<head><title>University Chatbot</title></head>
<body>
    <h2>🎓 University Chatbot</h2>
    <form method="post">
        <input name="query" placeholder="Ask your question..." required>
        <button type="submit">Ask</button>
    </form>
    {% if response %}
        <p><strong>Bot:</strong> {{ response }}</p>
    {% endif %}
    {% if suggestions %}
        <p><strong>Related questions:</strong></p>
        <ul>
            {% for q in suggestions %}
                <li>{{ q }}</li>
            {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    suggestions = []
    if request.method == "POST":
        query = request.form["query"]
        match, answer = is_exact_match(query)
        if match:
            response = answer + " Would you like to ask another question?"
        else:
            response = "I couldn’t find an exact answer."
            suggestions = get_related_questions(query)
    return render_template_string(HTML, response=response, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

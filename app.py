from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained question-answering model from Hugging Face
qa_pipeline = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad")

# Route for the homepage where users will interact with the chatbot
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submissions and return the answer from the model
@app.route('/get_answer', methods=['POST'])
def get_answer():
    context = request.form['context']  # Get the context (input text)
    question = request.form['question']  # Get the user's question
    # Get the model's answer
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    
    # Return the rendered template with the answer, context, and question
    return render_template('index.html', answer=answer, context=context, question=question)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Expanded in-memory storage for symbols
symbols = [
    {"id": 1, "name": "Hello", "image_url": "/static/images/hello.png", "category": "Greetings"},
    {"id": 2, "name": "Goodbye", "image_url": "/static/images/goodbye.png", "category": "Greetings"},
    {"id": 3, "name": "Yes", "image_url": "/static/images/yes.png", "category": "Basic Responses"},
    {"id": 4, "name": "No", "image_url": "/static/images/no.png", "category": "Basic Responses"},
    {"id": 5, "name": "Help", "image_url": "/static/images/help.png", "category": "Basic Needs"},
    {"id": 6, "name": "Hungry", "image_url": "/static/images/hungry.png", "category": "Basic Needs"},
    {"id": 7, "name": "Thirsty", "image_url": "/static/images/thirsty.png", "category": "Basic Needs"},
    {"id": 8, "name": "Happy", "image_url": "/static/images/happy.png", "category": "Emotions"},
    {"id": 9, "name": "Sad", "image_url": "/static/images/sad.png", "category": "Emotions"},
    {"id": 10, "name": "Angry", "image_url": "/static/images/angry.png", "category": "Emotions"},
]

@app.route('/')
def index():
    return render_template('board.html')

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    return jsonify(symbols)

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = list(set(symbol['category'] for symbol in symbols))
    return jsonify(categories)

@app.route('/api/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    return jsonify({"text": text})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import numpy as np
import cv2
import random

app = Flask(__name__)

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def get_feedback(target_emotion, detected_emotion, confidence):
    if target_emotion == detected_emotion:
        return f"Đúng rồi, làm tốt lắm! Bạn đã thể hiện cảm xúc {target_emotion} rất chính xác."
    else:
        tips = {
            'angry': "Bạn cần nhíu mày và có vẻ mặt căng thẳng hơn.",
            'disgust': "Hãy nhăn mũi và có vẻ mặt khó chịu hơn.",
            'fear': "Mở to mắt và có vẻ mặt lo lắng hơn.",
            'happy': "Hãy cười tươi hơn nữa, để lộ răng nếu có thể.",
            'sad': "Hãy cúi mặt xuống và có vẻ buồn bã hơn.",
            'surprise': "Mở to mắt và miệng, có vẻ ngạc nhiên hơn.",
            'neutral': "Giữ khuôn mặt thư giãn, không biểu lộ cảm xúc."
        }
        return f"Gần đúng rồi! {tips[target_emotion]} Hệ thống nhận diện bạn đang thể hiện cảm xúc {detected_emotion}."

@app.route('/')
def index():
    return render_template('emotion_game.html')

@app.route('/get_random_emotion')
def get_random_emotion():
    return jsonify({'emotion': random.choice(emotions)})

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    data = request.json
    image_data = data['image'].split(',')[1]
    target_emotion = data['target_emotion']

    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    detected_emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][detected_emotion]

    feedback = get_feedback(target_emotion, detected_emotion, confidence)

    return jsonify({
        'detected_emotion': detected_emotion,
        'confidence': confidence,
        'feedback': feedback
    })

if __name__ == '__main__':
    app.run(debug=True)
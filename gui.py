from flask import Flask, render_template, jsonify, request, Response, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS
import google.generativeai as genai
import os
import cv2
import mediapipe as mp
import numpy as np
import random
from deepface import DeepFace
import base64
import threading
import time

app = Flask(__name__)
CORS(app)

# Fake user data
fake_user = {
    "username": "Tran Duc Bo",
    "level": 3,
    "progress": 65.0
}

# Fake feedback data
fake_feedbacks = [
    {
        "game_type": "Emotion Game",
        "content": "Great job recognizing happiness and surprise! Try to focus more on distinguishing between anger and frustration.",
        "timestamp": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "game_type": "Chat Game",
        "content": "You're making good progress with greetings and small talk. Let's work on more complex conversation scenarios next time.",
        "timestamp": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "game_type": "Action Recognition",
        "content": "Excellent work on identifying basic actions! We'll introduce more nuanced gestures in the next session.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
]

# Fake progress data
fake_progress_data = {
    "emotion_game": [20, 35, 50, 65],
    "chat_game": [15, 30, 45, 60],
    "action_recognition": [25, 40, 55, 70],
    "image_reader": [10, 25, 40, 55]
}

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


# MediaPipe initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define actions and their feedback
actions = {
    "raise your hand": {
        "instruction": "Raise either of your hands above your head.",
        "feedback": {
            "not_high_enough": "Try raising your hand higher, above your head.",
            "wrong_body_part": "Make sure you're raising your hand, not another body part.",
            "almost": "You're close! Just a little higher.",
            "success": "Excellent! Your hand is perfectly raised."
        }
    },
    "wave": {
        "instruction": "Wave your hand from side to side.",
        "feedback": {
            "not_moving": "Try moving your hand from side to side more.",
            "wrong_body_part": "Focus on moving your hand, not your entire arm.",
            "almost": "You're getting there! Try a wider motion.",
            "success": "Great waving! You've got the hang of it."
        }
    },
    "shrug your shoulders": {
        "instruction": "Lift your shoulders towards your ears.",
        "feedback": {
            "not_high_enough": "Try lifting your shoulders higher, towards your ears.",
            "wrong_body_part": "Make sure you're moving your shoulders, not just your arms.",
            "almost": "Almost there! Just a bit higher with your shoulders.",
            "success": "Perfect shrug! You've mastered the shoulder shrug."
        }
    },
    "touch your nose": {
        "instruction": "Bring one of your hands to touch your nose.",
        "feedback": {
            "not_close_enough": "Move your hand closer to your nose.",
            "wrong_body_part": "Aim for your nose, not another part of your face.",
            "almost": "You're very close! Just a little adjustment needed.",
            "success": "Spot on! You've touched your nose perfectly."
        }
    },
    "cross your arms": {
        "instruction": "Fold your arms across your chest.",
        "feedback": {
            "arms_not_crossed": "Try bringing your arms closer together across your chest.",
            "wrong_position": "Make sure your arms are crossing in front of your chest.",
            "almost": "Nearly there! Just adjust your arm position slightly.",
            "success": "Excellent arm crossing! You look very authoritative."
        }
    },
    "stand on one leg": {
        "instruction": "Lift one foot off the ground and balance on the other leg.",
        "feedback": {
            "both_feet_down": "Try lifting one of your feet off the ground.",
            "not_stable": "Focus on maintaining your balance on one leg.",
            "almost": "You're close! Try to hold the position a bit longer.",
            "success": "Great balance! You're standing on one leg like a pro."
        }
    },
    "do a squat": {
        "instruction": "Bend your knees and lower your body as if sitting back into a chair.",
        "feedback": {
            "not_low_enough": "Try to lower your body more, bending your knees further.",
            "wrong_form": "Keep your back straight and your knees behind your toes.",
            "almost": "You're getting there! Go a little lower if you can.",
            "success": "Perfect squat form! You've nailed it."
        }
    }
}

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

current_action = None
action_completed = threading.Event()
action_start_time = None
action_timeout = 10  # 10 seconds for each action


def get_random_action():
    global current_action, action_start_time
    current_action = random.choice(list(actions.keys()))
    action_completed.clear()
    action_start_time = time.time()
    return current_action


def recognize_action(landmarks):
    global current_action

    if current_action == "raise your hand":
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.NOSE.value].y or \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.NOSE.value].y:
            return "success"
        elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER.value].y or \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y:
            return "almost"
        else:
            return "not_high_enough"

    elif current_action == "wave":
        left_wrist_movement = abs(
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x)
        right_wrist_movement = abs(
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x)
        if left_wrist_movement > 0.1 or right_wrist_movement > 0.1:
            return "success"
        elif left_wrist_movement > 0.05 or right_wrist_movement > 0.05:
            return "almost"
        else:
            return "not_moving"

    elif current_action == "shrug your shoulders":
        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.LEFT_EAR].value.y and \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < landmarks[
            mp_pose.PoseLandmark.RIGHT_EAR].value.y:
            return "success"
        elif landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.NOSE].value.y and \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.NOSE].value.y:
            return "almost"
        else:
            return "not_high_enough"

    elif current_action == "touch your nose":
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        distance_left = np.linalg.norm(np.array([nose.x, nose.y]) - np.array([left_wrist.x, left_wrist.y]))
        distance_right = np.linalg.norm(np.array([nose.x, nose.y]) - np.array([right_wrist.x, right_wrist.y]))
        if distance_left < 0.1 or distance_right < 0.1:
            return "success"
        elif distance_left < 0.2 or distance_right < 0.2:
            return "almost"
        else:
            return "not_close_enough"

    elif current_action == "cross your arms":
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x > landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x and \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x < landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER.value].x:
            return "success"
        elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x > landmarks[
            mp_pose.PoseLandmark.RIGHT_ELBOW.value].x and \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x < landmarks[
            mp_pose.PoseLandmark.LEFT_ELBOW.value].x:
            return "almost"
        else:
            return "arms_not_crossed"

    elif current_action == "stand on one leg":
        left_ankle_height = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_height = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        if abs(left_ankle_height - right_ankle_height) > 0.1:
            return "success"
        elif abs(left_ankle_height - right_ankle_height) > 0.05:
            return "almost"
        else:
            return "both_feet_down"

    elif current_action == "do a squat":
        hip_height = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[
            mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        knee_height = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y + landmarks[
            mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2
        if knee_height > hip_height:
            return "success"
        elif knee_height > landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - 0.1:
            return "almost"
        else:
            return "not_low_enough"

    return "wrong_body_part"


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                recognition_result = recognize_action(landmarks)

                if recognition_result == "success":
                    cv2.putText(frame, "Action recognized!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    action_completed.set()
                else:
                    cv2.putText(frame, f"Perform: {current_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Configure Gemini API
API_KEY = "AIzaSyC9ztlMxH0g9lotzLH4iJX8tNAMcoJFGlg"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')

# Sample scenarios
scenarios = [
    {
        "id": 1,
        "title": "At the Café",
        "description": "You're ordering coffee at a busy café. The barista seems stressed.",
        "difficulty": "Easy"
    },
    {
        "id": 2,
        "title": "Job Interview",
        "description": "You're interviewing for your dream job at a tech company.",
        "difficulty": "Medium"
    },
    {
        "id": 3,
        "title": "Lost in Translation",
        "description": "You're in a foreign country and need to ask for directions, but there's a language barrier.",
        "difficulty": "Hard"
    }
]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', user=fake_user)

@app.route('/emotion_game')
def emotion_game():
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

@app.route('/chat_game')
def chat_game():
    return render_template('chat_game.html', scenarios=scenarios)

@app.route('/action_recognition')
def action_recognition():
    action = get_random_action()
    return render_template('action_recognition.html', action=action, instruction=actions[action]['instruction'])


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_action')
def check_action():
    global action_start_time
    current_time = time.time()

    if action_completed.is_set():
        new_action = get_random_action()
        return jsonify({
            "completed": True,
            "message": actions[current_action]['feedback']['success'],
            "new_action": new_action,
            "instruction": actions[new_action]['instruction']
        })
    elif current_time - action_start_time > action_timeout:
        new_action = get_random_action()
        return jsonify({
            "completed": False,
            "message": "Time's up! Let's try a new action.",
            "new_action": new_action,
            "instruction": actions[new_action]['instruction']
        })
    else:
        # In a real-time application, you'd want to get the latest frame here
        # For simplicity, we're using the last recognized result
        recognition_result = recognize_action(
            pose.process(cv2.cvtColor(cv2.imread('temp_frame.jpg'), cv2.COLOR_BGR2RGB)).pose_landmarks.landmark)
        return jsonify({
            "completed": False,
            "message": actions[current_action]['feedback'].get(recognition_result, "Keep trying!"),
            "instruction": actions[current_action]['instruction']
        })

@app.route('/image_reader')
def image_reader():
    return render_template('image_reader.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html', feedbacks=fake_feedbacks)

@app.route('/progress')
def progress():
    return render_template('progress.html', user=fake_user, progress_data=fake_progress_data)

@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    # In a real app, you'd update the user's progress here
    return jsonify({"status": "success"})

@app.route('/api/add_feedback', methods=['POST'])
def add_feedback():
    # In a real app, you'd add the feedback here
    return jsonify({"status": "success"})

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

@app.route('/scenario/<int:scenario_id>', methods=['GET', 'POST'])
def scenario(scenario_id):
    scenario = next((s for s in scenarios if s['id'] == scenario_id), None)
    if not scenario:
        return "Scenario not found", 404

    if request.method == 'POST':
        user_input = request.json['user_input']

        # Generate system response using Gemini API
        prompt = f"Scenario: {scenario['description']}\nUser: {user_input}\nSystem (role-playing as the other person):"
        response = model.generate_content(prompt)
        system_response = response.text
        print(system_response)

        # Check if the user's input is meaningful
        check_prompt = f"Is the following user input meaningful and relevant to the scenario? User input: {user_input}"
        check_response = model.generate_content(check_prompt)

        if "no" in check_response.text.lower():
            return jsonify({"error": f"{system_response}"})

        return jsonify({"response": system_response})

    return render_template('scenario.html', scenario=scenario)


@app.route('/random-scenario')
def random_scenario():
    scenario = random.choice(scenarios)
    return jsonify(scenario)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

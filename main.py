import cv2
import numpy as np
from deepface import DeepFace
import random
import time
import tensorflow as tf
import threading
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import pygame
import tempfile
import os
import queue

# GPU Configuration
print("Số GPU khả dụng: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "GPU vật lý,", len(logical_gpus), "GPU logic")
    except RuntimeError as e:
        print(e)
else:
    print("Không tìm thấy GPU. Đang chạy trên CPU.")

# List of emotions and corresponding emojis
emotions = {
    "tức giận": "😠", "ghê tởm": "🤢", "sợ hãi": "😨",
    "hạnh phúc": "😊", "buồn bã": "😢", "ngạc nhiên": "😲", "bình thường": "😐"
}

# Success time for each emotion (in seconds)
emotion_success_times = {
    "tức giận": 2.0, "ghê tởm": 1.5, "sợ hãi": 1.5,
    "hạnh phúc": 2.0, "buồn bã": 2.0, "ngạc nhiên": 1.5, "bình thường": 2.5
}

# Detailed guides for each emotion in Vietnamese
emotion_guides = {
    "tức giận": [
        "Nhíu mày mạnh, kéo lông mày lại gần nhau",
        "Mím môi hoặc đưa hàm dưới ra",
        "Mở to mắt, nhìn thẳng"
    ],
    "ghê tởm": [
        "Nhăn mũi, kéo môi trên lên",
        "Nheo mắt một chút",
        "Nghiêng đầu ra sau một chút"
    ],
    "sợ hãi": [
        "Mở to mắt, kéo lông mày lên",
        "Hé miệng một chút",
        "Ngả người ra sau nếu có thể"
    ],
    "hạnh phúc": [
        "Cười rộng, để lộ răng",
        "Nhướn mày lên một chút",
        "Để mắt sáng lên, có nếp nhăn ở đuôi mắt"
    ],
    "buồn bã": [
        "Kéo khóe miệng xuống",
        "Nhíu mày, tạo nếp nhăn giữa hai lông mày",
        "Hạ ánh mắt xuống, có thể nhìn xuống dưới"
    ],
    "ngạc nhiên": [
        "Mở to mắt, nâng lông mày cao",
        "Hé miệng, hạ hàm dưới xuống một chút",
        "Tạo nếp nhăn trên trán"
    ],
    "bình thường": [
        "Thả lỏng các cơ mặt",
        "Giữ miệng khép, nhưng không chặt",
        "Nhìn thẳng với ánh mắt bình tĩnh"
    ]
}

# Colors for different types of text
COLORS = {
    'target': (255, 105, 180),  # Hồng đậm
    'success': (0, 255, 0),  # Xanh lá
    'error': (0, 0, 255),  # Đỏ
    'info': (255, 165, 0)  # Cam
}


class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.speech_queue = queue.Queue()
        self.current_speech = None
        self.interrupt_flag = threading.Event()
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()

    def speak(self, text, interrupt=False):
        if interrupt:
            self.interrupt_flag.set()
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        self.speech_queue.put(text)

    def _process_speech_queue(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self._play_speech(text)

    def _play_speech(self, text):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts = gTTS(text=text, lang='vi')
                tts.save(fp.name)
                self.current_speech = fp.name

            pygame.mixer.music.load(self.current_speech)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self.interrupt_flag.is_set():
                    pygame.mixer.music.stop()
                    self.interrupt_flag.clear()
                    break
                pygame.time.Clock().tick(10)

            os.unlink(self.current_speech)
            self.current_speech = None
        except Exception as e:
            print(f"Lỗi khi phát âm thanh: {e}")

    def stop(self):
        self.speech_queue.put(None)
        self.speech_thread.join()
        pygame.mixer.quit()


# Khởi tạo AudioManager
audio_manager = AudioManager()


def speak(text, interrupt=False):
    audio_manager.speak(text, interrupt)


# Load a font that supports Vietnamese
font = ImageFont.truetype("arial.ttf", 16)


def cv2_im_to_pil(cv2_im):
    return Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))


def pil_to_cv2_im(pil_im):
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def draw_text_with_pil(cv2_im, text, position, font, text_color=(255, 255, 255)):
    pil_im = cv2_im_to_pil(cv2_im)
    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=font, fill=text_color)
    return pil_to_cv2_im(pil_im)


def display_emotion_and_feedback(frame, target_emotion, feedback, detected_emotion=None):
    """Hiển thị cảm xúc mục tiêu và phản hồi trên khung hình"""
    height, width, _ = frame.shape

    text = f"Hãy thể hiện cảm xúc: {emotions.get(target_emotion, '?')} ({target_emotion})"
    frame = draw_text_with_pil(frame, text, (10, 15), font, COLORS['target'])

    y_offset = 45
    color = COLORS['success'] if "Tuyệt vời" in feedback else COLORS['error']
    for line in feedback.split('\n'):
        frame = draw_text_with_pil(frame, line, (10, y_offset), font, color)
        y_offset += 22

    if detected_emotion:
        text = f"Cảm xúc phát hiện: {emotions.get(detected_emotion, '?')} ({detected_emotion})"
        frame = draw_text_with_pil(frame, text, (10, height - 30), font, COLORS['info'])

    return frame


def analyze_emotion(frame):
    """Phân tích cảm xúc sử dụng DeepFace"""
    try:
        with tf.device('/GPU:0'):
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        detected_emotion = result[0]['dominant_emotion']
        # Map English emotion to Vietnamese
        emotion_map = {
            "angry": "tức giận", "disgust": "ghê tởm", "fear": "sợ hãi",
            "happy": "hạnh phúc", "sad": "buồn bã", "surprise": "ngạc nhiên",
            "neutral": "bình thường"
        }
        return emotion_map.get(detected_emotion, "unknown")
    except Exception as e:
        print(f"Lỗi khi phân tích cảm xúc: {e}")
        return "unknown"


def give_feedback(expected_emotion, detected_emotion):
    """Đưa ra phản hồi dựa trên cảm xúc phát hiện được"""
    if expected_emotion == detected_emotion:
        feedback = f"Tuyệt vời! Bạn đã thể hiện chính xác cảm xúc {expected_emotion}."
    elif detected_emotion == "unknown":
        feedback = "Không thể phát hiện cảm xúc. Hãy thử lại và đảm bảo khuôn mặt của bạn được nhìn thấy rõ ràng."
    else:
        feedback = f"Bạn đang thể hiện {detected_emotion}, không phải {expected_emotion}.\n"
        feedback += "Hãy thử lại với những gợi ý sau:\n"
        for guide in emotion_guides.get(expected_emotion, []):
            feedback += f"- {guide}\n"

    speak(feedback)
    return feedback


def congratulate(emotion, duration):
    """Đưa ra lời chúc mừng chi tiết hơn"""
    congratulation = f"Xuất sắc! Bạn đã giữ được biểu cảm {emotion} trong {duration:.1f} giây. "
    congratulation += "Khả năng kiểm soát khuôn mặt của bạn thật ấn tượng. "
    congratulation += "Hãy tiếp tục phát huy và thử thách bản thân với cảm xúc tiếp theo!"

    print(congratulation)
    speak(congratulation, interrupt=True)

    congrats_image = np.zeros((200, 400, 3), dtype=np.uint8)
    congrats_image = draw_text_with_pil(congrats_image, "Chúc mừng!", (30, 30), font, (0, 255, 0))
    congrats_image = draw_text_with_pil(congrats_image, f"Bạn đã thể hiện {emotion}", (30, 70), font, (255, 255, 255))
    congrats_image = draw_text_with_pil(congrats_image, f"trong {duration:.1f} giây!", (30, 110), font, (255, 255, 255))
    congrats_image = draw_text_with_pil(congrats_image, "Làm tốt lắm!", (30, 150), font, (0, 255, 255))

    cv2.imshow("Chúc mừng!", congrats_image)
    cv2.waitKey(5000)  # Hiển thị trong 5 giây
    cv2.destroyWindow("Chúc mừng!")


def emotion_analysis_thread(frame_queue, result_queue):
    """Luồng phân tích cảm xúc"""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        emotion = analyze_emotion(frame)
        result_queue.put(emotion)


def main():
    print("Chào mừng đến với Trò chơi Nhận diện Cảm xúc!")
    speak("Chào mừng đến với Trò chơi Nhận diện Cảm xúc!")
    print("Nhấn 'q' để thoát trò chơi bất cứ lúc nào.")
    speak("Nhấn q để thoát trò chơi bất cứ lúc nào.")

    cv2.namedWindow('Trò chơi Nhận diện Cảm xúc', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trò chơi Nhận diện Cảm xúc', 800, 600)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    analysis_thread = threading.Thread(target=emotion_analysis_thread, args=(frame_queue, result_queue))
    analysis_thread.start()

    target_emotion = random.choice(list(emotions.keys()))
    speak(f"Hãy thể hiện cảm xúc {target_emotion}")

    success_time = None
    last_frame_time = time.time()
    detected_emotion = None
    feedback = "Hãy bắt đầu thể hiện cảm xúc!"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - last_frame_time)
        last_frame_time = current_time

        # Cập nhật frame cho phân tích cảm xúc
        if frame_queue.empty():
            frame_queue.put(frame)

        # Kiểm tra kết quả phân tích cảm xúc
        if not result_queue.empty():
            detected_emotion = result_queue.get()

            if target_emotion == detected_emotion:
                if success_time is None:
                    success_time = current_time
                elif current_time - success_time >= emotion_success_times[target_emotion]:
                    duration = current_time - success_time
                    congratulate(target_emotion, duration)
                    target_emotion = random.choice(list(emotions.keys()))
                    speak(f"Bây giờ, hãy thử thể hiện {target_emotion}")
                    success_time = None
                    feedback = "Hãy bắt đầu thể hiện cảm xúc mới!"
            else:
                success_time = None
                feedback = give_feedback(target_emotion, detected_emotion)

        # Hiển thị thông tin trên frame
        frame = display_emotion_and_feedback(frame, target_emotion, feedback, detected_emotion)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Trò chơi Nhận diện Cảm xúc', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    frame_queue.put(None)
    analysis_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Cảm ơn bạn đã chơi!")
    speak("Cảm ơn bạn đã chơi!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
    finally:
        audio_manager.stop()
        pygame.quit()
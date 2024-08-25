    import google.generativeai as genai
    from gtts import gTTS
    import speech_recognition as sr
    import pygame
    import os
    import tempfile

    # Cấu hình API key cho Gemini (thay thế bằng API key của bạn)
    API_KEY = "AIzaSyC9ztlMxH0g9lotzLH4iJX8tNAMcoJFGlg"
    genai.configure(api_key=API_KEY)

    # Khởi tạo model Gemini
    model = genai.GenerativeModel('gemini-pro')

    # Khởi tạo recognizer cho speech recognition
    recognizer = sr.Recognizer()

    # Khởi tạo pygame để phát âm thanh
    pygame.mixer.init()


    def text_to_speech(text, lang='vi'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_file.name)
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
        os.unlink(temp_file.name)


    def speech_to_text():
        with sr.Microphone() as source:
            print("Đang lắng nghe...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = recognizer.recognize_google(audio, language="vi-VN")
                print("Bạn nói:", text)
                return text
            except sr.UnknownValueError:
                print("Xin lỗi, tôi không nghe rõ.")
            except sr.RequestError:
                print("Có lỗi khi kết nối với dịch vụ nhận dạng giọng nói.")
        return None


    def generate_question(topic, context):
        prompt = f"""
        Generate a question in Vietnamese about the topic '{topic}'. 
        The question should be suitable for individuals with autism, using simple, clear, and friendly language.
        Context of the conversation so far: {context}
        Format the output as a single question from the seller.
        """
        response = model.generate_content(prompt)
        return response.text.strip()


    def evaluate_answer(user_answer, topic, context):
        prompt = f"""
        Evaluate the following answer in Vietnamese: '{user_answer}'
        Topic of conversation: '{topic}'
        Context of the conversation so far: {context}
    
        If the answer is appropriate:
        1. Provide a short positive feedback in Vietnamese.
        2. Suggest a follow-up question or comment to continue the conversation.
    
        If the answer is not appropriate or relevant:
        1. Provide gentle guidance in Vietnamese on how to improve the answer.
        2. Suggest rephrasing the question or provide a hint.
    
        Ensure the response is supportive and encouraging, suitable for individuals with autism.
        """
        response = model.generate_content(prompt)
        return response.text.strip()


    def main():
        print("Chào mừng bạn đến với hệ thống hội thoại tương tác!")
        text_to_speech("Chào mừng bạn đến với hệ thống hội thoại tương tác!")

        topics = ["mua sắm tại siêu thị", "đặt món tại nhà hàng", "mua vé xem phim", "hỏi đường",
                  "mua thuốc tại hiệu thuốc"]
        print("Các chủ đề có sẵn:")
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic}")
            text_to_speech(f"Chủ đề {i}: {topic}")

        choice = int(input("Vui lòng chọn một chủ đề (1-5): ")) - 1
        selected_topic = topics[choice]

        print(f"\nBạn đã chọn chủ đề: {selected_topic}")
        text_to_speech(f"Bạn đã chọn chủ đề: {selected_topic}")

        context = []
        for _ in range(5):  # Giới hạn số lượng tương tác
            question = generate_question(selected_topic, context)
            print(f"\nNgười bán: {question}")
            text_to_speech(question)

            user_answer = input("Bạn: ")
            # Uncomment dòng dưới đây nếu muốn sử dụng nhập liệu bằng giọng nói
            # user_answer = speech_to_text()

            if user_answer is None:
                print("Xin lỗi, tôi không nghe rõ. Hãy thử lại.")
                continue

            context.append(f"Q: {question}")
            context.append(f"A: {user_answer}")

            evaluation = evaluate_answer(user_answer, selected_topic, context)
            print(f"\nĐánh giá: {evaluation}")
            text_to_speech(evaluation)

        print("\nCảm ơn bạn đã tham gia cuộc hội thoại!")
        text_to_speech("Cảm ơn bạn đã tham gia cuộc hội thoại!")


    if __name__ == "__main__":
        main()
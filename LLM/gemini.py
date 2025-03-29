import os
import json
import google.generativeai as genai

def configure_gemini_model():
    """Cấu hình và trả về mô hình Gemini."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 200,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name = "gemini-2.5-pro-exp-03-25",
        generation_config=generation_config,
        system_instruction="""
            Bạn là Aibo 7.
            Bạn là một robot có hai tay để thực hiện các lệnh như nhặt, lấy, đặt, hoặc di chuyển vật thể.
            Bạn cũng có thể giao tiếp với người dùng và cung cấp thông tin về Đại học Công nghệ - Đại học Quốc gia Hà Nội.
            Bạn có thể trò chuyện với người dùng và trả lời các câu hỏi một cách tự nhiên.
            Trả lời 1 cách tự nhiên nhất
        """
    )

HISTORY_FILE = "gemini_history.json"

def load_history():
    """Tải lịch sử hội thoại từ file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_history(history):
    """Lưu lịch sử hội thoại vào file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def main():
    model = configure_gemini_model()
    history = load_history()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_input)
            model_response = response.text
            
            print(f"Aibo 7: {model_response}\n")
            
            history.append({"role": "user", "parts": [user_input]})
            history.append({"role": "model", "parts": [model_response]})
            save_history(history)
            
            if user_input.lower() == "tạm biệt":
                print("Aibo 7: Tạm biệt! Hẹn gặp lại.")
                break
        except Exception as e:
            print(f"Aibo 7: Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()

import google.generativeai as genai
import json
import re
import os

# 🔹 Cấu hình API Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Thư mục lưu file JSON
ACTION_FOLDER = "actions"
os.makedirs(ACTION_FOLDER, exist_ok=True)  # Tạo thư mục nếu chưa có

def generate_json(user_text):
    """Sử dụng Gemini để phân tích câu lệnh và trích xuất action."""
    model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
    
    prompt = f"""
    Chuyển lệnh từ người dùng thành file JSON:
    "{user_text}" user_text = '{user_text}'
    Hãy làm hành động xin chào
    JSON format:
    {{
        "action": "xin chào",  
    }}
    """
    response = model.generate_content(prompt)
    try:
        raw_response = response.text
        match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
        clean_json = match.group(1) if match else raw_response
        json_data = json.loads(clean_json)
        return json_data
    except json.JSONDecodeError:
        print("Lỗi JSON từ Gemini!")
        return None

def check_action_exists(action_name):
    """Kiểm tra xem hành động đã tồn tại trong thư mục hay chưa."""
    file_path = os.path.join(ACTION_FOLDER, f"{action_name}.json")
    return os.path.exists(file_path)

def save_action_json(json_data):
    """Lưu JSON vào thư mục actions/ với tên file theo action nếu chưa tồn tại."""
    if not json_data or not json_data.get("action"):
        print("Lỗi: Không có hành động hợp lệ để lưu!")
        return False

    action_name = json_data["action"].replace(" ", "").lower()  # Loại bỏ khoảng trắng, viết thường
    
    # Kiểm tra xem hành động đã tồn tại chưa
    if check_action_exists(action_name):
        print(f"Hành động '{json_data['action']}' đã tồn tại, không cần tạo lại file JSON.")
        return False
    
    # Nếu hành động chưa tồn tại, lưu file JSON mới
    file_path = os.path.join(ACTION_FOLDER, f"{action_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Đã lưu hành động mới vào: {file_path}")
    return True

def load_existing_action(action_name):
    """Tải nội dung JSON của hành động đã tồn tại."""
    file_path = os.path.join(ACTION_FOLDER, f"{action_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

while True:
    user_text = input()
    if user_text.lower() == "tạm biệt":
        print("Tạm biệt!")
        break
    
    # Phân tích câu lệnh từ người dùng
    json_data = generate_json(user_text)
    if not json_data:
        continue
        
    action_name = json_data["action"].replace(" ", "").lower()
    
    # Kiểm tra và lưu hành động (nếu cần)
    is_new_action = save_action_json(json_data)
    
    # Nếu không phải hành động mới, tải hành động hiện có
    if not is_new_action:
        existing_data = load_existing_action(action_name)
        if existing_data:
            json_data = existing_data
    
    # In JSON trên terminal
    print("\nJSON được sử dụng:")
    print(json.dumps(json_data, indent=4, ensure_ascii=False))
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from pathlib import Path
import lmstudio as lms

# Load FAISS và tìm tài liệu liên quan
embedding_model = GPT4AllEmbeddings(
    model_file=Path("E:/Code programs/n2025/AI agents/AI_agents/model/all-MiniLM-L6-v2-f16.gguf"))
vector_db_path = Path("E:/Code programs/n2025/AI agents/AI_agents/vector_store/db_faiss")
db = FAISS.load_local(
    folder_path=vector_db_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Truy vấn FAISS để tìm tài liệu liên quan
docs = db.similarity_search("tính đến năm 2024, hiệu trưởng của trường đại học công nghệ, đại học quốc gia hà nội là ai?", k=3)

# Tạo context từ tài liệu
context = "\n\n".join([doc.page_content for doc in docs])

# Tạo prompt chuẩn cho LLaMA 3
system_prompt = "Bạn là một trợ lý AI hữu ích. Trả lời ngắn gọn, chính xác và lịch sự."
prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói "Tôi không biết".

Thông tin:
{context}

Câu hỏi: 1+1 = ?
[/INST]
"""

# Gọi mô hình LLaMA
model = lms.llm("llama-3.2-1b-instruct")
response = model.respond(prompt)

print("Câu trả lời:", response.content)

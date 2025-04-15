import lmstudio as lms
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain_core.language_models import LLM
from pydantic import Field
from typing import List, Optional, Dict, Any
from pathlib import Path

# Đường dẫn đến vector store
vector_db_path = Path("E:/Code programs/n2025/AI agents/AI_agents/vector_store/db_faiss")

# Tạo wrapper cho LMStudio
class LMStudioLangchainLLM(LLM):
    model_name: str = Field(default="gemma-3-4b-it")
    temperature: float = Field(default=0.2)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # Khởi tạo model từ LMStudio
        model = lms.llm(self.model_name) 
        # Gọi API để nhận phản hồi (tương thích với Gemma-3 format prompt)
        response = model.respond(prompt)
        
        # Trích xuất nội dung phản hồi
        return response.content

    @property
    def _llm_type(self) -> str:
        return "lmstudio-custom"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }

#Tạo Prompt Template chuẩn cho Gemma 3
def create_prompt_template():
    # Prompt chuẩn cho Gemma 3 với context và question
    system_prompt = (
        "<bos><start_of_turn>user\n"
        "Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết.\n\n"
        "Thông tin: {context}\n\n"
        "Câu hỏi: {input}<end_of_turn>\n"
        "<start_of_turn>model"
    )
    # Sử dụng ChatPromptTemplate để tương thích với LangChain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])
    return prompt
#RAG 
#Tạo Retrieval QA Chain
def create_qa_chain(llm, vectorstore, prompt):
    # Tạo document chain để xử lý context và trả lời câu hỏi
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Tạo retrieval chain
    retriever = db.as_retriever(search_kwargs={"k": 3}) #k = 3 để lấy 3 tài liệu liên quan nhất
    qa_chain = create_retrieval_chain(retriever, document_chain)
    #qa_chain.invoke = lambda inputs: qa_chain({"input": inputs["input"]}) # Đảm bảo đầu vào là đúng định dạng
    return qa_chain

#Load Vector DB
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(
        model_file=Path("E:/Code programs/n2025/AI agents/AI_agents/model/all-MiniLM-L6-v2-f16.gguf")
    )
    db = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

#Main Logic
if __name__ == "__main__":
    # Load vector database
    db = read_vectors_db()
    
    # Khởi tạo LLM
    llm = LMStudioLangchainLLM(model_name="gemma-3-4b-it", temperature=0.2)

    # Tạo prompt
    prompt = create_prompt_template()

    # Tạo retrieval chain
    qa_chain = create_qa_chain(llm, db, prompt)

    # Demo interactive mode (optional)
    print("\n--- Chế độ tương tác ---")
    model = lms.llm("gemma-3-4b-it")
    
    while True:
        try:
            user_input = input("Bạn (để trống để thoát): ")
        except EOFError:
            print()
            break
        if not user_input:
            break
            
        # Xử lý RAG cho câu hỏi
        docs = db.similarity_search(user_input, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Tạo prompt theo định dạng Gemma 3
        gemma_prompt = (
            "<bos><start_of_turn>user\n"
            "Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói hãy hướng dẫn tôi.\n\n"
            f"Thông tin: {context}\n\n"
            f"Câu hỏi: {user_input}<end_of_turn>\n"
            "<start_of_turn>model"
        )
        
        # Stream phản hồi
        prediction_stream = model.respond_stream(gemma_prompt)
        
        print("LLM: ", end="", flush=True)
        for fragment in prediction_stream:
            print(fragment.content, end="", flush=True)
        print("\n")
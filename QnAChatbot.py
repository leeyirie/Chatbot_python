import os
import warnings
warnings.filterwarnings("ignore")
# from langchain_community.document_loaders import PyMuPDFLoader # type: ignore
# from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama 

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
# from chromadb import Chroma
import chromadb



# Ollama 파이썬 라이브러리 (아직 공식적이지 않을 수 있으며, REST API 직접 호출 또는 LangChain 통합 등을 고려)
# 여기서는 LangChain을 사용한 예시를 보여드립니다.
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


# PDF 파일 경로
pdf_path = "./pdf/qna.pdf"

# Ollama 모델 이름
ollama_model = "exaone3.5"  # 또는 사용하려는 다른 모델
# ollama_model = "benedict/linkbricks-llama3.1-korean:70b"

# 임베딩 모델
embedding_model_name = "all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# ChromaDB 저장 경로 (선택 사항: 메모리에 저장하려면 None)
persist_directory = "chroma_db"



def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PDF 파일 읽기 오류: {e}")
    return text



def create_knowledge_base(pdf_text, embeddings, persist_directory=None):
    # 텍스트를 문서 단위로 분할 (예: 문장 또는 특정 길이 청크)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # ChromaDB에 벡터 임베딩 저장
    if persist_directory:
        db = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)
        db.persist()
        return db
    else:
        return Chroma.from_texts(texts, embeddings)

def create_chatbot(ollama_model, knowledge_base):
    llm = Ollama(model=ollama_model)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=knowledge_base.as_retriever())
    return qa


if __name__ == "__main__":
    # 1. PDF 파일 내용 추출
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        exit()

    # 2. 및 3. 지식 베이스 생성 (벡터 임베딩 및 저장)
    knowledge_base = create_knowledge_base(pdf_text, embeddings, persist_directory)

    # 4. 챗봇 생성
    chatbot = create_chatbot(ollama_model, knowledge_base)

    # 질의응답 루프
    print("Ollama 기반 챗봇 시작 (종료하려면 'quit' 입력)")
    while True:
        query = input("질문: ")
        if query.lower() == 'quit':
            break

        try:
            response = chatbot({"query": query})
            print("답변:", response['result'])
        except Exception as e:
            print(f"오류 발생: {e}")

    # (선택 사항) ChromaDB 영구 저장 시 연결 닫기
    if persist_directory and knowledge_base:
        knowledge_base._client.close()




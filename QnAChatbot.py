import os
import warnings
warnings.filterwarnings("ignore")

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 파일 경로 - PDF 또는 TXT 파일
file_path = "./pdf/qna.txt"  # PDF가 아닌 텍스트 파일 사용

# 임베딩 모델
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)


def extract_text_from_file(file_path):
    """파일에서 텍스트 추출 - PDF 또는 TXT 파일 모두 지원"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            # PDF 파일 처리
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif file_path.lower().endswith('.txt'):
            # 텍스트 파일 처리
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            print(f"지원되지 않는 파일 형식: {file_path}")
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
    return text


def create_knowledge_base(text):
    """텍스트를 파싱하여 질문-답변 쌍 추출"""
    # 간단한 구현: "Question:" 및 "Answer:" 포맷을 찾아 파싱
    qa_pairs = []
    
    # 텍스트를 줄 단위로 분할
    lines = text.split('\n')
    
    current_question = None
    current_answer = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            # 이전 QA 쌍이 있으면 저장
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            
            # 새 질문 시작
            current_question = line[len("Question:"):].strip()
            current_answer = None
        elif line.startswith("Answer:") and current_question:
            # 답변 시작
            current_answer = line[len("Answer:"):].strip()
    
    # 마지막 QA 쌍 추가
    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))
    
    return qa_pairs


def find_most_similar_qa(query, qa_pairs, model):
    """질문에 가장 유사한 QA 쌍 찾기"""
    if not qa_pairs:
        return "지식 베이스에 데이터가 없습니다."
    
    # 쿼리 임베딩
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 모든 질문 임베딩
    question_embeddings = model.encode([q for q, _ in qa_pairs], convert_to_tensor=True)
    
    # 코사인 유사도 계산
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    
    # 가장 유사한 인덱스 찾기
    best_idx = int(np.argmax(similarities))
    similarity_score = float(similarities[best_idx])
    
    # 임계값 설정 (0.5 이상일 때만 관련 있는 것으로 간주)
    if similarity_score < 0.5:
        return "죄송합니다. 질문에 답할 수 있는 정보가 없습니다."
    
    # 가장 유사한 QA 쌍 반환
    return qa_pairs[best_idx][1]


if __name__ == "__main__":
    # 1. 파일 내용 추출
    file_text = extract_text_from_file(file_path)
    if not file_text:
        print("파일을 읽을 수 없습니다. 파일 경로를 확인하세요.")
        exit()

    # 2. 지식 베이스 생성 (질문-답변 쌍 추출)
    qa_pairs = create_knowledge_base(file_text)
    
    # 추출된 질문-답변 쌍 출력
    print(f"추출된 질문-답변 쌍: {len(qa_pairs)}개")
    for i, (q, a) in enumerate(qa_pairs):
        print(f"[{i+1}] Q: {q}")
        print(f"    A: {a}")
    print()

    # 질의응답 루프
    print("질의응답 챗봇 시작 (종료하려면 'quit' 입력)")
    while True:
        query = input("질문: ")
        if query.lower() == 'quit':
            break

        try:
            answer = find_most_similar_qa(query, qa_pairs, model)
            print("답변:", answer)
        except Exception as e:
            print(f"오류 발생: {e}")




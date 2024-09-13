import os
import torch
import numpy as np
import speech_recognition as sr
import pyttsx3
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langserve import RemoteRunnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document

# 원격 LLM 초기화
remote_llm = RemoteRunnable("https://holy-integral-redfish.ngrok-free.app/llm/")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# KoReRanker 모델 초기화
model_path = "Dongjin-kr/ko-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore, texts

def bm25_retriever(query, docs, k=4):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:k]
    return [docs[i] for i in top_n]

def rerank_with_koreranker(query, docs):
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
    reranked_docs = [doc for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked_docs[:min(3, len(reranked_docs))]

answer_prompt = PromptTemplate.from_template(
    "주어진 단락을 이용하여 다음 질문에 100단어 이내로 간결하게 답하시오. 반드시 100단어를 초과하지 마세요. 답변이 끝나면 반드시 '<END_OF_RESPONSE>'를 추가하세요:\n질문: {question}\n\n단락: {context}\n\n답변:"
)

def create_rag_chain(vectorstore, docs):
    def retrieve_and_rerank(query):
        initial_docs = bm25_retriever(query, docs)
        reranked_docs = rerank_with_koreranker(query, initial_docs)
        return reranked_docs

    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=RunnablePassthrough() | retrieve_and_rerank,
        )
        | {
            "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
            "question": lambda x: x["question"],
        }
        | answer_prompt
        | remote_llm
    )
    return rag_chain

def truncate_response(response, max_words=100):
    response = response.split('<END_OF_RESPONSE>')[0].strip()
    words = response.split()
    if len(words) <= max_words:
        return response
    truncated = ' '.join(words[:max_words])
    last_sentence_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_sentence_end != -1:
        truncated = truncated[:last_sentence_end+1]
    return truncated.strip()

def recognize_speech_from_mic(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `speech_recognition.Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `speech_recognition.Microphone` instance")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("질문을 말씀해주세요!")
        audio = recognizer.listen(source)
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }
    try:
        response["transcription"] = recognizer.recognize_google(audio, language='ko-KR')
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    return response

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    pdf_path = "./컴소학과v5_4.pdf"
    vectorstore, docs = create_vector_store(pdf_path)
    rag_chain = create_rag_chain(vectorstore, docs)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        speech_result = recognize_speech_from_mic(recognizer, microphone)
        if speech_result["error"]:
            print("Error:", speech_result["error"])
            continue
        if not speech_result["transcription"]:
            print("음성을 인식하지 못했습니다. 다시 시도해주세요.")
            continue

        user_input = speech_result["transcription"]
        print("인식된 질문:", user_input)

        if user_input.lower() == '종료':
            break
        
        try:
            rag_response = rag_chain.invoke(user_input)
            truncated_response = truncate_response(rag_response.content)
            print("응답:", truncated_response)
            speak_text(truncated_response)
        except Exception as e:
            print("오류 발생:", str(e))
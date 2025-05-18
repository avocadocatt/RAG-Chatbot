import google.generativeai as genai
from src.config import GOOGLE_API_KEY, GEMINI_GENERATION_MODEL

genai.configure(api_key=GOOGLE_API_KEY)

class GeminiLLMHandler:
    def __init__(self, model_name=GEMINI_GENERATION_MODEL):
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, question, context):
        """
        Generates an answer using Gemini based on the question and provided context.
        """
        prompt = f"""Answer based on the information provided. If you do not find the relevant information, please say you dont know, do not try to make up an answer. Only use the information provided in the "Reference information" to answer the question. Do not add any additional information.

        Reference information:
        ---
        {context}
        ---

        Question: {question}

        Answer:
        """
        try:
            # print(f"\n---PROMPT TO LLM---\n{prompt}\n---------------------\n")
            response = self.model.generate_content(contents=prompt)
            return response.text
        except Exception as e:
            print(f"Error generating answer with Gemini: {e}")
            return "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời."


if __name__ == '__main__':
    llm = GeminiLLMHandler()
    sample_question = "RAG là gì?"
    sample_context = "RAG (Retrieval Augmented Generation) là một kỹ thuật kết hợp truy xuất thông tin với sinh văn bản."
    answer = llm.generate_answer(sample_question, sample_context)
    print(f"Question: {sample_question}")
    print(f"Context: {sample_context}")
    print(f"Answer: {answer}")

    sample_question_2 = "Ai là tổng thống Mỹ?"
    answer_2 = llm.generate_answer(sample_question_2, sample_context)
    print(f"\nQuestion: {sample_question_2}")
    print(f"Context: {sample_context}") # Context không liên quan
    print(f"Answer: {answer_2}")
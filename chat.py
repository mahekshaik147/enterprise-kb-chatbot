from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retrieve import retrieve

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

PROMPT = """You are a knowledge base assistant for TechNova Solutions.
Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I couldn't find that in the company documents."
Keep the answer short and mention the source number like [1].

Context:
{context}

Question: {question}
Answer:"""

def answer(question, k=3, role="employee"):
    chunks = retrieve(question, k=k, role=role)
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}, page {c['page'] + 1}) {c['text']}"
        for i, c in enumerate(chunks)
    )
    reply = llm.invoke(PROMPT.format(context=context, question=question)).text
    return reply, chunks

if __name__ == "__main__":
    role = input("Role (employee/hr): ").strip().lower()
    if role == "":
        role = "employee"

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        reply, chunks = answer(q, role=role)
        print("\n" + reply)
        print("\nSources:")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] {c['source']} (page {c['page'] + 1}, distance {c['distance']:.2f})")
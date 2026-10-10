from retrieve import retrieve

questions = [
    "How many sick leaves do I get?",
    "How do I reset my password?",
    "What is the notice period after confirmation?",
    "What is the salary of a senior engineer?",
    "What is the capital of France?",
]

for q in questions:
    top = retrieve(q, k=1)[0]
    print(f"\nQ: {q}")
    print(f"   Top: {top['source']} | distance {top['distance']:.3f}")
    print(f"   {top['text'][:200]}")
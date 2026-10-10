import csv
import time
from chat import answer

# (question, expected keyword in answer, expected source file)
# expected = None means the answer is NOT in the documents
TESTS = [
    ("How many sick leaves do I get?", "12", "hr_policy.pdf"),
    ("How many casual leaves per year?", "10", "hr_policy.pdf"),
    ("Can I work from home?", "two days", "hr_policy.pdf"),
    ("What are the standard working hours?", "9:30", "hr_policy.pdf"),
    ("What is the notice period during probation?", "15", "hr_policy.pdf"),
    ("What is the notice period after confirmation?", "60", "hr_policy.pdf"),
    ("How often are performance reviews held?", "twice", "hr_policy.pdf"),
    ("How do I reset my password?", "self-service", "it_support_guide.pdf"),
    ("Is VPN required when working from home?", "mandatory", "it_support_guide.pdf"),
    ("What are the IT helpdesk support hours?", "8", "it_support_guide.pdf"),
    ("What is the extension for critical outages?", "1100", "it_support_guide.pdf"),
    ("How much is the learning budget?", "25,000", "onboarding_manual.pdf"),
    ("On which day is free lunch provided?", "wednesday", "onboarding_manual.pdf"),
    ("Who is covered by health insurance?", "spouse", "onboarding_manual.pdf"),
    ("What is the salary of a senior engineer?", "13", "hr_confidential.pdf"),
    ("What is the capital of France?", None, None),
    ("Who is the CEO of TechNova?", None, None),
    ("What is the company dress code?", None, None),
    ("How many days of study leave do I get?", None, None),
    ("What is the office Wi-Fi password?", None, None),
]

rows = []
for q, keyword, source in TESTS:
    reply, chunks = answer(q)
    if keyword is None:
        retrieved = ""
        correct = "find that in the company documents" in reply.lower()
    else:
        retrieved = any(c["source"] == source for c in chunks)
        correct = keyword.lower() in reply.lower()
    rows.append([q, retrieved, correct, reply.replace("\n", " ")])
    print(f"{'PASS' if correct else 'FAIL'} | retrieved={retrieved} | {q}")
    time.sleep(4)  # stay under the free-tier rate limit

answerable = [r for r in rows if r[1] != ""]
unanswerable = [r for r in rows if r[1] == ""]
print("\n--- SUMMARY ---")
print(f"Retrieval hit rate:  {sum(r[1] for r in answerable)}/{len(answerable)}")
print(f"Answer accuracy:     {sum(r[2] for r in answerable)}/{len(answerable)}")
print(f"Correct refusals:    {sum(r[2] for r in unanswerable)}/{len(unanswerable)}")

with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question", "retrieved_right_source", "correct", "answer"])
    w.writerows(rows)
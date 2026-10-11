import csv
import os
import time
from chat import answer

RESULTS_FILE = "eval_results.csv"

# (question, expected keyword, expected source file, role)
# keyword None = the answer should NOT be available to this role
TESTS = [
    ("How many sick leaves do I get?", "12", "hr_policy.pdf", "employee"),
    ("Can I work from home?", "two days", "hr_policy.pdf", "employee"),
    ("What are the standard working hours?", "9:30", "hr_policy.pdf", "employee"),
    ("What is the notice period during probation?", "15", "hr_policy.pdf", "employee"),
    ("What is the notice period after confirmation?", "60", "hr_policy.pdf", "employee"),
    ("How do I reset my password?", "self-service", "it_support_guide.pdf", "employee"),
    ("Is VPN required when working from home?", "mandatory", "it_support_guide.pdf", "employee"),
    ("What is the extension for critical outages?", "1100", "it_support_guide.pdf", "employee"),
    ("How much is the learning budget?", "25,000", "onboarding_manual.pdf", "employee"),
    ("On which day is free lunch provided?", "wednesday", "onboarding_manual.pdf", "employee"),
    ("What is the salary of a senior engineer?", "13", "hr_confidential.pdf", "hr"),
    ("What is the salary of a senior engineer?", None, None, "employee"),
    ("What is the capital of France?", None, None, "employee"),
    ("Who is the CEO of TechNova?", None, None, "employee"),
    ("What is the company dress code?", None, None, "employee"),
]

RETRY_ERRORS = ["503", "504", "UNAVAILABLE", "DEADLINE"]

def save(rows):
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "role", "retrieved_right_source", "correct", "answer"])
        w.writerows(rows)

def load():
    rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            for r in list(csv.reader(f))[1:]:
                retrieved = "" if r[2] == "" else (r[2] == "True")
                rows.append([r[0], r[1], retrieved, r[3] == "True", r[4]])
    return rows

def ask(q, role):
    # retry when Google's servers are overloaded or time out
    for attempt in range(4):
        try:
            return answer(q, role=role)
        except Exception as e:
            retryable = any(x in str(e) for x in RETRY_ERRORS)
            if retryable and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"   server busy, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

rows = load()
done = {(r[0], r[1]) for r in rows}
print(f"Resuming: {len(rows)} questions already done")

try:
    for q, keyword, source, role in TESTS:
        if (q, role) in done:
            continue
        reply, chunks = ask(q, role)
        if keyword is None:
            retrieved = ""
            correct = "find that in the company documents" in reply.lower()
        else:
            retrieved = any(c["source"] == source for c in chunks)
            correct = keyword.lower() in reply.lower()
        rows.append([q, role, retrieved, correct, reply.replace("\n", " ")])
        save(rows)
        print(f"{'PASS' if correct else 'FAIL'} | {role} | retrieved={retrieved} | {q}")
        time.sleep(13)  # stay under 5 requests per minute
except Exception as e:
    print(f"\nStopped early: {str(e)[:200]}")

answerable = [r for r in rows if r[2] != ""]
unanswerable = [r for r in rows if r[2] == ""]
print("\n--- SUMMARY ---")
print(f"Questions run:      {len(rows)}/{len(TESTS)}")
print(f"Retrieval hit rate: {sum(r[2] for r in answerable)}/{len(answerable)}")
print(f"Answer accuracy:    {sum(r[3] for r in answerable)}/{len(answerable)}")
print(f"Correct refusals:   {sum(r[3] for r in unanswerable)}/{len(unanswerable)}")
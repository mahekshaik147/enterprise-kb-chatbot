from retrieve import retrieve

# (question, expected source file) - None means not in the documents
TESTS = [
    ("How many sick leaves do I get?", "hr_policy.pdf"),
    ("How many casual leaves per year?", "hr_policy.pdf"),
    ("Can I work from home?", "hr_policy.pdf"),
    ("What are the standard working hours?", "hr_policy.pdf"),
    ("What is the notice period during probation?", "hr_policy.pdf"),
    ("What is the notice period after confirmation?", "hr_policy.pdf"),
    ("How often are performance reviews held?", "hr_policy.pdf"),
    ("How do I reset my password?", "it_support_guide.pdf"),
    ("Is VPN required when working from home?", "it_support_guide.pdf"),
    ("What are the IT helpdesk support hours?", "it_support_guide.pdf"),
    ("What is the extension for critical outages?", "it_support_guide.pdf"),
    ("How much is the learning budget?", "onboarding_manual.pdf"),
    ("On which day is free lunch provided?", "onboarding_manual.pdf"),
    ("Who is covered by health insurance?", "onboarding_manual.pdf"),
    ("What is the salary of a senior engineer?", "hr_confidential.pdf"),
    ("What is the capital of France?", None),
    ("Who is the CEO of TechNova?", None),
    ("What is the company dress code?", None),
]

top1 = top3 = total = 0
for q, source in TESTS:
    results = retrieve(q, k=3)
    best = results[0]["distance"]
    if source is None:
        print(f"NOT IN DOCS | best distance {best:.2f} | {q}")
        continue
    total += 1
    hit1 = results[0]["source"] == source
    hit3 = any(r["source"] == source for r in results)
    top1 += hit1
    top3 += hit3
    print(f"{'PASS' if hit3 else 'FAIL'} | top1={hit1} | distance {best:.2f} | {q}")

print(f"\nTop-1 retrieval accuracy: {top1}/{total}")
print(f"Top-3 retrieval accuracy: {top3}/{total}")
from retrieve import retrieve

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
]

for use_rerank in [False, True]:
    top1 = top3 = 0
    missed = []
    for q, source in TESTS:
        results = retrieve(q, k=3, role="hr", rerank=use_rerank)
        if results[0]["source"] == source:
            top1 += 1
        else:
            missed.append(q)
        top3 += any(r["source"] == source for r in results)
    name = "WITH reranker" if use_rerank else "WITHOUT reranker"
    print(f"\n{name}: Top-1 {top1}/{len(TESTS)} | Top-3 {top3}/{len(TESTS)}")
    for q in missed:
        print(f"   Top-1 miss: {q}")
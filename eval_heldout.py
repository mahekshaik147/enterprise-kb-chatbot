from retrieve import retrieve

# New questions, never used for tuning: (question, expected source, role)
TESTS = [
    ("How many days of paternity leave do I get?", "hr_policy.pdf", "employee"),
    ("How long is maternity leave?", "hr_policy.pdf", "employee"),
    ("How many earned leave days can be carried forward?", "hr_policy.pdf", "employee"),
    ("In which months are performance reviews held?", "hr_policy.pdf", "employee"),
    ("How often do passwords expire?", "it_support_guide.pdf", "employee"),
    ("How long does it take to get a replacement laptop?", "it_support_guide.pdf", "employee"),
    ("Where do I collect my laptop on the first day?", "onboarding_manual.pdf", "employee"),
    ("How many hours is the security training?", "onboarding_manual.pdf", "employee"),
    ("When is the cafeteria open?", "onboarding_manual.pdf", "employee"),
    ("What is the maximum annual bonus?", "hr_confidential.pdf", "hr"),
    ("What is the maximum annual bonus?", None, "employee"),  # must not leak
    ("What is the parking policy?", None, "employee"),         # not in docs
]

top1 = top3 = total = 0
for q, source, role in TESTS:
    results = retrieve(q, k=3, role=role)
    if source is None:
        leaked = any(r["source"] == "hr_confidential.pdf" for r in results)
        print(f"{'LEAK!' if leaked else 'OK   '} | {role} | best distance {results[0]['distance']:.2f} | {q}")
        continue
    total += 1
    hit1 = results[0]["source"] == source
    hit3 = any(r["source"] == source for r in results)
    top1 += hit1
    top3 += hit3
    print(f"{'PASS' if hit3 else 'FAIL'} | top1={hit1} | distance {results[0]['distance']:.2f} | {q}")

print(f"\nHeld-out Top-1: {top1}/{total}")
print(f"Held-out Top-3: {top3}/{total}")
from retrieve import retrieve

q = "What is the salary of a senior engineer?"
for role in ["employee", "hr"]:
    print(f"\n--- role: {role} ---")
    for r in retrieve(q, k=3, role=role):
        print(f"{r['source']} | distance {r['distance']:.2f}")
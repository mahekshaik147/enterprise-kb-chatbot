from retrieve import retrieve

q = "Can I work from home?"
for use_rerank in [False, True]:
    print(f"\n--- rerank={use_rerank} ---")
    for r in retrieve(q, k=4, role="hr", rerank=use_rerank):
        score = f" | rerank score {r['score']:.2f}" if "score" in r else ""
        print(f"{r['source']} | distance {r['distance']:.2f}{score}")
        print("   ", r["text"][:150])
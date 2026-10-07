import time
import requests

url = "http://127.0.0.1:8000/rag/retrieve"
body = {"query": "广安要求党员的岗位", "top_k": 2, "data_type": "kaogong"}

for i in [1, 2, 3]:
    t0 = time.time()
    r = requests.post(url, json=body)
    dt = (time.time() - t0) * 1000
    j = r.json()
    print(f"[{i}] {dt:.0f}ms → {r.status_code} strategy={j.get('strategy')} count={j.get('count')}")
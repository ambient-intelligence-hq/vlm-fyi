"""Moonshot direct, kimi-k3.1-pro: tiny images per request, one request per ~70 s so every
request starts with a full rate-limit window. Separates an image-count limit from TPM."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import probe as P
tiny = P.data_uri(P.solid_jpeg(64, 64))
def send(n):
    body = {"model": "kimi-k3.1-pro", "max_tokens": 16, "messages": P.image_messages([tiny] * n) + [{"role": "user", "content": "Reply with OK."}]}
    req = urllib.request.Request("https://api.moonshot.ai/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + os.environ["MOONSHOT_API_KEY"], "Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300)); return "OK", r["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        m = json.loads(e.read() or b"{}").get("error", {}).get("message", "")
        return e.code, m.split("<")[0].replace("Your account ", "")[-90:] if "account" in m else m[:120]
lo, hi = 256, 512
for n in [384, 512] + [None] * 5:                   # probe 384 (same count as the 768x432 frames that passed), confirm 512, then bisect
    if n is None:
        if hi - lo <= 8: break
        n = (lo + hi) // 2
    res = send(n); ok = res[0] == "OK"
    if ok: lo = max(lo, n)
    else: hi = min(hi, n)
    print(f"{time.strftime('%H:%M:%S')} {n:4} images -> {res}  [pass<= {lo}, fail>= {hi}]", flush=True)
    time.sleep(70)
print("BISECT DONE")

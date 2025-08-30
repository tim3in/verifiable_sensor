# verifier.py
import os, time, json, hashlib, base64, sys, requests
from nacl.signing import VerifyKey
GQL_URL=os.getenv("GQL_URL","http://localhost:9181/api/v0/graphql")
CANON_KEYS=["deviceId","timestamp","prediction","probability"]
QUERY="""query($limit:Int!){ SensorEvent(order:{timestamp:ASC}, limit:$limit){
  deviceId timestamp prediction probability payload_hash signature signer_pubkey } }"""
last_ts=-1.0
print(f"[verify] Polling {GQL_URL} …")
while True:
    try:
        r=requests.post(GQL_URL,json={"query":QUERY,"variables":{"limit":200}},timeout=10)
        r.raise_for_status()
        data=r.json().get("data",{}).get("SensorEvent",[])
        new=[e for e in data if float(e.get("timestamp",-1))>last_ts]
        if not data: print("[verify] no rows yet"); 
        for e in new:
            payload={k:e.get(k) for k in CANON_KEYS}
            blob=json.dumps(payload,separators=(",",":"),sort_keys=True).encode()
            h=hashlib.sha256(blob).hexdigest()
            sig_ok=False
            try:
                VerifyKey(base64.b64decode(e["signer_pubkey"])).verify(blob, base64.b64decode(e["signature"]))
                sig_ok=True
            except Exception: pass
            status="OK" if (h==e["payload_hash"] and sig_ok) else "FAIL"
            print(f"[{status}] ts={e['timestamp']} dev={e['deviceId']} pred={e['prediction']} p={float(e['probability']):.3f}")
            last_ts=max(last_ts,float(e["timestamp"]))
    except Exception as ex:
        print("[verify] error:", ex)
    time.sleep(2)


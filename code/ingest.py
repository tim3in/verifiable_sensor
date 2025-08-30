# ingest_serial_sign_store.py
import os, json, base64, hashlib, time, sys
import requests, serial
from nacl import signing
from dotenv import load_dotenv

load_dotenv()
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
BAUD        = int(os.getenv("BAUD", "115200"))
GQL_URL     = os.getenv("GQL_URL", "http://localhost:9181/api/v0/graphql")
KEY_PATH    = os.getenv("SIGNING_KEY_PATH", "pi_signer_ed25519.key")

# Load or create Ed25519 private key (for signing)
if os.path.exists(KEY_PATH):
    sk = signing.SigningKey(open(KEY_PATH, "rb").read())
else:
    sk = signing.SigningKey.generate()
    open(KEY_PATH, "wb").write(sk.encode())
pk_b64 = base64.b64encode(sk.verify_key.encode()).decode()

# Canonical order must match verification
CANON_KEYS = ["deviceId", "timestamp", "prediction", "probability"]

def canonical_payload(evt: dict):
    # Keep only expected fields in a deterministic order
    return {k: evt.get(k) for k in CANON_KEYS}

def hash_payload(payload_dict):
    blob = json.dumps(payload_dict, separators=(",",":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), blob

def gql_create(event_with_proofs):
    # Your server exposes create_SensorEvent(input: [SensorEventMutationInputArg!]!)
    return {
        "query": """
          mutation($in: [SensorEventMutationInputArg!]!) {
            create_SensorEvent(input: $in) { _docID }
          }
        """,
        "variables": {"in": [event_with_proofs]}
    }

def main():
    print(f"[ingest] Serial={SERIAL_PORT}@{BAUD}  GraphQL={GQL_URL}")
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)

    last_hash = None  # simple de-dupe of consecutive identical payloads

    while True:
        line = ser.readline().decode("utf-8", "ignore").strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Minimal validation (match Arduino JSON)
        if not all(k in evt for k in ("deviceId","timestamp","prediction","probability")):
            continue

        payload = canonical_payload(evt)
        hexdig, blob = hash_payload(payload)

        # Skip consecutive duplicates
        if hexdig == last_hash:
            continue
        last_hash = hexdig

        sig_b64 = base64.b64encode(sk.sign(blob).signature).decode()

        body = dict(payload)
        body.update({
            "payload_hash": hexdig,
            "signature": sig_b64,
            "signer_pubkey": pk_b64
        })

        try:
            r = requests.post(GQL_URL, json=gql_create(body), timeout=5)
            # Print the raw response for visibility while testing
            print("[resp]", r.status_code, r.text)

            if r.ok:
                data = r.json()
                if "errors" in data:
                    # Ignore duplicate docID error (content-addressed collision)
                    msg = json.dumps(data["errors"])
                    if "already exists" in msg:
                        print("[dup] identical document already stored, skipping.")
                    else:
                        print("[err]", msg[:200])
                else:
                    print(f"[ok] {payload['deviceId']} ts={payload['timestamp']} "
                          f"pred={payload['prediction']} p={float(payload['probability']):.3f}")
            else:
                print("[http-err]", r.status_code, r.text[:200])

        except Exception as e:
            print("[net-err]", e)
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)


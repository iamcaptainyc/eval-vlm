import json
import urllib.request
import sys

BASE_URL = "http://127.0.0.1:8080"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"GET {path} failed: {resp.status}"
        return resp.read().decode("utf-8")

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"POST {path} failed: {resp.status}"
        return json.loads(resp.read().decode("utf-8"))

def put(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}, method="PUT")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"PUT {path} failed: {resp.status}"
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("1. Testing index.html & static assets...")
    html = get("/")
    assert "global-ds-dropdown" in html
    assert "view-sweep" in html
    assert "view-config" in html
    assert "view-settings" in html
    print("   [OK] index.html rendered with all new views.")

    css = get("/styles.css")
    assert ".global-ds-dropdown" in css
    assert ".config-studio-header" in css
    print("   [OK] styles.css served.")

    js = get("/app.js")
    assert "selectGlobalDataset" in js
    assert "saveAllConfigChanges" in js
    print("   [OK] app.js served.")

    print("2. Testing /api/datasets...")
    ds_data = json.loads(get("/api/datasets"))
    assert isinstance(ds_data, list)
    print(f"   [OK] datasets count: {len(ds_data)}")

    print("3. Testing /api/settings...")
    settings = json.loads(get("/api/settings"))
    assert "workspace" in settings
    assert "hf_models_dir" in settings
    print(f"   [OK] settings retrieved: {settings}")

    print("4. Testing /api/models...")
    models = json.loads(get("/api/models"))
    assert "hf_models" in models
    assert "mnn_models" in models
    print(f"   [OK] models scanned: hf={len(models['hf_models'])}, mnn={len(models['mnn_models'])}")

    print("5. Testing /api/datasets/demo_dataset/config...")
    cfg = json.loads(get("/api/datasets/demo_dataset/config"))
    assert "raw" in cfg
    assert "config" in cfg
    assert "settable_doc" in cfg
    print("   [OK] config retrieved successfully.")

    print("6. Testing /api/sweep/jobs with multi-datasets...")
    sweep_job = post("/api/sweep/jobs", {
        "type": "sweep",
        "dataset": "demo_dataset",
        "params": {
            "backend": "openai",
            "model": "gpt-4o-mini",
            "dry_run": True,
        }
    })
    assert sweep_job.get("status") in ["queued", "running", "succeeded"]
    assert "-d" in sweep_job.get("command", "")
    assert "demo_dataset" in sweep_job.get("command", "")
    print(f"   [OK] Sweep job created with command: {sweep_job.get('command')}")

    print("\nALL LIVE WEBUI TESTS PASSED!")

if __name__ == "__main__":
    main()

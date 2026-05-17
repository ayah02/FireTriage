import os, sys, traceback, tempfile

token = os.environ.get("HF_TOKEN", "")
print(f"Token present: {bool(token)}, length: {len(token)}, prefix: {token[:8] if token else 'EMPTY'}")

try:
    from huggingface_hub import HfApi, whoami
    info = whoami(token=token)
    print(f"Authenticated as: {info['name']}")
except Exception as e:
    print(f"AUTH FAILED: {e}")
    sys.exit(1)

api = HfApi(token=token)

try:
    url = api.create_repo(
        repo_id="AYAHAlqassab/FireTriage",
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        exist_ok=True,
    )
    print(f"Space ready: {url}")
except Exception as e:
    print(f"CREATE REPO FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

for key, env in [("GROQ_API_KEY", "GROQ_API_KEY"), ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY")]:
    val = os.environ.get(env, "")
    if val:
        try:
            api.add_space_secret("AYAHAlqassab/FireTriage", key, val)
            print(f"Secret {key} set.")
        except Exception as e:
            print(f"Secret {key} warning: {e}")

for fname in ["app.py", "requirements.txt"]:
    try:
        sha = api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id="AYAHAlqassab/FireTriage",
            repo_type="space",
        )
        print(f"Uploaded {fname}: {sha}")
    except Exception as e:
        print(f"UPLOAD {fname} FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

readme = """---
title: FireTriage — Project IgnitionHorizon
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: true
license: mit
short_description: Wildfire tactical triage dashboard with AI copilot
---
# FireTriage
Wildfire tactical triage dashboard with survival analysis ML and AI Copilot.
"""

with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
    f.write(readme)
    tmp = f.name

try:
    sha = api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="README.md",
        repo_id="AYAHAlqassab/FireTriage",
        repo_type="space",
    )
    print(f"Uploaded README.md: {sha}")
except Exception as e:
    print(f"UPLOAD README FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("DEPLOYMENT COMPLETE: https://huggingface.co/spaces/AYAHAlqassab/FireTriage")

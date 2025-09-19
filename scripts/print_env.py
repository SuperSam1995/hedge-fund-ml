import os, sys, platform, subprocess, json
def sh(x):
    try: return subprocess.check_output(x, shell=True, text=True).strip()
    except Exception as e: return f"ERR: {e}"
out = {
  "python": sys.version,
  "platform": platform.platform(),
  "cwd": os.getcwd(),
  "env": {k:v for k,v in os.environ.items() if k in ["DATA_ROOT","CUDA_VISIBLE_DEVICES"]},
  "git_commit": sh("git rev-parse --short HEAD"),
  "pip_freeze_head": sh("python -m pip freeze | head -n 20")
}
print(json.dumps(out, indent=2))

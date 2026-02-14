import subprocess, sys, webbrowser, time, threading
from pathlib import Path

_PIP_TO_IMPORT = {"scikit-learn": "sklearn", "python-multipart": "multipart", "uvicorn[standard]": "uvicorn", "dask[dataframe]": "dask"}

def _read_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists(): return []
    return [line.strip() for line in req_file.read_text().splitlines() if line.strip() and not line.startswith('#')]

def check_dependencies():
    from importlib.util import find_spec
    reqs = _read_requirements()
    missing = [r for r in reqs if find_spec(_PIP_TO_IMPORT.get(r, r.split('[')[0])) is None]
    if not missing: return print("[OK] All dependencies found")
    print(f"[*] Installing {len(missing)} missing: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])
    print("[OK] Dependencies installed")

def main():
    print("=" * 40 + "\n  SmartData Optimizer - Starting...\n" + "=" * 40)
    try: check_dependencies()
    except Exception as e: print(f"[!] Warning: {e}")
    print(f"\n[>] Starting server at http://localhost:8080\nPress Ctrl+C to stop\n")
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open("http://localhost:8080")), daemon=True).start()
    import uvicorn
    uvicorn.run("smartOp.app:app", host="0.0.0.0", port=8080, reload=False)

if __name__ == "__main__":
    main()

import subprocess, sys, webbrowser, time, threading, socket
from pathlib import Path

_PIP_TO_IMPORT = {"scikit-learn": "sklearn", "python-multipart": "multipart", "uvicorn[standard]": "uvicorn", "dask[dataframe]": "dask", "defusedxml": "defusedxml"}

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

def _free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # port already free
    print(f"[*] Port {port} in use, killing occupying process...")
    if sys.platform == "win32":
        out = subprocess.run(f"netstat -ano | findstr :{port}", capture_output=True, text=True, shell=True).stdout
        pids = {line.split()[-1] for line in out.strip().splitlines() if "LISTENING" in line}
        for pid in pids:
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
    else:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
    time.sleep(0.5)
    print(f"[OK] Port {port} freed")

def main():
    print("=" * 40 + "\n  SmartData Optimizer - Starting...\n" + "=" * 40)
    try: check_dependencies()
    except Exception as e: print(f"[!] Warning: {e}")
    from smartOp.config import SERVER_PORT as _PORT
    _free_port(_PORT)
    print(f"\n[>] Starting server at http://localhost:{_PORT}\nPress Ctrl+C to stop\n")
    threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(f"http://localhost:{_PORT}")), daemon=True).start()
    import uvicorn
    uvicorn.run("smartOp.app:app", host="0.0.0.0", port=_PORT, reload=False)
if __name__ == "__main__":
    main()

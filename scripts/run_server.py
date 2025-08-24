import subprocess
import time

fail_count = 0
while True:
    error_occurred = False
    try:
        result = subprocess.run(["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000",  "--proxy-headers"]);
    except KeyboardInterrupt:
        error_occurred = True
    except Exception:
        error_occurred = True
    else:
        if result.returncode == 0:
            break
        else:
            error_occurred = True
    if error_occurred:
        fail_count += 1
        if fail_count >= 100:
            raise RuntimeError("Server crashed 100 times in a row, aborting.")
        print("[runner] server crashed, restarting...")
        time.sleep(1)

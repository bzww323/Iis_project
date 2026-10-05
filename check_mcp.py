import os
import sys
import subprocess

# Перезапускаем Python с флагом -X utf8 и -u (unbuffered)
if not os.environ.get("MCP_FIXED"):
    env = os.environ.copy()
    env["MCP_FIXED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Запускаем тот же скрипт, но как подпроцесс с правильным окружением
    # -X utf8 включает UTF-8 режим, -u отключает буферизацию
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-u", __file__],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    sys.exit(result.returncode)

# --- Основной код ---
from orchestrator import Orchestrator

o = Orchestrator()
try:
    for c in o.tools.search("Что такое self-attention?"):
        page = f" стр. {c.page}" if c.page else ""
        print(f"{c.score:.3f} {c.source}{page} [{c.category}]")
finally:
    o.close()
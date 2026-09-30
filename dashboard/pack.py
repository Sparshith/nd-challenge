# Exports the baseline into assets/baseline.onnx for the page, with the starter's own export script.
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / "assets").mkdir(exist_ok=True)
subprocess.run([sys.executable, str(ROOT.parent / "starter/export.py"), "--int4", "--output", str(ROOT / "assets/baseline.onnx")], check=True)

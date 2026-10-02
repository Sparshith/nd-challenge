"""Build a standalone GitHub Pages artifact in output/site."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site = ROOT.parent / 'output/site'

subprocess.run([sys.executable, str(ROOT / 'build.py')], check=True)
subprocess.run([sys.executable, str(ROOT / 'pack.py')], check=True)
files = ['index.html', 'style.css', 'app.bundle.js', 'worker.bundle.js', 'ort/ort-wasm-simd-threaded.wasm',
         'ort/ort-wasm-simd-threaded.mjs', 'assets/baseline.onnx']
if site.exists():
    shutil.rmtree(site)
site.mkdir(parents=True)
for name in files:
    (site / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, site / name)
(site / '.nojekyll').touch()
total = sum((site / name).stat().st_size for name in files)
print(f"{site}: {len(files)} runtime files, {total / 1e6:.1f} MB")

# Builds the page and copies the files it needs at run time into the blog, for GitHub Pages.
#   python3 deploy.py --to /path/to/blog/nd-challenge
#   python3 deploy.py --to /some/dir
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--to', type=Path, required=True, help='directory in the site repository to receive the built page')
args = parser.parse_args()

subprocess.run([sys.executable, str(ROOT / 'build.py')], check=True)
subprocess.run([sys.executable, str(ROOT / 'pack.py')], check=True)
files = ['index.html', 'style.css', 'app.bundle.js', 'worker.bundle.js', 'ort/ort-wasm-simd-threaded.wasm',
         'ort/ort-wasm-simd-threaded.mjs', 'assets/baseline.onnx']
args.to.mkdir(parents=True, exist_ok=True)
for name in files:
    (args.to / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, args.to / name)
total = sum((args.to / name).stat().st_size for name in files)
print(f"{args.to}: {len(files)} files, {total / 1e6:.1f} MB")

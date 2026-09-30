import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ('app', 'worker'):
    subprocess.run([str(ROOT / 'node_modules/.bin/esbuild'), str(ROOT / f'{name}.js'),
                    '--bundle', '--format=esm', '--loader:.tsv=text', '--target=chrome120', f'--outfile={ROOT / (name + ".bundle.js")}'], check=True)
ort = ROOT / 'ort'
ort.mkdir(exist_ok=True)
for name in ('ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'):
    shutil.copyfile(ROOT / 'node_modules/onnxruntime-web/dist' / name, ort / name)

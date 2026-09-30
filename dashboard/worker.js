// Runs one ONNX model with ONNX Runtime Web, off the page thread.
import * as ort from 'onnxruntime-web/wasm';

ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;

let session, inputName;
self.onmessage = async ({data}) => {
  const {id, type} = data;
  try {
    if (type === 'load') {
      session = await ort.InferenceSession.create(data.bytes, {executionProviders: ['wasm']});
      inputName = session.inputNames[0];
      self.postMessage({id, result: {inputs: session.inputNames, outputs: session.outputNames}});
    } else if (type === 'predict') {
      const tokens = BigInt64Array.from(data.tokens, BigInt);
      const outputs = await session.run({[inputName]: new ort.Tensor('int64', tokens, [1, tokens.length])});
      const logits = outputs[session.outputNames[0]];
      const size = logits.dims.at(-1), all = logits.data;
      self.postMessage({id, result: Float32Array.from(all.subarray(all.length - size))});
    }
  } catch (error) { self.postMessage({id, error: error.message || String(error)}); }
};

// Starts a worker for one model and exposes load/predict calls with timeouts.
export function createRunner() {
  const worker = new Worker('worker.bundle.js', {type: 'module'});
  const pending = new Map();
  let sequence = 0, stopped = false;
  worker.onmessage = ({data}) => {
    const request = pending.get(data?.id);
    if (!request) return;
    clearTimeout(request.timer); pending.delete(data.id);
    if (data.error) request.reject(new Error(data.error)); else request.resolve(data.result);
  };
  worker.onerror = event => { for (const request of pending.values()) request.reject(new Error(event.message || 'Model worker failed.')); pending.clear(); };
  const stop = () => {
    stopped = true; worker.terminate();
    for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error('Model stopped.')); }
    pending.clear();
  };
  return {
    stop,
    call(type, payload = {}, timeout = 120_000) {
      if (stopped) return Promise.reject(new Error('Model stopped.'));
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        const timer = setTimeout(() => { reject(new Error('Model timed out. It has been stopped; check again to retry.')); stop(); }, timeout);
        pending.set(id, {resolve, reject, timer});
        worker.postMessage({id, type, ...payload}, payload.bytes ? [payload.bytes.buffer] : []);
      });
    },
  };
}

import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {readEntry, readOnnxMeta} from './onnx_meta.js';

const varint = n => { const out = []; do { let byte = n & 127; n >>= 7; if (n) byte |= 128; out.push(byte); } while (n); return out; };
const bytes = (field, content) => [field << 3 | 2, ...varint(content.length), ...content];
const text = s => [...new TextEncoder().encode(s)];
const entry = (key, value) => bytes(14, [...bytes(1, text(key)), ...bytes(2, text(value))]);
const model = (...props) => Uint8Array.from([8, 9, ...bytes(8, [...bytes(1, []), 16, 18]), ...bytes(7, text('graph bytes')), ...props.flat()]);

test('reads metadata and opset without touching the graph', () => {
  const meta = readOnnxMeta(model(entry('name', 'Stub'), entry('vocabulary', '["<bos>","e4"]'), entry('context_length', '256')));
  assert.equal(meta.irVersion, 9);
  assert.deepEqual(meta.opsets, [{domain: '', version: 18}]);
  assert.deepEqual(meta.metadata, {name: 'Stub', vocabulary: '["<bos>","e4"]', context_length: '256'});
  const parsed = readEntry(model(entry('vocabulary', '["<bos>","e4"]')));
  assert.deepEqual(parsed, {name: '', vocabulary: ['<bos>', 'e4'], contextLength: 512, opset: 18});
});

test('rejects files without vocabulary metadata or that are not ONNX', () => {
  assert.throws(() => readEntry(model()), /vocabulary/);
  assert.throws(() => readOnnxMeta(new TextEncoder().encode('PK not onnx')), /ONNX/);
});

test('the starter export carries the default vocabulary', async t => {
  const path = new URL('../starter/model.onnx', import.meta.url);
  const file = await readFile(path).catch(() => null);
  if (!file) return t.skip('starter not exported');
  const parsed = readEntry(new Uint8Array(file));
  assert.equal(parsed.vocabulary.length, 14750);
  assert.equal(parsed.vocabulary[0], '<bos>');
  assert.equal(parsed.contextLength, 512);
});

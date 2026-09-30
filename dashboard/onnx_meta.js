// Reads the metadata and opset from an ONNX file without loading the graph.
// ONNX files are protobuf: ModelProto field 8 is opset_import, field 14 is metadata_props.
const decoder = new TextDecoder();

function* fields(bytes) {
  let at = 0;
  const varint = () => { let value = 0n, shift = 0n; for (;;) { const byte = bytes[at++]; value |= BigInt(byte & 127) << shift; if (byte < 128) return value; shift += 7n; if (at > bytes.length) throw new Error('Truncated ONNX file.'); } };
  while (at < bytes.length) {
    const tag = Number(varint()), field = tag >> 3, wire = tag & 7;
    if (wire === 0) yield {field, value: varint()};
    else if (wire === 2) { const length = Number(varint()); yield {field, bytes: bytes.subarray(at, at + length)}; at += length; }
    else if (wire === 1) at += 8;
    else if (wire === 5) at += 4;
    else throw new Error('Not an ONNX file.');
  }
}

export function readOnnxMeta(bytes) {
  const metadata = {}, opsets = [];
  let irVersion = 0;
  for (const item of fields(bytes)) {
    if (item.field === 1) irVersion = Number(item.value);
    else if (item.field === 8 && item.bytes) {
      let domain = '', version = 0;
      for (const part of fields(item.bytes)) { if (part.field === 1) domain = decoder.decode(part.bytes); else if (part.field === 2) version = Number(part.value); }
      opsets.push({domain, version});
    } else if (item.field === 14 && item.bytes) {
      let key = '', value = '';
      for (const part of fields(item.bytes)) { if (part.field === 1) key = decoder.decode(part.bytes); else if (part.field === 2) value = decoder.decode(part.bytes); }
      metadata[key] = value;
    }
  }
  if (!irVersion) throw new Error('Not an ONNX file.');
  return {irVersion, opsets, metadata};
}

export function readEntry(bytes) {
  const {metadata, opsets} = readOnnxMeta(bytes);
  if (!metadata.vocabulary) throw new Error('The ONNX file has no "vocabulary" metadata. Export it with the starter script.');
  let vocabulary;
  try { vocabulary = JSON.parse(metadata.vocabulary); } catch { throw new Error('The "vocabulary" metadata is not a JSON array.'); }
  const contextLength = Number(metadata.context_length || 512);
  return {name: metadata.name || '', vocabulary, contextLength, opset: opsets.find(o => !o.domain)?.version ?? 0};
}

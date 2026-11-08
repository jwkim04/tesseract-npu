#!/usr/bin/env python3
"""gguf_truncate.py IN.gguf OUT.gguf N: copy a GGUF with only blocks 0 .. N-1.

The block count becomes N, the multi-token-prediction block count 0, and a per-layer array (such as
qwen35.attention.recurrent_layers) is cut to N entries. Tensors outside the blocks (embedding, output norm, output
head) are copied as they are. The result is not a useful language model; it has the real layer shapes of a model that
is too big to simulate whole, and its first N layers compute what the first N layers of the full model compute. Used
for the per-layer cost extrapolation of docs/design/06-software-verification.md section 7.9 (a hybrid Qwen3.5 / 3.8
group is 4 layers: N = 4 and N = 8 give one group by difference).

Needs numpy and the pinned llama.cpp's gguf-py:
  PYTHONPATH=third_party/llama.cpp/gguf-py python sw/tools/gguf_truncate.py models/big.gguf models/big-first4.gguf 4
"""
import re
import sys

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = GGUFReader(src)
    arch = [bytes(f.parts[f.data[0]]).decode() for f in r.fields.values() if f.name == "general.architecture"][0]
    orig_n = [f.contents() for f in r.fields.values() if f.name == f"{arch}.block_count"][0]
    if not 0 < n <= orig_n:
        print(f"N must be in 1 .. {orig_n}")
        return 2
    w = GGUFWriter(dst, arch)
    for name, f in r.fields.items():
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
        t = f.types[0]
        if name == f"{arch}.nextn_predict_layers":
            w.add_uint32(name, 0)
        elif name == f"{arch}.block_count":
            w.add_uint32(name, n)
        elif t == GGUFValueType.ARRAY:
            vals = f.contents()
            w.add_array(name, vals[:n] if len(vals) == orig_n else vals)
        elif t == GGUFValueType.STRING:
            w.add_string(name, f.contents())
        else:
            w.add_key_value(name, f.contents(), t)
    kept = [t for t in r.tensors if not (re.match(r"blk\.(\d+)\.", t.name) and int(re.match(r"blk\.(\d+)\.", t.name).group(1)) >= n)]
    for t in kept:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in kept:
        w.write_tensor_data(t.data)
    w.close()
    print(f"kept {len(kept)} tensors, block_count {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

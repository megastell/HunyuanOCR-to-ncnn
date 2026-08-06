# Dynamic Decoder KV Cache Milestone

## Scope

This milestone replaces the three fixed decode exports with one dynamic ncnn
model per decoder layer. Each model accepts a variable past-key/value length and
keeps the six-input, three-output decoder contract used by the C++ chain.

The exporter is `tools/export/export_decoder_dynamic.py`. It traces each of the
24 layers with two cache lengths so pnnx preserves the sequence dimension as
dynamic, then validates the resulting TorchScript model at decode steps 1, 2,
and 3.

## Verified Results

- Dynamic export reports: 24/24.
- TorchScript parity metrics: 216/216 have zero maximum and mean error.
- pnnx logs with a dynamic cache dimension: 24/24.
- Dynamic and fixed Step 1 ncnn weight binaries are byte-identical: 24/24.
- C++ packed three-token chain: `5112 -> 206 -> 1717` passed.
- C++ unpacked three-token chain: `5112 -> 206 -> 1717` passed.
- Tested cache transitions: `313 -> 314`, `314 -> 315`, and `315 -> 316`.

## Evidence

- Per-layer export reports: `docs/decoder_layer*_decode_dynamic.json`.
- Per-layer pnnx logs: `docs/decoder_layer*_decode_dynamic_pnnx.txt`.
- Layer 0 packed/unpacked runtime reports:
  `docs/decoder_layer0_dynamic_step{1,2,3}_{packed,unpacked}.txt`.
- Persistent chain logs:
  `~/hunyuanocr-recovery/three_token_chain_dynamic_{packed,unpacked}.log`.
- Persistent audit log: `~/hunyuanocr-recovery/dynamic_kv_audit.log`.
- Persistent audit script: `~/hunyuanocr-recovery/audit_dynamic_kv.sh`.

## Remaining Boundary

This stage validates the reusable decoder model and live KV handoff for three
decode iterations. It still starts from PyTorch-captured prefill hidden state
and KV caches. General generation through EOS is saved as the next milestone.

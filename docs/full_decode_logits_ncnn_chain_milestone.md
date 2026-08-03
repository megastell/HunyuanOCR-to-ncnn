# Full Single-token Decode Logits ncnn Milestone

## Scope

This milestone validates the complete HunyuanOCR-1.5
single-token text decode calculation path:

Decoder Layer 0
→ Decoder Layer 1
→ ...
→ Decoder Layer 23
→ final RMSNorm
→ LM Head
→ vocabulary logits

## Runtime

- device: CPU
- precision: FP32
- decoder layers: 24
- threads: 9
- Vulkan: disabled
- packing layout: enabled
- ncnn version: 20260802
- compiler warnings: 0
- vocabulary size: 120818

## Tensor handoff

The runtime reports:

- intermediate reload: disabled
- decoder handoff: previous out0.clone() -> next in0

Each decoder layer output is cloned into an independently
owned ncnn::Mat before the current layer network is released.

The Layer 23 ncnn output is passed directly to final RMSNorm.
The final RMSNorm ncnn output is then cloned and passed
directly to LM Head.

Reference hidden-state tensors participate only in numerical
validation. They do not replace intermediate inference
inputs.

## Decoder numerical results

Maximum errors across Decoder Layer 0 through Layer 23:

- maximum input-boundary error: 2.8312206268e-07
- maximum layer-output error: 7.1525573730e-07
- maximum present-key error: 3.9935112000e-06
- maximum present-value error: 2.3841857910e-07

All 24 layers preserve the key and value history prefixes
with zero error.

## Decoder-to-final-norm boundary

Layer 23 chained output versus the PyTorch final RMSNorm
reference input:

- maximum absolute error: 7.1525573730e-07
- mean absolute error: 9.7184056358e-08
- RMSE: 1.2211550740e-07
- cosine similarity: 1

## Final RMSNorm results

- maximum absolute error: 1.3351440430e-05
- mean absolute error: 2.4471476081e-06
- RMSE: 3.0681087029e-06
- cosine similarity: 1

## Full logits results

The final output contains 120818 FP32 vocabulary logits.

- maximum absolute error: 1.1503696442e-05
- mean absolute error: 2.2112453449e-06
- RMSE: 2.7531951306e-06
- cosine similarity: 1

Token results:

- expected decode token: 5112
- ncnn decode token: 5112
- reference-contract token: 5112

## Determinism

The complete packed decode logits chain was executed nine
times using nine CPU threads.

All runs completed successfully and produced identical
numerical summaries.

## Result

The complete HunyuanOCR-1.5 single-token decoder calculation
path is validated as a packed ncnn FP32 CPU pipeline.

The validated path starts with the Layer 0 decode hidden
state and ends with the 120818-element vocabulary logits and
decode token 5112.

No PyTorch, NumPy, or binary hidden-state conversion is used
between ncnn decoder layers, final RMSNorm, and LM Head.

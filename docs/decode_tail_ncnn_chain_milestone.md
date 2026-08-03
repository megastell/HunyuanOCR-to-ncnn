# Single-token Decode Tail ncnn Milestone

## Scope

This milestone validates the HunyuanOCR-1.5 single-token
decode tail:

Layer 23 output
→ final RMSNorm
→ LM Head
→ vocabulary logits

## Reference contract

The reference tensors belong to the same single-token decode
iteration.

- hidden size: 1024
- vocabulary size: 120818
- final RMSNorm epsilon: 1e-5
- expected decode token: 5112

The PyTorch tail output is element-wise identical to the
logits captured from the complete model.

## ncnn execution

Runtime configuration:

- device: CPU
- precision: FP32
- threads: 9
- Vulkan: disabled
- packing layout: enabled
- ncnn version: 20260802
- intermediate hidden reload: disabled

Tensor handoff:

1. Layer 23 output is supplied directly to final RMSNorm.
2. final RMSNorm out0 is cloned into an independently owned
   ncnn::Mat.
3. That ncnn::Mat is supplied directly to LM Head in0.

## Numerical results

Layer 23 output versus final RMSNorm reference input:

- maximum absolute error: 0
- mean absolute error: 0
- cosine similarity: 1

Final RMSNorm output:

- maximum absolute error: 1.9073486328e-06
- mean absolute error: 3.0571140996e-07
- RMSE: 4.1563692362e-07
- cosine similarity: 1

Final logits:

- maximum absolute error: 4.7683715820e-06
- mean absolute error: 5.7507581807e-07
- RMSE: 7.4756836746e-07
- cosine similarity: 1

Token result:

- expected token: 5112
- ncnn token: 5112

## Result

The final RMSNorm and LM Head are validated as a directly
chained packed ncnn FP32 CPU decode tail.

No PyTorch, NumPy, or binary hidden-state conversion is used
between final RMSNorm and LM Head.

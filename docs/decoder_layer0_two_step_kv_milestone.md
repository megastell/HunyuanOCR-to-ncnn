# Decoder Layer 0 Two-step KV ncnn Milestone

## Scope

This milestone validates direct KV-cache transfer across two
consecutive HunyuanOCR-1.5 single-token decode iterations for
Decoder Layer 0.

Step 1:

- past KV length: 313
- present KV length: 314

Step 2:

- past KV length: 314
- present KV length: 315

## Runtime

- device: CPU
- precision: FP32
- threads: 9
- Vulkan: disabled
- packed and unpacked execution validated
- ncnn version: 20260802
- compiler warnings: 0

## Direct KV handoff

Step 1 outputs are copied before the Step 1 network is
released:

- Step 1 out1.clone() becomes Step 2 in4
- Step 1 out2.clone() becomes Step 2 in5

Step 2 does not reload its past key or past value from a
reference file.

Reference past-KV tensors participate only in numerical
boundary validation.

## Step 1 results

Layer output:

- maximum absolute error: 6.7055225372e-08
- mean absolute error: 8.7139824245e-09
- cosine similarity: 1

Present key:

- maximum absolute error: 2.3841857910e-06
- mean absolute error: 7.6836643205e-10
- cosine similarity: 1

Present value:

- maximum absolute error: 2.9802322388e-08
- mean absolute error: 1.4355859742e-11
- cosine similarity: 1

## Step 1 to Step 2 boundary

The directly transferred ncnn KV cache differs from the
PyTorch Step 2 past-KV reference only by the normal Step 1
ncnn numerical error.

- key maximum absolute error: 2.3841857910e-06
- value maximum absolute error: 2.9802322388e-08
- key cosine similarity: 1
- value cosine similarity: 1

## Step 2 results

Layer output:

- maximum absolute error: 1.1920928955e-07
- mean absolute error: 1.3850012692e-08
- cosine similarity: 1

Present key:

- maximum absolute error: 2.3841857910e-06
- mean absolute error: 1.5037706304e-09
- cosine similarity: 1

Present value:

- maximum absolute error: 2.9802322388e-08
- mean absolute error: 2.8660271334e-11
- cosine similarity: 1

## Cache integrity

- Step 2 key prefix versus Step 1 handoff: exactly zero
- Step 2 value prefix versus Step 1 handoff: exactly zero
- Step 2 appended key error: 1.4305114746e-06
- Step 2 appended value error: 2.6077032089e-08

The first 314 positions of the Step 2 output cache are
therefore exact copies of the ncnn cache supplied by Step 1.

## Result

Decoder Layer 0 successfully preserves and transfers its
ncnn-generated KV cache across two consecutive decode steps.

No PyTorch, NumPy, or binary-file conversion occurs between
the Step 1 present KV and the Step 2 past KV.

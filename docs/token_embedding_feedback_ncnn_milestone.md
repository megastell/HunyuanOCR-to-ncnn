# Token Embedding ncnn Feedback Milestone

## Scope

This milestone validates the standard ncnn Embed path required for
autoregressive token feedback.

## Runtime path

- token ID: 5112
- input type: int32
- ncnn layer: Embed
- vocabulary size: 120818
- hidden size: 1024
- output shape: one token by 1024 hidden elements

## Model

The validated model is:

- `artifacts/text_embedding/text_embedding.ncnn.param`
- `artifacts/text_embedding/text_embedding.ncnn.bin`

Its Embed contract is:

- `num_output = 1024`
- `input_dim = 120818`
- `weight_data_size = 123717632`

The text-embedding and LM Head ncnn weight files are byte-identical,
consistent with the model's tied input/output weights.

## Numerical result

Both packed and unpacked FP32 CPU execution produced:

- maximum absolute error: 0
- mean absolute error: 0
- RMSE: 0
- cosine similarity: 1
- byte-identical output: true

The ncnn output for token 5112 is therefore an exact copy of the
PyTorch token-embedding output and the Step 2 Layer 0 reference
hidden state.

## Next milestone

The next runtime must execute:

Step 1 Decoder Layer 0-23
-> Final RMSNorm
-> LM Head
-> argmax token 5112
-> ncnn Embed
-> Step 2 Decoder Layer 0

The Step 2 initial hidden reference reload must then be removed.

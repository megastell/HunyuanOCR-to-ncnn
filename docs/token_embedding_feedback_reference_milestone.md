# Token Embedding Feedback Reference Milestone

## Scope

This milestone validates the hidden-state feedback contract between
the first and second single-token decode iterations.

The Step 1 decoder and LM Head produce token 5112. The model input
embedding for token 5112 is compared against the PyTorch reference
hidden state supplied to Step 2 Decoder Layer 0.

## Model structure

- input embedding module: model.language_model.embed_tokens
- output embedding module: lm_head
- embedding weight shape: [120818, 1024]
- embedding dtype: torch.float32
- token ID: 5112
- hidden size: 1024
- embedding output shape: [1, 1, 1024]

## Numerical contract

- maximum embedding versus Step 2 hidden error:
  0.0000000000e+00
- mean embedding versus Step 2 hidden error:
  0.0000000000e+00
- byte-identical:
  True
- embedding output versus weight-row maximum error:
  0.0000000000e+00

## Tied weights

- input Embedding and LM Head share weight storage:
  True
- Embedding row 5112 versus LM Head row 5112 maximum error:
  0.0000000000e+00

## Result

The following feedback contract is exact:

Step 1 output token 5112
-> token Embedding row 5112
-> Step 2 Decoder Layer 0 hidden input

No additional scaling, normalization, position embedding, or hidden
transformation is required between token lookup and Step 2 Layer 0.

This milestone validates the PyTorch reference relationship only.
The runtime still loads the Step 2 initial hidden state from a
reference binary. The next milestone must generate that hidden state
inside the C++/ncnn runtime from token 5112.

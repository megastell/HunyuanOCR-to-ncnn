# ncnn full-width RotaryEmbed dependency

HunyuanOCR-1.5 uses multidimensional RoPE with:

- head dimension: 128
- mrope_section: [16, 16, 16, 16]
- full-width cos/sin cache

## Base ncnn

- upstream base commit:
  a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a
- original installed version:
  20260730

## Required RotaryEmbed fix

- upstream source commit:
  5967676e
- local cherry-picked commit:
  6cc4ef9d
- local branch:
  experiment/fullwidth-rotaryembed-6834
- patched installed version:
  20260801
- patched install prefix:
  ~/.local/ncnn-cpu-ropefix

Without this patch, the first significant numerical divergence occurs
at the Q/K RotaryEmbed outputs.

Before the fix:

- Q after RoPE max error: 2.399411e-01
- K after RoPE max error: 3.215269e-01
- Decoder layer output max error: about 8.53e-01

After the fix:

- Q after RoPE max error: 3.427267e-07
- K after RoPE max error: 3.129244e-07
- intermediate layer output max error: 4.577637e-05
- final parity cosine similarity: 0.9999999998
- final parity exit status: 0

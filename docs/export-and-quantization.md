# ONNX export and INT8 quantization

`DCT2d`/`IDCT2d` have no trainable parameters, so in a quantized model they are fixed
preprocessing. What matters is that they export cleanly and that the quantized layers after them
read their output well.

## Export

Under `torch.onnx.export` (both the TorchScript and the `dynamo=True` exporter), `torch.export`,
`torch.compile` and `torch.jit.trace`, the PyTorch implementation is always used. The
[CPU kernel](cpu-kernel.md) steps aside, so the graph contains only standard ops:

| Part | ONNX ops |
| --- | --- |
| forward transform | `Conv` (strided, constant weights) |
| inverse transform | `MatMul` + `DepthToSpace` |
| per-channel `selections` | `Gather` (forward), `ScatterND` / `ScatterElements` (inverse) |
| integer inputs | `Cast` to `float32` |

The forward transform is the node named after the module, e.g. `/dct/Conv` for a submodule
called `dct`.

```python
torch.onnx.export(model, (example_input,), "model.onnx", opset_version=17)
```

## Quantized `torch.ao` tensors

Quantized tensors (e.g. after a `QuantStub` in eager-mode quantization) are dequantized and
transformed as float, returning `float32`.

## Quantizing a model that uses the transform

DCT coefficients have very different ranges. The DC term goes up to `8 * 255` for `uint8` images,
while most AC terms are in the tens. A single per-tensor int8 scale, which ONNX Runtime and
TensorRT use for activations by default, is set by the DC term and rounds most AC coefficients
to zero. Measured on natural-image-like data with all 64 coefficients:

| int8 quantization of the DCT output | AC SNR | AC coefficients zeroed |
| --- | --- | --- |
| per-tensor | 13 dB | 83% |
| per-tensor, after centering (`x - 128`) | 18 dB | 55% |
| one scale per coefficient channel | 40 dB | 1% |

So:

1. **Keep the DCT node in float.** For ONNX Runtime, use
   `quantize_static(..., nodes_to_exclude=["/dct/Conv"])`.
2. **Give each coefficient channel its own range** before the first quantized layer: quantize
   the coefficients per channel, or rescale each coefficient channel to a similar range.

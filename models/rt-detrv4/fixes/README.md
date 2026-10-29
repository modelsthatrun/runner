# RT-DETRv4: runs with a fix

The README says to clone the repository and run `tools/inference/torch_inf.py`. At the current
`main` (55fefaa, checked 2026-10-27) that fails on CUDA:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  engine/rtv4/hybrid_encoder.py, in with_pos_embed: tensor + pos_embed
```

Pull request #20 (merged 2026-05-22, commit bba6458 "perf: enable torch.compile by avoiding .to
usage") stopped moving the encoder's fixed position embedding to the input's device. Its comment
says an "inference wrapper" registers it as a buffer, but the repository has no such wrapper, so
the embedding stays on the CPU when the model runs on the GPU.

The fix is to run the last `main` commit before that merge,
df4412261c6162f62aa4fb84e90b2df3c1b5f223 (2026-01-09). It holds the released code
(64c0e34, "released code and weights") plus a torchvision fix and README updates. No code is
patched. `adapter.py` pins this commit.

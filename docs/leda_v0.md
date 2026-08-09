# Leda v0 reference specification

Leda is the first concrete language-model family built with Spar. Version 0 is a correctness-first
reference definition, not a competitive pretrained model or a production-scale claim.

## Architecture

Leda v0 is a dense causal decoder-only Transformer:

```text
token embedding
N × pre-norm Transformer block
  RMSNorm
  grouped-query causal self-attention
    QK-Norm
    rotary position embeddings (RoPE)
  residual connection
  RMSNorm
  SwiGLU MLP
  residual connection
final RMSNorm
tied token-embedding projection
vocabulary logits
```

The reference presets use no attention or MLP biases. The output projection reuses the token
embedding Parameter and is not counted twice. Leda owns one ordinary `spar::nn::DecoderLM`; its
configuration is mapped field by field and it does not duplicate the neural-network implementation.

## Tokenization and document boundaries

Leda v0 uses Spar's raw-byte BPE tokenizer. Tokenizer artifacts remain `SPARTOKN v1` and contain raw
bytes plus learned merges only. The packed data layer reserves one model-level EOD token at
`tokenizer.vocab_size()`, so the decoder vocabulary has one additional entry. Every source document,
including an empty or final document, receives exactly one EOD. Reference windows may attend across
an EOD boundary; there is no boundary-reset mask in v0.

## Reference presets

These are CPU reference-development presets:

| preset | model dim | hidden dim | layers | query heads | KV heads | QK-Norm |
|---|---:|---:|---:|---:|---:|---:|
| `leda_tiny` | 128 | 384 | 4 | 4 | 2 | yes |
| `leda_small` | 256 | 768 | 8 | 8 | 2 | yes |

Both default to Float32, RMSNorm epsilon `1e-5`, QK-Norm epsilon `1e-6`, and RoPE theta `10000`.

For a representative model vocabulary of 257, the independently audited no-bias parameter formula is

```text
V*D
+ L * (2*D*D + 2*D*(D*Hkv/Hq) + 2*(D/Hq) + 3*D*hidden + 2*D)
+ D
```

giving 820,736 Parameters for `leda_tiny` and 6,099,968 for `leda_small`. This includes the token
embedding once and the final RMSNorm.

## Reference pretraining recipe

The Phase 19 local reference run uses deterministic raw-document splitting before tokenization,
trains ByteBPE on training documents only, and constructs separate train and validation corpora.
Its compact smoke-test recipe uses:

```text
sequence length:            16
stride:                     12
microbatch size:            1
accumulation steps:         2
AdamW beta1 / beta2:        0.9 / 0.95
AdamW epsilon:              1e-8
weight decay:               0.01
maximum gradient norm:      1.0
learning-rate schedule:     linear warmup then cosine decay
peak / minimum LR:          2e-3 / 2e-4
warmup / decay steps:       2 / 12
```

Each microbatch uses Sum-reduced language-model cross-entropy. Gradients are divided once by the
actual number of next-token targets in the accumulation group, clipped once, and passed to AdamW.
`TrainingProgress.global_step` counts optimizer updates; `tokens_seen` counts objective targets.
Final partial accumulation groups are retained.

The executable uses 12 synthetic, original science-prose documents, holds out two documents before
tokenizer training, performs 12 optimizer updates, evaluates at the beginning, midpoint, and end,
and performs a checkpoint save/load round-trip after update 6. Seeds for splitting, shuffling, and
model initialization are fixed in the source. Its loss check requires final held-out cross-entropy
to be finite and below the initial held-out value; wall-clock timings are diagnostic only.

Checkpointing remains `SPARCKPT v1` at a clean optimizer boundary. Leda v0 checkpoint compatibility
is defined by its mapped `DecoderConfig` and canonical `DecoderLM` Parameters. Iterator state remains
an adjacent caller-owned value. The stateless learning-rate schedule is reconstructed from the loaded
`global_step`.

One deterministic Release run with tokenizer vocabulary 272 (model vocabulary 273) reduced full
training CE from 5.814369 to 4.183725 and held-out CE from 5.809016 to 4.292330. It completed 12
updates and 360 objective targets. The observed training interval, including the midpoint validation
and checkpoint round-trip, was 13.866 seconds: 26.0 targets/s, 0.865 updates/s, or 1155.5 ms/update.
Only the loss values are reproducibility expectations; timing depends on the machine and build.

## Memory and performance constraints

The reported persistent AdamW estimate includes Parameter, gradient, first-moment, and second-moment
storage. It excludes activations, autograd graph metadata, temporary operator tensors, allocator
overhead, and data batches.

The attention score tensor scales approximately as

```text
B × Hq × T × T
```

elements per attention operation. This quadratic term, along with reference C++ matrix multiplication
and backward execution, makes long-context CPU training expensive. Evaluation also constructs
temporary autograd graphs because Spar does not yet provide a no-grad mode.

On the Phase 19 arm64 macOS Release baseline, one `leda_tiny` update took approximately 1.05, 2.15,
and 4.61 seconds at sequence lengths 32, 64, and 128. Backward occupied about 67% of update time.
The isolated causal-attention forward observation grew from 0.0040 to 0.0161 to 0.0632 seconds,
showing the expected quadratic trend. These are coarse observations from one run, not benchmark
contracts; use `spar_leda_profile` to measure the current machine.

## Current limitations

- CPU reference implementation only
- full quadratic attention
- Float32 and Float64 only
- no CUDA or vendor numerical libraries
- no distributed training
- no FlashAttention or fused Transformer kernels
- no mixed precision or loss scaling
- no generation or KV cache
- no disk-backed corpus format or asynchronous data workers

These limitations are deliberate. Profiling the reference implementation determines which backend
and kernel work should follow; Phase 19 does not optimize the correctness oracle.

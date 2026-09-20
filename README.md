# GPT From Scratch

A compact character-level GPT implemented in PyTorch, including tokenisation, causal multi-head self-attention, transformer blocks, training, checkpointing, and autoregressive text generation.

The project is intentionally small enough to read end to end while still following the core structure used by decoder-only transformer language models.

## Architecture

```text
characters
   ↓
character tokenizer
   ↓
token + position embeddings
   ↓
┌──────────────────────────────┐
│ Transformer block × N        │
│  LayerNorm                   │
│  Causal multi-head attention │
│  Residual connection         │
│  LayerNorm                   │
│  Feed-forward network        │
│  Residual connection         │
└──────────────────────────────┘
   ↓
final LayerNorm
   ↓
vocabulary projection
   ↓
next-token logits
```

## Features

- character-level tokenisation
- learned token and positional embeddings
- causal scaled dot-product attention
- multi-head self-attention
- Pre-LN transformer blocks
- GELU feed-forward layers
- residual connections
- tied input/output embeddings
- AdamW training
- gradient clipping
- train/validation loss evaluation
- checkpoint save/load
- temperature sampling
- top-k generation
- CPU, CUDA and Apple Metal support

## Project structure

```text
data/
  input.txt       example training corpus
  tokenizer.py   character tokenizer

model/
  gpt.py         GPT architecture

train.py         end-to-end training pipeline
generate.py      checkpoint loading and text generation
smoke_test.py    quick forward/backward/generation check
```

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick test

Run the lightweight smoke test:

```bash
python smoke_test.py
```

It performs a forward pass, computes loss, backpropagates gradients, and generates tokens from a tiny model.

## Train

A small example corpus is included at `data/input.txt`.

```bash
python train.py
```

The default configuration trains a small model and saves:

```text
checkpoints/gpt.pt
```

For a faster test run:

```bash
python train.py --steps 100 --model-dim 64 --layers 2 --batch-size 16
```

For a custom text corpus:

```bash
python train.py --data path/to/corpus.txt --steps 2000
```

Useful options include:

```text
--context-length
--model-dim
--heads
--layers
--dropout
--batch-size
--lr
--device
```

## Generate text

After training:

```bash
python generate.py --prompt "The " --tokens 300
```

Sampling can be adjusted with temperature and top-k:

```bash
python generate.py \
  --prompt "The " \
  --tokens 300 \
  --temperature 0.8 \
  --top-k 30
```

The tokenizer vocabulary is stored inside the checkpoint, so generation uses exactly the same character mapping as training.

## Model configuration

The default model uses:

| Setting | Value |
| --- | ---: |
| Context length | 64 |
| Model dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.1 |

These values are intentionally modest so the project can run on ordinary hardware. The architecture can be scaled through command-line arguments.

## Training data

The included corpus exists only to make the repository runnable immediately. For meaningful generation, replace it with a substantially larger plain-text dataset and train for more steps.

## Scope

This is an educational implementation of a decoder-only transformer designed to make the mechanics of GPT understandable and executable. It is not intended to reproduce the scale or performance of production language models.

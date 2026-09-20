import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small character-level GPT.")
    parser.add_argument("--data", default="data/input.txt", help="Path to a UTF-8 text corpus")
    parser.add_argument("--out", default="checkpoints/gpt.pt", help="Checkpoint output path")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(
    source: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(source) - context_length
    if max_start <= 0:
        raise ValueError("dataset split is too small for the selected context length")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i : i + context_length] for i in starts])
    y = torch.stack([source[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for name, source in (("train", train_data), ("val", val_data)):
        samples = []
        for _ in range(eval_batches):
            x, y = get_batch(source, batch_size, context_length, device)
            _, loss = model(x, y)
            samples.append(loss.item())
        losses[name] = sum(samples) / len(samples)

    model.train()
    return losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    corpus_path = Path(args.data)
    if not corpus_path.exists():
        raise FileNotFoundError(f"training corpus not found: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    min_size = 2 * (args.context_length + 1)
    if len(encoded) < min_size:
        raise ValueError(
            f"corpus is too small for context length {args.context_length}; "
            f"need at least {min_size} characters"
        )

    split = int(0.9 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]

    if len(val_data) <= args.context_length:
        raise ValueError(
            "validation split is too small; use a larger corpus or shorter context length"
        )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        model_dim=args.model_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"characters: {len(encoded):,}")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print(f"parameters: {parameter_count:,}")

    model.train()
    latest_loss = None

    for step in range(1, args.steps + 1):
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest_loss = loss.item()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.context_length,
                args.eval_batches,
                device,
            )
            print(
                f"step {step:>5} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.state_dict(),
        "step": args.steps,
        "loss": latest_loss,
    }
    torch.save(checkpoint, output_path)
    print(f"saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()

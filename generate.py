import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/gpt.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt cannot be empty")

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        context,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

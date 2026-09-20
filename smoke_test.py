import torch

from data import CharTokenizer
from model import GPT, GPTConfig


def main() -> None:
    torch.manual_seed(0)

    text = ("small models should still train and generate text\n" * 20)
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = GPT(config)

    x = tokens[:16].unsqueeze(0)
    y = tokens[1:17].unsqueeze(0)

    _, loss = model(x, y)
    loss.backward()

    generated = model.generate(
        x[:, :4],
        max_new_tokens=8,
        temperature=1.0,
        top_k=10,
    )
    decoded = tokenizer.decode(generated[0].tolist())

    assert generated.shape == (1, 12)
    assert torch.isfinite(loss)
    assert len(decoded) == 12

    print(f"smoke test passed | loss={loss.item():.4f} | sample={decoded!r}")


if __name__ == "__main__":
    main()

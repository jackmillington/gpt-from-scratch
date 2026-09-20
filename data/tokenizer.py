from typing import Dict, Iterable, List


class CharTokenizer:
    """Simple deterministic character-level tokenizer."""

    def __init__(self, chars: Iterable[str]):
        unique_chars = sorted(set(chars))
        if not unique_chars:
            raise ValueError("tokenizer vocabulary cannot be empty")

        self.chars: List[str] = unique_chars
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            shown = ", ".join(repr(ch) for ch in unknown[:8])
            raise ValueError(f"text contains characters not in the vocabulary: {shown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.itos[int(token_id)] for token_id in token_ids)

    def state_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        return cls(state["chars"])

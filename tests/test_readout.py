import numpy as np

from jspace_oversight.lenses.readout import _rank


class Tokenizer:
    def decode(self, token_ids):
        return f"token-{token_ids[0]}"


def test_readout_top_k_with_synthetic_tensor() -> None:
    logits = {10: np.array([[0.0, 4.0, 2.0]])}
    result = _rank(
        logits,
        positions=[-2],
        tokenizer=Tokenizer(),
        top_k=2,
        method="jacobian_lens",
    )
    assert result[0].layer == 10
    assert result[0].position == -2
    assert [item.token_id for item in result[0].rankings] == [1, 2]

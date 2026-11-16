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
        input_ids=np.array([[8, 9, 10]]),
    )
    assert result[0].layer == 10
    assert result[0].position == -2
    assert result[0].input_token_id == 9
    assert result[0].input_token == "token-9"
    assert [item.token_id for item in result[0].rankings] == [1, 2]


def test_readout_rejects_batched_input_ids() -> None:
    with np.testing.assert_raises_regex(ValueError, "input_ids"):
        _rank(
            {10: np.array([[0.0, 1.0]])},
            positions=[-1],
            tokenizer=Tokenizer(),
            top_k=1,
            method="logit_lens",
            input_ids=np.zeros((2, 3)),
        )

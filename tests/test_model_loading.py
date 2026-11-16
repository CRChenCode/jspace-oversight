import sys
from types import ModuleType, SimpleNamespace

from jspace_oversight.config import ModelConfig
from jspace_oversight.models.loading import format_prompt, load_causal_lm


class Factory:
    def __init__(self, value):
        self.value = value
        self.kwargs = None

    def from_pretrained(self, _model_id, **kwargs):
        self.kwargs = kwargs
        return self.value


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_commit_hash="resolved-sha")
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True


def test_model_loader_falls_back_to_cpu(monkeypatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.float16 = object()
    fake_torch.bfloat16 = object()
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

    model = FakeModel()
    tokenizer = SimpleNamespace()
    model_factory = Factory(model)
    tokenizer_factory = Factory(tokenizer)
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = model_factory
    fake_transformers.AutoTokenizer = tokenizer_factory
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = load_causal_lm(
        ModelConfig(model_id="org/model", device="cuda", dtype="bfloat16")
    )
    assert loaded.model.device == "cpu"
    assert loaded.model.evaluated is True
    assert model_factory.kwargs["dtype"] is fake_torch.float32
    assert loaded.resolved_revision == "resolved-sha"


def test_format_prompt_uses_chat_template_when_available() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "hello"
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "formatted"

    assert format_prompt(Tokenizer(), "hello", use_chat_template=True) == "formatted"
    assert format_prompt(Tokenizer(), "hello", use_chat_template=False) == "hello"

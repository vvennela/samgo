"""Decoder backend on Fireworks' serverless sampler: a promoted LoRA model or the base model, with no
deployment and no training slot. Billed per token (prefill, cached prefill and sample meters), so it suits
spot checks; a full 250-prompt eval is cheaper on a deployment through decode.Completions.

Run with the cookbook venv (~/.venvs/fwcookbook) and the cookbook root on PYTHONPATH.
"""

import os

from .decode import END, split_lines

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
HF_MODEL = "meta-models/Muse-Glimmer-30B"


class Sampler:
    """Same `lines()` interface as decode.Completions: prompt text in, whole lines out."""

    def __init__(self, model: str | None = None):
        """`model`: a promoted LoRA (accounts/<a>/models/<id>); None samples the base model."""
        from fireworks.training.sdk import FiretitanServiceClient
        from training.utils.serverless import serverless_base_url
        from training.utils.tokenizers import load_tokenizer

        self.tok = load_tokenizer(HF_MODEL)
        self.service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                              base_url=serverless_base_url("https://api.fireworks.ai"))
        kwargs = {"model_path": model} if model else {"base_model": BASE_MODEL}
        self.client = self.service.create_sampling_client(tokenizer=self.tok, **kwargs)
        self.end_id = self.tok.convert_tokens_to_ids(END)

    def lines(self, prompt: str, temperature: float, top_k: int, max_tokens: int) -> tuple[list[str], bool]:
        from fireworks.training import sdk

        ids = self.tok.encode(prompt, add_special_tokens=False)
        params = sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_k=top_k, stop=[END])
        out = self.client.sample(prompt=sdk.ModelInput.from_ints(ids), num_samples=1, sampling_params=params).result()
        tokens = list(out.sequences[0].tokens)
        ended = self.end_id in tokens or len(tokens) < max_tokens  # stopped by the end of answer, not the limit
        if self.end_id in tokens:
            tokens = tokens[: tokens.index(self.end_id)]
        return split_lines(self.tok.decode(tokens, skip_special_tokens=True), ended)

    def close(self) -> None:
        for obj in (self.client, self.service):
            close = getattr(obj, "close", None)
            if close:
                close()

"""Decoder backend on Fireworks' serverless sampler: a promoted LoRA model or the base model, with no
deployment and no training slot. Billed per token (prefill, cached prefill and sample meters), so it suits
spot checks; a full 250-prompt eval is cheaper on a deployment through decode.Completions.

Run with the cookbook venv (~/.venvs/fwcookbook) and the cookbook root on PYTHONPATH.
"""

import os

from .decode import END, split_lines

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
# Serverless sampling rates for Glimmer per 1M tokens (fireworks.ai/pricing, fetched Oct 29). The budget guard
# counts every prompt token at the uncached rate, so the real bill is at or below its estimate.
PREFILL_PER_M, SAMPLE_PER_M = 1.96, 4.88
HF_MODEL = "meta-models/Muse-Glimmer-30B"


class Sampler:
    """Same `lines()` interface as decode.Completions: prompt text in, whole lines out."""

    def __init__(self, model: str | None = None, budget_usd: float | None = None):
        """`model`: a promoted LoRA (accounts/<a>/models/<id>); None samples the base model.
        `budget_usd`: stop calling the API once the worst-case cost estimate reaches it (the build then ends)."""
        self.budget, self.prefill_tokens, self.sample_tokens = budget_usd, 0, 0
        from fireworks.training.sdk import FiretitanServiceClient
        from training.utils.serverless import serverless_base_url
        from training.utils.tokenizers import load_tokenizer

        self.tok = load_tokenizer(HF_MODEL)
        self.service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                              base_url=serverless_base_url("https://api.fireworks.ai"))
        kwargs = {"model_path": model} if model else {"base_model": BASE_MODEL}
        self.client = self.service.create_sampling_client(tokenizer=self.tok, **kwargs)
        self.end_id = self.tok.convert_tokens_to_ids(END)

    def cost(self) -> float:
        return self.prefill_tokens / 1e6 * PREFILL_PER_M + self.sample_tokens / 1e6 * SAMPLE_PER_M

    def over_budget(self) -> bool:
        return self.budget is not None and self.cost() >= self.budget

    def lines(self, prompt: str, temperature: float, top_k: int, max_tokens: int) -> tuple[list[str], bool]:
        from fireworks.training import sdk

        if self.over_budget():
            return [], True
        ids = self.tok.encode(prompt, add_special_tokens=False)
        self.prefill_tokens += len(ids)
        params = sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_k=top_k, stop=[END])
        out = self.client.sample(prompt=sdk.ModelInput.from_ints(ids), num_samples=1, sampling_params=params).result()
        tokens = list(out.sequences[0].tokens)
        self.sample_tokens += len(tokens)
        ended = self.end_id in tokens or len(tokens) < max_tokens  # stopped by the end of answer, not the limit
        if self.end_id in tokens:
            tokens = tokens[: tokens.index(self.end_id)]
        return split_lines(self.tok.decode(tokens, skip_special_tokens=True), ended)

    def candidates(self, prompt: str, temperature: float, top_k: int, max_tokens: int, k: int) -> list:
        """k sampled next lines; None for a sample that ended the answer."""
        from fireworks.training import sdk

        if self.over_budget():
            return [None] * k
        ids = self.tok.encode(prompt, add_special_tokens=False)
        self.prefill_tokens += len(ids)
        params = sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_k=top_k, stop=["\n", END])
        out = self.client.sample(prompt=sdk.ModelInput.from_ints(ids), num_samples=k, sampling_params=params).result()
        texts = []
        for seq in out.sequences:
            tokens = list(seq.tokens)
            self.sample_tokens += len(tokens)
            text = self.tok.decode([t for t in tokens if t != self.end_id], skip_special_tokens=True).strip()
            texts.append(text or None)
        return texts

    def close(self) -> None:
        for obj in (self.client, self.service):
            close = getattr(obj, "close", None)
            if close:
                close()

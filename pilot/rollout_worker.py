"""vLLM in its own process, pinned to its own GPUs, serving rollout requests for the GRPO loop.

The trainer holds the policy on the other GPUs; each round it sends prompts plus the latest
adapter path, and this process samples with that adapter (hot-loaded by a new LoRA id, no restart).
Kept free of torch imports at module level so CUDA_VISIBLE_DEVICES is set before anything touches CUDA.
"""

import os


def serve(requests, responses, cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    common = dict(model=cfg["model"], tensor_parallel_size=len(cfg["gpus"].split(",")), max_model_len=12288,
                  generation_config="vllm", gpu_memory_utilization=0.90, enable_lora=True, max_lora_rank=16,
                  max_loras=1, enable_prefix_caching=True)
    try:
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **common)
    except (ValueError, KeyError, TypeError):
        llm = LLM(limit_mm_per_prompt={"image": 0}, **common)
    responses.put({"ready": True})
    while (req := requests.get()) is not None:
        # Plain temperature-1 sampling: the policy's own distribution, so the update is on-policy.
        params = SamplingParams(n=req["n"], temperature=1.0, max_tokens=req["max_tokens"],
                                stop_token_ids=req["stop"], ignore_eos=True, skip_special_tokens=False)
        lora = LoRARequest(f"r{req['lora_id']}", req["lora_id"], req["lora_path"]) if req["lora_path"] else None
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in req["prompts"]], params, lora_request=lora)
        responses.put({"outputs": [[(list(s.token_ids), s.finish_reason) for s in o.outputs] for o in outs]})

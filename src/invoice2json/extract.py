"""Stage 3: field extraction. One public function: extract(layout_text, cfg) -> dict.

Runs a local instruction-tuned LLM via Ollama with the JSON schema supplied
as structured output, so output is schema-constrained JSON.
Temperature 0 for R9 (same input, same output).
"""
from __future__ import annotations

import json
from typing import Optional

from ollama import chat

from .config import Config


def _chunk_line_items(layout_text: str, max_chars: int) -> list[str]:
    """Split layout text into chunks if it exceeds the context budget.

    Header fields are taken from the first chunk only (per design doc);
    line items are merged across all chunks by the caller.
    """
    if len(layout_text) <= max_chars:
        return [layout_text]

    lines = layout_text.split("\n")
    chunks, current = [], []
    current_len = 0

    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0

        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def extract(
    layout_text: str,
    cfg: Config = Config(),
    model_path: Optional[str] = None,
) -> dict:

    model = model_path or cfg.llm_model
    prompt_template = cfg.prompt_path.read_text()

    # Load JSON schema for Ollama structured output
    schema = json.loads(cfg.schema_path.read_text())

    # Rough budget: leave headroom for prompt scaffolding + output tokens
    char_budget = (
        cfg.llm_context_tokens
        - cfg.llm_max_output_tokens
        - 300
    ) * 3

    chunks = _chunk_line_items(layout_text, char_budget)

    merged_result: Optional[dict] = None
    all_line_items: list[dict] = []

    for chunk in chunks:
        prompt = prompt_template.format(layout_text=chunk)

        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=schema,
            options={
                "temperature": cfg.llm_temperature,
                "seed": cfg.llm_seed,
                "num_ctx": cfg.llm_context_tokens,
            },
        )

        text = response["message"]["content"]
        parsed = json.loads(text)

        if merged_result is None:
            merged_result = parsed

        all_line_items.extend(parsed.get("line_items", []))

    assert merged_result is not None
    merged_result["line_items"] = all_line_items

    return merged_result
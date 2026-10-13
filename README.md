# invoice2json

Local, offline invoice photo -> validated JSON. Implements the staged
pipeline from the design doc: preprocess -> OCR -> schema-constrained LLM
extraction -> arithmetic/format validation. No GPU, no network calls at
runtime.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.lock
pip install -e .
```

Download GGUF models separately (not included), record them in `MODELS.md`,
and point `config.py`'s `llm_model_path` / `fallback_llm_model_path` at the
local files.

## Run on one invoice

```bash
invoice2json convert path/to/photo.jpg -o out.json
```

## Watch a folder

```bash
invoice2json watch --watch ./inbox
```
Results land in `./done` (ok / ok_with_warnings) or `./review` (needs_review).
Every run also writes a `debug/<filename>/` bundle: the preprocessed image,
OCR layout text, raw model output, and the validation report — this is what
makes a bad result diagnosable.

## Evaluate against the golden set

Build `eval/golden/<name>.jpg` + `eval/golden/<name>.expected.json` pairs
first (design doc section 7), then:

```bash
python eval/run_eval.py --golden eval/golden --ocr-engine rapidocr
```

Swap `--ocr-engine tesseract` or point `--llm-model-path` at a different
GGUF to run the bake-off comparisons. Results get written to `eval/report.md`.

## Status vs. the design doc

Built: preprocess, OCR (RapidOCR + Tesseract), schema-constrained extraction
via llama.cpp grammar, full validation (hard + soft checks), orchestrator,
CLI, watch mode, eval harness.

Not yet wired up (explicitly deferred per design doc section 4.5): the VLM
fallback step (`fallback.py::_try_vlm`) — it's the slowest, last-resort path,
and whether it's worth building depends on how much the (a)/(b) fallback
steps already close the gap once real numbers are in from the golden set.

Nothing in this repo has been run against real invoices yet — that's the
next step, via `eval/run_eval.py` once a golden set and GGUF models are in
place.

"""All tunables live here. Nothing is hard-coded in the other modules."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- locale ---
    date_order: str = "DMY"          # DMY or MDY, used when the date is ambiguous
    decimal_separator: str = "."     # "." or ","
    thousands_separator: str = ","   # "," or "."
    default_currency: str = "INR"

    # --- quality gate (preprocess.py) ---
    min_sharpness: float = 40.0      # Laplacian variance below this -> rejected_low_quality
    min_page_area_ratio: float = 0.25  # detected page must fill at least this fraction of the frame
    target_long_edge_px: int = 2000

    # --- OCR (ocr.py) ---
    ocr_engine: str = "rapidocr"     # "rapidocr" or "tesseract"

   
    # --- extraction (extract.py) ---
    llm_backend: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_context_tokens: int = 8192
    llm_max_output_tokens: int = 2048
    llm_temperature: float = 0.0
    llm_seed: int = 0
     # --- fallback (fallback.py) ---
    fallback_enabled: bool = True
    fallback_llm_model: str = "qwen2.5:7b"
    fallback_vlm_model_path: str = "models/qwen2.5-vl-7b-q4_k_m.gguf"
    # --- validation (validate.py) ---
    arithmetic_tolerance: float = 0.01
    known_vendors: list[str] = field(default_factory=list)

    # --- paths ---
    schema_path: Path = Path(__file__).resolve().parents[2] / "schema" / "invoice.schema.json"
    prompt_path: Path = Path(__file__).resolve().parents[2] / "prompts" / "extract_v1.txt"
    debug_dir: Path = Path("debug")
    done_dir: Path = Path("done")
    review_dir: Path = Path("review")


DEFAULT_CONFIG = Config()

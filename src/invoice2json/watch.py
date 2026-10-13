from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .pipeline import process_invoice, write_result

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".pdf"}


class _InvoiceHandler(FileSystemEventHandler):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in VALID_SUFFIXES:
            return
        time.sleep(0.5)  # let the file finish writing before we read it
        result = process_invoice(path, self.cfg)
        out_path = write_result(result, self.cfg)
        print(f"[watch] {path.name} -> {out_path} ({result['status']})")


def watch_folder(inbox: Path, cfg: Config = Config()) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    handler = _InvoiceHandler(cfg)
    observer = Observer()
    observer.schedule(handler, str(inbox), recursive=False)
    observer.start()
    print(f"Watching {inbox} — drop invoices here. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import Config
from .pipeline import process_invoice, write_result

app = typer.Typer(add_completion=False)


@app.command()
def convert(
    photo: Path = typer.Argument(..., help="Path to a JPG/PNG/HEIC/PDF invoice"),
    output: Path = typer.Option(None, "-o", "--output", help="Write JSON here instead of done/review"),
):
    """Convert a single invoice photo to JSON."""
    cfg = Config()
    result = process_invoice(photo, cfg)
    if output:
        output.write_text(json.dumps(result, indent=2, default=str))
        typer.echo(f"Wrote {output} (status: {result['status']})")
    else:
        path = write_result(result, cfg)
        typer.echo(f"Wrote {path} (status: {result['status']})")


@app.command()
def watch(
    inbox: Path = typer.Option(Path("inbox"), "--watch", help="Folder to watch for new invoices"),
):
    """Watch a folder and process new files as they arrive."""
    from .watch import watch_folder
    watch_folder(inbox, Config())


if __name__ == "__main__":
    app()

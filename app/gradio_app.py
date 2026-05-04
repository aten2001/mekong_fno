"""Gradio entrypoint wrapper for the existing HF Space application."""

from __future__ import annotations

from typing import Any


def build_gradio_app():
    """Build the current Gradio UI without moving callback internals yet."""
    from app.app import build_app

    return build_app()


create_gradio_app = build_gradio_app


def launch_gradio_app(
    *,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    **launch_kwargs: Any,
):
    """Launch the existing Gradio app with the same warmup/sync behavior."""
    from app.app import (
        BACKFILL_P,
        DATASET_REPO,
        HF_READ_TOKEN,
        LAYOUT,
        _load_service,
        gr,
        sync_backfill_from_dataset,
        sync_status_from_dataset,
    )

    print(f"[runtime] root={LAYOUT.root}")
    print(f"[runtime] cache={LAYOUT.cache}")
    print(f"[runtime] artifacts={LAYOUT.artifacts}")

    if DATASET_REPO:
        try:
            sync_status_from_dataset(DATASET_REPO, LAYOUT.artifacts, token=HF_READ_TOKEN)
            sync_backfill_from_dataset(DATASET_REPO, BACKFILL_P, token=HF_READ_TOKEN)
        except Exception as exc:
            print("[sync][warn] startup sync failed:", repr(exc))

    _load_service()

    app = build_gradio_app()
    return app.launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Soft(),
        **launch_kwargs,
    )


if __name__ == "__main__":
    launch_gradio_app()

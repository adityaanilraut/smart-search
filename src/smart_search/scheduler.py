"""Rebuild the configured corpus at 02:00 UTC each day."""

from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import time

from .index import Encoder, build


def main():
    encoder = None
    while True:
        now = datetime.now(UTC)
        run_at = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if run_at <= now:
            run_at += timedelta(days=1)
        time.sleep((run_at - now).total_seconds())
        try:
            encoder = encoder or Encoder(4)
            print(build(Path(os.environ["SOURCE_FILE"]), Path(os.environ["INDEX_DIR"]), encoder), flush=True)
        except Exception as exc:
            print(f"Reindex failed; keeping current generation: {exc}", flush=True)


if __name__ == "__main__":
    main()

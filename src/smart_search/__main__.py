import argparse
import json
import os
from pathlib import Path

from .index import Encoder, build


def main():
    parser = argparse.ArgumentParser(description="Build and activate a Smart Search index")
    parser.add_argument("source", type=Path)
    parser.add_argument("--index", type=Path, default=Path(os.getenv("INDEX_DIR", "index")))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--apple-gpu", action="store_true", help="Use the optional MLX MiniLM encoder")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and thread count must be positive")
    if args.apple_gpu:
        from .apple import AppleEncoder

        encoder = AppleEncoder()
    else:
        encoder = Encoder(args.threads)
    print(json.dumps(build(args.source, args.index, encoder, args.batch_size), indent=2))


if __name__ == "__main__":
    main()

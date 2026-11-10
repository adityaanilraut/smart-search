#!/bin/sh
set -eu
mkdir -p data
archive=data/trec-covid.zip
if [ ! -f "$archive" ]; then
  curl --fail --location --retry 3 \
    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip \
    --output "$archive.part"
  mv "$archive.part" "$archive"
fi
uv run --locked python -c 'import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); assert hashlib.md5(p.read_bytes()).hexdigest()=="ce62140cb23feb9becf6270d0d1fe6d1", "BEIR archive checksum mismatch"' "$archive"
unzip -oq "$archive" -d data

#!/usr/bin/env bash
# Stream each LinuxArena shard once (public, CC-BY-4.0, ~19 GB total, nothing kept), verify its SHA-256 against the
# entry for that file in data/SHA256SUMS.raw, extract (a) monitor-view text -> data/text/ and (b) structured actions +
# record counts -> data/struct/, delete the shard. Extracts are written atomically, so an interrupted run is resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=ed82b89a260f8a9cb2fabe499e2085c3d8b057db   # dataset revision pinned 2026-11-20 (lastModified 2026-11-17T18:41Z)
BASE=https://huggingface.co/datasets/anonymouslinuxarena/linuxarena-public/resolve/$REV/data
# Optional overrides (used by `make smoke`): SHARDS="<env> ..." restricts the shards; OUTDIR replaces data/ for the extracts.
ALL="auto_workflow clinical_trial file_indexing media_processing model_registry shared_dotfiles slack_clone tiktok user_tracking web_scraping"
SHARDS=${SHARDS:-$ALL}; O=${OUTDIR:-data}
mkdir -p data/raw $O/text $O/struct
for s in $SHARDS; do
  [ -s $O/text/$s.jsonl.gz ] && [ -s $O/struct/$s.jsonl.gz ] && [ -s $O/struct/$s.counts.json ] && continue
  for i in 1 2 3 4; do curl -sS -L -m 1800 -o data/raw/$s.jsonl "$BASE/$s.jsonl" && break || sleep $((i*10)); done
  grep "  data/raw/$s.jsonl\$" data/SHA256SUMS.raw | sha256sum -c --quiet - || { echo "$s: SHA-256 mismatch" >&2; exit 1; }
  echo "$s sha256 OK"
  python3 -I src/extract_text.py data/raw/$s.jsonl $O/text/$s.jsonl.gz
  python3 -I src/extract_struct.py data/raw/$s.jsonl $O/struct/$s.jsonl.gz
  rm -f data/raw/$s.jsonl
done
[ "$SHARDS" = "$ALL" ] && [ "$O" = data ] && touch data/.complete
echo DONE

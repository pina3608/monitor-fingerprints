# Data

No trajectory data is shipped. Everything is regenerated from the public LinuxArena mirror
(Hugging Face `anonymouslinuxarena/linuxarena-public` at pinned revision `ed82b89a260f8a9cb2fabe499e2085c3d8b057db`, CC-BY-4.0) by `make data` (`scripts/fetch_data.sh`).

| Path | Produced by | Content | Size |
|---|---|---|---|
| `raw/<env>.jsonl` | `scripts/fetch_data.sh` | the 10 environment shards; streamed one at a time and deleted after extraction | ~19 GB total, not kept |
| `SHA256SUMS.raw` | (shipped) | SHA-256 of each shard as downloaded on 2026-11-17, plus `sample.jsonl`; the fetch script aborts on a mismatch | small |
| `text/<env>.jsonl.gz` | `src/extract_text.py` | per agent run: trajectory and run IDs, model, label, env, main/side task, main-task success, monitor-view text (`raw`, first 100k chars) and tool-calls-only text (`calls`) | 63 MB |
| `struct/<env>.jsonl.gz` | `src/extract_struct.py` | per agent run: trajectory and run IDs, main/side-task success, per-action tool name, argument keys/order, value reprs, command, text-editor sub-command; system-prompt template hash | 18 MB |
| `struct/<env>.counts.json` | `src/extract_struct.py` | record counts per shard by type (monitor re-scorings, cohort runs, other agent runs, duplicate trajectory IDs); summed by `src/cohort_audit.py` | small |
| `raw/sample.jsonl` | manual download (only for `make test-fidelity`) | the dataset's 932-trajectory sample file | 672 MB |

`sample.jsonl`:
```bash
curl -L -o data/raw/sample.jsonl https://huggingface.co/datasets/anonymouslinuxarena/linuxarena-public/resolve/main/data/sample.jsonl
```

Only agent runs named `"<Model> honest"` / `"<Model> attack"` are extracted; the dataset's monitor re-scoring records
(`"<Monitor> monitor on <Model> ..."`, 12,248 of 22,215) re-score existing trajectories and are ignored.
`manifest/inputs.jsonl` lists every extracted trajectory with the SHA-256 of its monitor view, so a regenerated
`data/text/` can be checked line by line (`make manifest`, then compare `manifest/MANIFEST_SHA256`).

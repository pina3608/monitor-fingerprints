# Reproduce the study from public data. No API keys, no paid calls. Run from the repository root.
# Safe under `make -j`: every analysis depends on the data stamp, which `data` creates only after all shards are extracted.
PY = python3 -I
DATA = data/.complete

.PHONY: reproduce smoke test test-fidelity manifest cohort identification retention pairwise harness robustness baselines figure
reproduce: test manifest cohort identification retention pairwise harness robustness baselines figure

$(DATA):         ## stream 10 LinuxArena shards (~19 GB, nothing kept; ~15 min), verify SHA-256, extract data/text + data/struct
	./scripts/fetch_data.sh
data: $(DATA)

SHARD ?= user_tracking
SMOKE = data/smoke
smoke:           ## one-shard end-to-end check (~1 GB download, a few minutes): fetch+verify SHARD, extract, run cohort + baselines on it
	SHARDS=$(SHARD) OUTDIR=$(SMOKE) ./scripts/fetch_data.sh
	$(PY) src/cohort_audit.py $(SMOKE)/text $(SMOKE)/struct $(SMOKE)/cohort_audit
	$(PY) src/baselines.py $(SMOKE)/text $(SMOKE)/struct $(SMOKE)/baselines

test:            ## transform unit tests (seconds)
	$(PY) tests/test_transforms.py
	$(PY) tests/test_canonical.py

test-fidelity:   ## formatter byte-equality vs upstream; needs CT=<control-tower clone @699d5ca> and data/raw/sample.jsonl
	$(PY) tests/test_formatter_fidelity.py $(CT) data/raw/sample.jsonl

manifest: $(DATA)        ## per-trajectory input manifest; compare manifest/MANIFEST_SHA256
	$(PY) src/make_manifest.py data/text manifest/inputs.jsonl

cohort: $(DATA)          ## record counts, cohort/provenance table, repeated views, planning-call counts (seconds)
	$(PY) src/cohort_audit.py data/text data/struct results/cohort_audit

identification: $(DATA)  ## identification + jamming transforms (~1 h)
	$(PY) src/identification.py data/text results/identification

pairwise: $(DATA)        ## task-matched own-vs-other pairs
	$(PY) src/pairwise.py data/text results/pairwise

harness: $(DATA)         ## harness/provenance audit (+ template-held-constant check)
	$(PY) src/harness_audit.py data/struct results/harness_audit
	$(PY) src/within_template.py data/struct results/within_template.txt

robustness: $(DATA)      ## one-vs-rest AUROC, leave-one-environment-out, artefact-stripped transfer, same-vendor, recursive canonicalisation
	$(PY) src/robustness.py data/text data/struct results/robustness
	$(PY) src/robustness2.py data/text data/struct results/robustness2
	$(PY) src/canonical_calls.py data/text data/struct results/canonical_calls

retention: $(DATA)       ## matched-task O4 (pooled + foldwise) and task-disjoint control; pooled vs foldwise O4 for all cohorts
	$(PY) src/retention_matched.py data/text results/retention_matched
	$(PY) src/o4_foldwise.py data/text results/o4_foldwise

baselines: $(DATA)       ## verb signal vs tool-name, length, outcome, template baselines; prefixes; J0/J1 prefixes
	$(PY) src/baselines.py data/text data/struct results/baselines

figure:          ## headline figure from committed results (no data needed)
	$(PY) src/make_figure.py results report/figures/headline.png

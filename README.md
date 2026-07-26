# benchmark-IxIDN-ORDON

Merging the `IxIDN_ORDON` disease-pair benchmark back onto its two source datasets, and
evaluating how well the DDA score separates them.

## Layout

```
raw_benchmarks/
  IxIDN_ORDON_benchmark.csv   10,336 disease pairs (ID, d1_disease_label, d2_disease_label)
  IxIDN_df.csv                 5,336 pairs with clinical-trial evidence (drug_labels, trial_count_total)
  ORDON_df.csv                 5,000 ontology-derived pairs
claude_science/
  dda_analysis.py                      all the logic — merge, metrics, figures
  merge_benchmark_sources_v1.ipynb     driver for DDA v1
  merge_benchmark_sources_v2.ipynb     driver for DDA v2
  compare_dda_v1_v2.ipynb              v1 vs v2 comparison
  IxIDN_ORDON_benchmark_DDA_v{1,2}.csv DDA scores
  IxIDN_ORDON_benchmark_v{1,2}_merged.csv  merged outputs
```

The notebooks are thin drivers — every function lives in `dda_analysis.py`, so the three
notebooks share one implementation. Figures render inline and are stored inside the
notebooks; nothing is written to disk as an image.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name benchmark-ixidn-ordon --display-name "Python (benchmark-IxIDN-ORDON)"
```

Then open a notebook in `claude_science/` and select that kernel. To re-run headless:

```sh
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  claude_science/merge_benchmark_sources_v2.ipynb
```

## Merge

Join key is the disease pair `(d1_disease_label, d2_disease_label)`.

**Pair orientation is not preserved between the benchmark and `IxIDN_df.csv`.** A naive merge on
`(d1, d2)` finds only 2,767 of the 5,336 IxIDN rows; joining on the order-insensitive canonical key
`tuple(sorted([d1, d2]))` finds all of them. ORDON is unaffected either way.

Output columns: `ID, d1_disease_label, d2_disease_label, DDA, exist_IxIDN, exist_ORDON, IxIDN_ID, ORDON_ID`,
where `IxIDN_ID` / `ORDON_ID` are 0-based row numbers in the source CSVs (blank when absent).

| bucket | rows | pct |
|---|---:|---:|
| IxIDN only | 5,334 | 51.61% |
| ORDON only | 4,998 | 48.36% |
| both | 4 | 0.04% |
| unmatched | 0 | 0.00% |
| total | 10,336 | 100% |

Every source row is accounted for: 5,336/5,336 IxIDN and 5,000/5,000 ORDON pairs appear in the
benchmark exactly once.

## DDA separation

Treating **IxIDN as positive** and **ORDON as negative**, with the 4 both-source pairs excluded as
unlabelable (v2 figures; v1 is indistinguishable — see below):

| group | n | mean | median | IQR |
|---|---:|---:|---:|---:|
| IxIDN (positive) | 5,334 | 0.3610 | 0.3356 | 0.2512 – 0.4488 |
| ORDON (negative) | 4,998 | 0.2077 | 0.2018 | 0.1611 – 0.2476 |

- **AUC 0.827**; Youden-optimal cut at `DDA >= 0.2745` (TPR 0.683, FPR 0.154).
- **Wasserstein W₁ = 0.1533** DDA units (1.27 pooled SD), IxIDN shifted higher.
- W₁ equals the mean difference to machine precision, which holds only when the two CDFs never
  cross — verified directly over the score range. IxIDN **stochastically dominates** ORDON
  everywhere, so the separation is monotone rather than confined to one region of the score.

**Caveat.** The two classes here are *which source file a pair came from*, not validated
positive/negative labels. IxIDN pairs carry clinical-trial evidence while ORDON pairs are
ontology-derived, so these numbers partly measure how well DDA recovers that provenance
difference. Read them as a separation measure, not as association validation.

## v1 vs v2

Only **11 of 10,336** scores differ (99.894% identical), all revised **downward**:

| metric | v1 | v2 | Δ |
|---|---:|---:|---:|
| AUC | 0.827330 | 0.827169 | −0.000161 |
| W₁ | 0.153284 | 0.153255 | −0.000029 |
| best threshold | 0.2745 | 0.2745 | 0 |
| TPR / FPR at best | 0.683 / 0.154 | 0.683 / 0.154 | 0 / 0 |

Two things stand out. All 11 changed rows are **IxIDN positives**, and all 11 share the same
`d1_disease_label` — *Polyneuropathy associated with IgM monoclonal gammopathy* — so the revision
is confined to one disease's pairs rather than spread across the benchmark. And because those rows
were already low-scoring positives being pushed lower, v2 is fractionally *worse* at the
IxIDN-vs-ORDON separation task, though the change is far too small to matter (ΔAUC −1.6e−4).

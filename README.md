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
  merge_benchmark_sources.ipynb        the pipeline
  IxIDN_ORDON_benchmark_DDA_v1.csv     DDA scores, v1
  IxIDN_ORDON_benchmark_DDA_v2.csv     DDA scores, v2 (11 rows differ from v1, max delta 0.0519)
  IxIDN_ORDON_benchmark_v2_merged.csv  merged output for v2
  dda_*.png                            rendered figures
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name benchmark-ixidn-ordon --display-name "Python (benchmark-IxIDN-ORDON)"
```

Then open `claude_science/merge_benchmark_sources.ipynb` and select that kernel.

The notebook is parameterized — the DDA input and output filename default to v2 and can be
overridden without editing:

```sh
cd claude_science
DDA_FILE=IxIDN_ORDON_benchmark_DDA_v1.csv \
MERGED_FILE=IxIDN_ORDON_benchmark_v1_merged.csv \
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace merge_benchmark_sources.ipynb
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

## DDA separation (v2)

Treating **IxIDN as positive** and **ORDON as negative**, with the 4 both-source pairs excluded as
unlabelable:

| group | n | mean | median | IQR |
|---|---:|---:|---:|---|
| IxIDN (positive) | 5,334 | 0.3610 | 0.3356 | 0.2512 – 0.4488 |
| ORDON (negative) | 4,998 | 0.2077 | 0.2018 | 0.1611 – 0.2476 |

- **AUC 0.827**; Youden-optimal cut at `DDA >= 0.275` (TPR 0.683, FPR 0.154).
- **Wasserstein W₁ = 0.1533** DDA units (1.27 pooled SD), IxIDN shifted higher.
- W₁ equals the mean difference to machine precision, which holds only when the two CDFs never
  cross — verified directly. IxIDN **stochastically dominates** ORDON across the whole DDA range,
  so the separation is monotone rather than confined to one region of the score.

**Caveat.** The two classes here are *which source file a pair came from*, not validated
positive/negative labels. IxIDN pairs carry clinical-trial evidence while ORDON pairs are
ontology-derived, so these numbers partly measure how well DDA recovers that provenance
difference. Read them as a separation measure, not as association validation.

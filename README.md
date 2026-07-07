# ChronoGLiNER: Label-aware and temporally adapted NER for historical documents

**ChronoGLiNER** is an experimental label-aware named entity recognition system for historical documents.

It evaluates GLiNER-style open-label NER on HIPE-style historical newspaper data. Instead of training only fixed BIO tags, ChronoGLiNER uses semantic label prompts such as `person`, `historical place`, `administrative institution`, or `date expression`.

The first goal is to compare several zero-shot and prompt-based GLiNER variants against a historical BERT NER baseline.


| Model | Label variant | Threshold | Precision | Recall | F1 | loc F1 | org F1 | pers F1 | prod F1 | time F1 | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ChronoGLiNER-ZS` | `simple` | 0.25 | 0.2326 | **0.5420** | 0.3255 | 0.55 | 0.15 | 0.38 | 0.00 | 0.05 | 0.23 | 0.43 |
| `ChronoGLiNER-ZS` | `simple` | 0.30 | 0.2467 | 0.5365 | 0.3380 | 0.56 | 0.16 | 0.39 | 0.00 | 0.05 | 0.23 | 0.44 |
| `ChronoGLiNER-ZS` | `simple` | 0.35 | 0.2597 | 0.5280 | 0.3482 | 0.57 | 0.16 | 0.41 | 0.00 | 0.04 | 0.24 | 0.44 |
| `ChronoGLiNER-ZS` | `simple` | 0.40 | 0.2735 | 0.5128 | 0.3567 | 0.57 | 0.18 | 0.42 | 0.00 | 0.04 | 0.24 | 0.45 |
| `ChronoGLiNER-ZS` | `simple` | 0.50 | 0.3082 | 0.4921 | 0.3790 | 0.58 | 0.20 | **0.44** | 0.00 | 0.04 | 0.25 | **0.46** |
| `ChronoGLiNER-D` | `historical` | 0.25 | **0.3445** | 0.4391 | **0.3861** | 0.53 | 0.19 | 0.43 | 0.00 | **0.10** | **0.25** | 0.44 |



| Variant | Name | Description | Current result |
|---|---|---|---|
| Zero-shot | `ChronoGLiNER-ZS` | GLiNER used directly with simple labels | Best so far: P=0.2467, R=0.5365, F1=0.3380 at threshold 0.30 |
| Descriptive labels | `ChronoGLiNER-D` | GLiNER with richer historical label descriptions | Running / TBD |
| Fine-tuned | `ChronoGLiNER-FT` | GLiNER fine-tuned on HIPE `NE-COARSE-LIT` | Planned |
| Teacher model | `ChronoGLiNER-Teacher` | Fine-tuned GLiNER used to pseudo-label more historical data | Planned |
| Student model | `ChronoGLiNER-BERT` | Historical BERT trained on gold + GLiNER pseudo-labels | Planned |
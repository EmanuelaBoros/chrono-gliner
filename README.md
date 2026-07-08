# ChronoGLiNER: Label-aware and temporally adapted NER for historical documents

**ChronoGLiNER** is an experimental label-aware named entity recognition system for historical documents.

It evaluates GLiNER-style open-label NER on HIPE-style historical newspaper data. Instead of training only fixed BIO tags, ChronoGLiNER uses semantic label prompts such as `person`, `historical place`, `administrative institution`, or `date expression`.

The first goal is to compare several zero-shot and prompt-based GLiNER variants against a historical BERT NER baseline.


## ChronoGLiNER Results on HIPE-2020 French Test Set

| Rank | Model | Label variant | Threshold | Precision | Recall | F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `ChronoGLiNER-D` | `historical_fr` | 0.35 | **0.4574** | 0.3855 | **0.4184** |
| 2 | `ChronoGLiNER-D` | `historical_fr` | 0.30 | 0.4193 | 0.4050 | 0.4120 |
| 3 | `ChronoGLiNER-D` | `historical_fr` | 0.40 | 0.4833 | 0.3532 | 0.4082 |
| 4 | `ChronoGLiNER-D` | `historical` | 0.40 | 0.4541 | 0.3672 | 0.4061 |
| 5 | `ChronoGLiNER-D` | `historical_fr` | 0.25 | 0.3829 | **0.4294** | 0.4048 |
| 6 | `ChronoGLiNER-D` | `historical` | 0.35 | 0.4172 | 0.3910 | 0.4036 |
| 7 | `ChronoGLiNER-D` | `historical` | 0.30 | 0.3823 | 0.4172 | 0.3990 |
| 8 | `ChronoGLiNER-D` | `historical` | 0.25 | 0.3445 | 0.4391 | 0.3861 |
| 9 | `ChronoGLiNER-D` | `historical` | 0.50 | 0.5039 | 0.3118 | 0.3853 |
| 10 | `ChronoGLiNER-ZS` | `simple` | 0.50 | 0.3082 | 0.4921 | 0.3790 |
| 11 | `ChronoGLiNER-D` | `historical_fr` | 0.50 | **0.5231** | 0.2893 | 0.3725 |
| 12 | `ChronoGLiNER-ZS` | `simple` | 0.40 | 0.2735 | 0.5128 | 0.3567 |
| 13 | `ChronoGLiNER-ZS` | `simple` | 0.35 | 0.2597 | 0.5280 | 0.3482 |
| 14 | `ChronoGLiNER-ZS` | `simple` | 0.30 | 0.2467 | 0.5365 | 0.3380 |
| 15 | `ChronoGLiNER-ZS` | `simple` | 0.25 | 0.2326 | **0.5420** | 0.3255 |


| Model / Variant | Name | Description | Threshold | Precision | Recall | F1 | loc F1 | org F1 | pers F1 | prod F1 | time F1 | Macro F1 | Weighted F1 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Supervised BERT baseline | [`historical-ner-baseline`](https://huggingface.co/emanuelaboros/historical-ner-baseline) | Historical BERT token classifier fine-tuned on HIPE `NE-COARSE-LIT` | — | **0.7797** | **0.7543** | **0.7668** | **0.87** | **0.61** | 0.68 | **0.66** | **0.40** | **0.65** | **0.76** | 5 epochs, loss 0.1132 |
| Zero-shot | `ChronoGLiNER-ZS` | GLiNER used directly with simple labels | 0.50 | 0.3082 | 0.4921 | 0.3790 | 0.58 | 0.20 | 0.44 | 0.00 | 0.04 | 0.25 | 0.46 | Best simple-label result |
| Descriptive labels | `ChronoGLiNER-D` | GLiNER with richer historical label descriptions | 0.35 | 0.4574 | 0.3855 | 0.4184 | — | — | — | — | — | — | — | Best prompt-only result with French historical labels |
| Fine-tuned | `ChronoGLiNER-FT` | GLiNER fine-tuned on HIPE `NE-COARSE-LIT` | 0.35 | 0.5699 | 0.7223 | 0.6371 | 0.80 | 0.41 | **0.69** | 0.24 | 0.30 | 0.49 | 0.69 | Best fine-tuned run so far, 15k steps |
| Teacher model | `ChronoGLiNER-Teacher` | Fine-tuned GLiNER used to pseudo-label more historical data | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Planned |
| Student model | `ChronoGLiNER-BERT` | Historical BERT trained on gold + GLiNER pseudo-labels | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Planned |
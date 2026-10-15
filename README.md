# Cybercrime Complaint Understanding System

Independent implementation of Rani et al. (2024), [Automated Classification of Cybercrime Complaints using Transformer-based Language Models for Hinglish Texts](https://arxiv.org/abs/2412.16614).

Fine-tune BERT, RoBERTa, HingBERT and HingRoBERTa to classify complaints into the paper's 14 categories, with preprocessing, optional train-only GenAI augmentation, baseline comparisons, and a local demo.

**Status:** code implementation; no original I4C dataset or trained research model included. Synthetic fixtures support smoke tests only. Read [RESEARCH.md](RESEARCH.md) for assumptions and differences. Classification is the paper's task; entity extraction and case summarization are not claimed as reproduced contributions.

## Setup

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,app]'
python -m spacy download en_core_web_sm
```

Install a PyTorch build suitable for your GPU host's driver using the official PyTorch installation selector if needed. This project does not administer the H200 server. Hugging Face downloads require network access; Llama access may require accepting its license and authenticating on your server.

## Dataset

JSONL, one record per line:

```json
{"text":"Mere account se paise transfer ho gaye bina permission ke.","label":"Online Financial Fraud"}
```

Labels use original Table III names or their simplified equivalents in `ccus/data.py`. Store raw data under ignored `data/raw/`. Record provenance and obtain appropriate access. Model files and prepared data are ignored by Git.

```bash
python -m ccus.prepare --input data/raw/train.jsonl --test data/raw/test.jsonl --spacy-model en_core_web_sm --linguistic
```

Omit `--test` if no official split exists. The resulting split is explicitly a replacement protocol. Do not input pre-generated paraphrases: split original complaints first. Regex/spaCy anonymization is imperfect and requires review, especially Hindi/Hinglish names and addresses.

## Training

```bash
python -m ccus.baseline --model random-forest --output artifacts/random-forest
python -m ccus.baseline --model adaboost --output artifacts/adaboost
python -m ccus.baseline --model knn --output artifacts/knn
python -m ccus.baseline --model xgboost --output artifacts/xgboost
python -m ccus.train --model hingroberta --output artifacts/hingroberta --bf16
```

For an 18 GB GPU slice, start with batch size 8 and length 256; memory needs depend on the installed stack. Add `--gradient-checkpointing` if needed. Use `--model bert`, `roberta`, `hingbert`, or `hingroberta`; use a separate output directory for every experiment. Hyperparameter options: `--learning-rate`, `--batch-size`, `--max-length`, `--epochs`, `--seed`.

Each run saves validation/test metrics, per-class reports, and a confusion matrix. Choose hyperparameters on validation only. No accuracy claims are made until your real evaluation completes.

## Optional augmentation

This uses a separate generative model and BERTScore, so it needs more memory than classifier training. On small GPU slices generate candidates on a larger allocation or change `--model`; record that deviation.

```bash
python -m ccus.augment --target-per-class 1000 --max-candidates 100
# Review artifacts/augmentation_candidates.jsonl; set approved=true for accepted rows.
python -m ccus.merge_augmentation --candidates artifacts/augmentation_candidates.jsonl
python -m ccus.train --data data/processed-augmented --output artifacts/hingroberta-augmented --bf16
```

Generation is bounded and may not fill the requested class target. Repeated invocations overwrite the candidate file; use separate `--output` files to preserve batches. All reviewed candidates must retain valid training source IDs. Validation/test stay unchanged. Check semantic duplication manually.

## Demo

```bash
streamlit run app.py
```

Choose your trained model directory in the sidebar. It displays category ranking, human review flags, preprocessing, token truncation warnings, and a downloadable classification report. It fails visibly if no model exists; it never substitutes made-up predictions. It does not save submitted text to a database.

## CPU smoke test

```bash
pip install -e .
python -m unittest discover -s tests -v
python -m ccus.prepare --input data/synthetic_smoke.jsonl --output data/processed-smoke
python -m ccus.baseline --data data/processed-smoke --output artifacts/smoke
```

The fixture is small, artificial and unsuitable for research evaluation.

## Private GitHub repository

If running this downloaded source locally, install GitHub CLI, run `gh auth login`, then:

```bash
bash scripts/publish_github.sh
```

This creates **a private** `cybercrime-complaint-understanding` repository under the authenticated account and pushes the code. No credentials belong in source files.

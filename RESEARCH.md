# Research basis and fidelity

Nanda Rani, Divyanshu Singh, Bikash Saha, Sandeep Kumar Shukla (2024).
**Automated Classification of Cybercrime Complaints using Transformer-based Language Models for Hinglish Texts.**
https://arxiv.org/abs/2412.16614 (implementation targets v1, 21 December 2024).

This is an independent method implementation, not the authors' code or an exact numerical reproduction.
The inspected preprint says the dataset and tool will be released upon acceptance. We do not have their I4C data.
No pretrained cybercrime classifier or fabricated benchmark results are included.

| Paper component | Implementation | Fidelity / unresolved detail |
|---|---|---|
| III-B dataset splits | `ccus.prepare`, original test supported, 20% training validation | Without original test, use documented per-class 64/16/20 approximation |
| III-C deduplication and label mapping | `ccus.data` | Uses all 14 Table III labels; fails visibly on unknown labels |
| III-C entity placeholders | `ccus.preprocess` | Regex phone/email/URL/money, spaCy PERSON/GPE/LOC/MONEY; spaCy entity recall is imperfect |
| III-C stopwords and lemmas | `--spacy-model en_core_web_sm --linguistic` | English linguistic processing; Hinglish rules/resources unspecified in paper |
| III-D GenAI paraphrases | `ccus.augment` | Paper names “LLaMA 3.1-7b”; official Llama 3.1 uses an 8B size. Default explicitly substitutes 8B Instruct |
| III-D BERT F1 >= .97 | BERTScore multilingual BERT | Paper does not specify exact scorer checkpoint/layers/rescaling; our scorer is an explicit assumption |
| III-E BERT/RoBERTa/HingBERT/HingRoBERTa | `ccus.train --model ...` | Public model IDs selected; paper does not pin model revisions |
| III-F AdamW, lr 1e-5–3e-5, batches 8–32, length 128/256 | CLI training arguments | Default lr 2e-5, batch 8, length 256; early stopping patience 5 |
| IV-B TF-IDF + SVD baselines | `ccus.baseline` RF/AdaBoost/kNN/XGBoost | SVD dimensions and estimator parameters are implementation choices; logistic is an extra baseline |
| IV-A metrics | accuracy, macro/weighted F1, per-class precision/recall, confusion matrix | Paper F1 averaging is unclear; report both rather than assert comparability |
| III-G web tool | Streamlit `app.py` | Interface substitution for Django + JS; classification method unaffected |

The paper reports 74.41% accuracy and 71.49% F1 for HingRoBERTa; these are **paper results**, not this project's results.

## Experimental protocol

1. Acquire permitted real complaint data, record provenance, and manually review labels and redactions.
2. Prepare fixed splits before generating paraphrases. Human-review near-duplicates across splits: normalized exact-match checking does not detect all semantic duplicates.
3. Train baseline and four transformer variants on identical unaugmented splits.
4. Generate training-only minority-class paraphrases; inspect semantics and labels, then approve candidates.
5. Train augmented variants using unchanged validation and test data.
6. Select learning rate / batch size / sequence length using validation macro-F1 only. Evaluate final selected configuration once on test. Individual runs write test reports; do not use those to select runs.
7. Repeat selected experiments with seeds 42/43/44 and report mean and standard deviation. Do not compare synthetic fixture accuracy with real-world paper scores.

Keep synthetic fixtures out of your claimed research evaluation. Any local dataset replacement changes the study: describe it as a method reimplementation on another dataset.

## Citation

```bibtex
@misc{rani2024cybercrime,
 title={Automated Classification of Cybercrime Complaints using Transformer-based Language Models for Hinglish Texts},
 author={Nanda Rani and Divyanshu Singh and Bikash Saha and Sandeep Kumar Shukla},
 year={2024}, eprint={2412.16614}, archivePrefix={arXiv}, primaryClass={cs.CR}
}
```

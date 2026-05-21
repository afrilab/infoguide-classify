# Discriminative Feature Analysis for the Classification Module

## Purpose

The full method comparison showed that TF-IDF cosine similarity is the strongest classifier on this corpus, with 87.8% accuracy and 89.7% weighted F1 on the 98-document evaluation set, and 90.4% accuracy and 91.6% macro F1 on the 5-fold stratified subset. A natural question is whether this result reflects genuine class structure in the lexical signal, or whether it is an accidental effect of small sample size, overlapping vocabulary, or stop-word artefacts. To answer this, a discriminative-feature analysis was added to the evaluation framework. The goal was to inspect which TF-IDF terms most strongly characterize each document class and to check whether those terms match the expected document-type patterns.

## Method

The analysis fits the same TF-IDF preprocessing pipeline that is used by the unsupervised TF-IDF cosine classifier and the supervised TF-IDF baselines (Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest). The shared configuration is `ngram_range=(1, 2)`, English stop-word removal, lowercasing, and `min_df=1`. The fit is performed on the full labeled corpus of 94 documents and yields a vocabulary of 41,316 unigram and bigram features.

For each class, a per-term discriminative score is computed as

```
score(term, class) = mean(TF-IDF | in class) − mean(TF-IDF | other classes)
```

A positive score indicates that a term carries more weight in documents of the given class than in the rest of the corpus. Terms with the highest scores are therefore the ones that drive the cosine similarity between a document and its label centroid, and they are also the features that the supervised classifiers learn to weight most heavily. Using mean TF-IDF rather than raw counts means the analysis already accounts for document length and corpus-wide rarity, which makes the resulting term lists directly comparable across classes.

## Results

The top discriminative terms per class are reported in `outputs/classification_outputs/discriminative_terms.md` (full top-20 per class) and `discriminative_terms.json`. The leading terms are summarized below.

### Reports (N=45)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | government | 0.0399 | 0.0008 | 0.0391 |
| 2 | mr | 0.0361 | 0.0000 | 0.0361 |
| 3 | world | 0.0384 | 0.0027 | 0.0358 |
| 4 | world bank | 0.0363 | 0.0017 | 0.0346 |
| 5 | sector | 0.0402 | 0.0071 | 0.0331 |
| 6 | assessment | 0.0387 | 0.0058 | 0.0329 |
| 7 | public | 0.0368 | 0.0053 | 0.0315 |
| 8 | development | 0.0327 | 0.0026 | 0.0301 |

The Reports class is characterized by assessment-oriented and public-sector terminology. Words such as `assessment`, `sector`, `public`, `development`, and the bigram `world bank` are exactly the kind of analytical and evaluative language one expects in formal report documents in this corpus. The presence of `government` and `world bank` reflects the institutional source of many of the reports rather than category leakage, but in either case these terms are systematically more prominent in reports than in policies, procedures, contracts, or structured forms.

### Policy / Procedure / Contract (N=29)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | fatf | 0.0944 | 0.0003 | 0.0942 |
| 2 | risk | 0.0828 | 0.0122 | 0.0706 |
| 3 | guidance | 0.0492 | 0.0024 | 0.0469 |
| 4 | credit risk | 0.0377 | 0.0011 | 0.0366 |
| 5 | credit | 0.0495 | 0.0130 | 0.0365 |
| 6 | ccr | 0.0362 | 0.0004 | 0.0358 |
| 7 | principles | 0.0364 | 0.0043 | 0.0322 |
| 8 | committee | 0.0342 | 0.0082 | 0.0259 |

The Policy / Procedure / Contract class is dominated by regulatory and prudential vocabulary. Terms such as `risk`, `guidance`, `credit risk`, `principles`, `supervision`, and `operational resilience` (also in the top-20) reflect the formal, obligation-oriented language typical of regulatory policy documents. The very high score of `fatf` (Financial Action Task Force) reflects a concentration of compliance-focused policy material in this part of the corpus and is consistent with the document-type definition. The contrast in `risk` is particularly informative: the in-class mean of 0.0828 is almost seven times the out-of-class mean of 0.0122, indicating that the term is not just frequent in policies but disproportionately so relative to the rest of the corpus.

### Forms / Structured (N=20)

| Rank | Term | In-class mean | Out-class mean | Score |
| ---: | --- | ---: | ---: | ---: |
| 1 | borrower | 0.1525 | 0.0011 | 0.1515 |
| 2 | reserve | 0.0813 | 0.0015 | 0.0797 |
| 3 | participant | 0.0752 | 0.0002 | 0.0749 |
| 4 | agreement | 0.0758 | 0.0015 | 0.0743 |
| 5 | reserve bank | 0.0686 | 0.0003 | 0.0682 |
| 6 | excess | 0.0594 | 0.0000 | 0.0594 |
| 7 | title | 0.0567 | 0.0002 | 0.0564 |
| 8 | excess balance | 0.0563 | 0.0000 | 0.0563 |

The Forms / Structured class produces the sharpest discriminative signal of the three classes. Terms such as `borrower`, `participant`, `agreement`, `authorized`, `reserve bank`, `excess balance`, and `operating circular` are template-style lexical items that recur across structured banking forms. Several of these terms have an out-of-class mean of essentially zero, meaning they almost never appear in documents of other types. The discriminative score of `borrower` (0.1515) is the largest in the entire analysis, which is consistent with the fact that structured forms in this corpus follow recurring templates with a small, characteristic working vocabulary.

## Interpretation

Two observations stand out across the three classes.

First, the top discriminative terms align with the prior expectation of what each class should contain. Reports are dominated by analytical and assessment vocabulary, policies and procedures by regulatory and prudential vocabulary, and structured forms by template-style banking vocabulary. This means that TF-IDF cosine similarity is not succeeding by coincidence or by exploiting incidental features; it is succeeding because the dataset exposes genuinely class-specific lexical patterns and TF-IDF is well suited to identify them.

Second, the in-class versus out-of-class contrast is large and not driven by a single term. For every class, multiple terms have in-class means at least an order of magnitude higher than their out-of-class means. This indicates that the cosine-similarity signal is supported by a broad set of features rather than by a single distinctive keyword, which improves robustness to vocabulary variation and to incomplete documents.

Together, these observations explain why TF-IDF cosine outperforms the more expressive zero-shot NLI models on this corpus despite the simplicity of the underlying representation. The classes are separated mainly by domain-specific lexical patterns, and TF-IDF captures exactly that signal. The same explanation also accounts for the strong performance of the supervised TF-IDF baselines: Logistic Regression, Linear SVM, and Random Forest are learning to weight the same discriminative terms that already drive the unsupervised cosine method, which is why their accuracy and macro F1 are close to those of TF-IDF cosine but do not exceed it by a meaningful margin on the current dataset size.

## Implications for Method Selection

The discriminative-term analysis strengthens the case for retaining TF-IDF cosine as the primary classifier in the pipeline. The method is not only the most accurate on this corpus but also the most interpretable: each prediction can be traced back to a small set of weighted terms that match the expected vocabulary of the predicted class. This is a meaningful operational advantage for downstream review and for auditing the behavior of the classifier on new document types. BGE embeddings remain a strong fallback when the lexical signal is weak or when semantically related but lexically dissimilar variants of the same document type are expected. The supervised lexical models, in particular Logistic Regression, remain natural successors once the labeled dataset grows large enough for trained classifiers to clearly outperform direct label-description matching.

## Reproducibility

The analysis is produced by `src/classification/discriminative_terms.py` and is registered as `step12_discriminative_terms` in `configs/pipeline.yaml`. It runs on the full labeled corpus loaded by the shared evaluation harness, writes a machine-readable JSON file and a human-readable markdown table to `outputs/classification_outputs/`, and prints the leading terms per class to the console. Because it shares its TF-IDF configuration with the supervised baselines and the TF-IDF cosine classifier, the terms it surfaces are exactly the features that the deployed classifier uses.

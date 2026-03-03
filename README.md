# Traditional NLP vs Transformer for News Classification

## 1. Project Motivation

Transformer-based models such as BERT are widely considered state-of-the-art in NLP tasks.  
However, in practical industry scenarios, classical machine learning methods are often significantly cheaper and sometimes equally effective.

This project investigates:

> Do transformer models always outperform traditional NLP approaches in structured news classification tasks?  
> What are the performance–efficiency trade-offs?

---

## 2. Task Description

**Dataset:** AG News Topic Classification  
**Classes (4):**
- World
- Sports
- Business
- Sci/Tech

**Data size:**
- 120,000 training samples
- 7,600 test samples

For faster experimentation:
- Training subset: 20,000 samples
- Evaluation subset: 5,000 samples

**Source:**
- Hugging Face: https://huggingface.co/datasets/datasets/ag_news
- Original corpus: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

---

## 3. Methods

### 3.1 Baseline: Traditional NLP

- TF-IDF feature extraction
- Logistic Regression classifier
- Feature importance analysis

Advantages:
- Extremely fast
- Interpretable
- Low computational cost

---

### 3.2 Transformer: DistilBERT Fine-Tuning

- Pretrained `distilbert-base-uncased`
- Fine-tuned for 4-class classification
- 1 epoch training
- Evaluated on both CPU and GPU

Advantages:
- Strong semantic representation
- Context-aware language understanding

---

## 4. Experimental Results

### Performance Comparison

| Model | Macro-F1 | Accuracy | Time |
|-------|----------|----------|------|
| TF-IDF + Logistic Regression | **0.9217** | 0.9218 | 19.85 s |
| DistilBERT (CPU) | 0.9171 | 0.9178 | 1612.85 s (~26.9 min) |
| DistilBERT (GPU RTX 4060) | 0.9167 | 0.9174 | 121.14 s (~2.0 min) |

---

## 5. Key Findings

1. Traditional TF-IDF + Logistic Regression slightly outperformed DistilBERT on this structured news classification task.

2. Transformer models required significantly higher computational cost.

3. GPU acceleration reduced training time by ~13× (26.9 min → 2.0 min).

4. For keyword-driven classification problems, classical linear models remain highly competitive.

5. Model selection in production systems should consider both predictive performance and computational efficiency.

---

## 6. Model Interpretability

The Logistic Regression baseline identified meaningful class-specific keywords:

- Sports: *game, team, season, win*
- Business: *market, company, stock*
- World: *government, country, war*
- Sci/Tech: *software, internet, technology*

This confirms that TF-IDF captured domain-relevant vocabulary effectively.

---

## 7. Repository Structure

baseline_tfidf_lr.py       # Traditional NLP baseline
finetune_distilbert.py     # Transformer fine-tuning
results.md                 # Detailed experiment outputs
README.md                  # Project documentation

---

## 8. How to Run

### Install Dependencies

pip install transformers datasets scikit-learn torch accelerate

### Run Baseline

python baseline_tfidf_lr.py

### Run Transformer (GPU recommended)

python finetune_distilbert.py

---

## 9. Conclusion

This project demonstrates that transformer models do not universally outperform classical NLP methods.  

In structured classification tasks with strong lexical signals, linear models with engineered features can match or exceed transformer performance at a fraction of computational cost.

Understanding such trade-offs is essential for real-world AI deployment.
## Task
AG News text classification (4 categories)

Train subset: 20,000 samples  
Evaluation subset: 5,000 samples  

---

## Baseline: TF-IDF + Logistic Regression

Accuracy: **0.9218**  
Macro-F1: **0.9217**  
Training + inference time: **19.85 seconds**

---

## Transformer: DistilBERT (Fine-tuning, 1 epoch)

### CPU
Accuracy: **0.9178**  
Macro-F1: **0.9171**  
Training + evaluation time: **1612.85 seconds (~26.9 minutes)**

### GPU (NVIDIA RTX 4060)
Accuracy: **0.9174**  
Macro-F1: **0.9167**  
Training + evaluation time: **121.14 seconds (~2.0 minutes)**

---

## Comparison & Analysis

| Model | Macro-F1 | Time |
|------|---------|------|
| TF-IDF + Logistic | **0.9217** | 19.85 s |
| DistilBERT (GPU) | 0.9167 | 121 s |
| DistilBERT (CPU) | 0.9171 | 1613 s |

**Key observations**

- The traditional TF-IDF + Logistic Regression baseline slightly outperformed DistilBERT on this structured news classification task.
- DistilBERT required significantly higher computational cost.
- GPU acceleration reduced training time by **~13×** (26.9 min → 2.0 min).
- For keyword-driven classification problems, linear models with engineered features remain highly competitive.
- Transformer models provide stronger semantic representation but should be selected based on performance–efficiency trade-offs.

---

## Conclusion

This experiment demonstrates the importance of comparing traditional and deep learning approaches and evaluating both performance and computational efficiency when selecting models for production scenarios.

## Model Interpretability

Top important words learned by the Logistic Regression model:

World: iran, palestinian, afp afp, athens greece, president, nuclear, canadian press, iraqi, afp, iraq 
Sports: players, olympic, baseball, stadium, season, sports, league, team, cup, coach 
Business: business, airlines, corp, tax, company, economic, bank, hellip, economy, oil 
Sci/Tech: reuters reuters, apple, technology, web, microsoft, scientists, software, space, nasa, internet 

The results show that the model captures meaningful domain-specific vocabulary for each category.
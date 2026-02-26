===== Baseline Results =====
Accuracy: 0.9218
Macro-F1: 0.9217
Training + inference time: 19.85 seconds

Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.91      0.92      1900
           1       0.96      0.98      0.97      1900
           2       0.90      0.89      0.89      1900
           3       0.90      0.90      0.90      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

## Model Interpretability

Top important words learned by the Logistic Regression model:

World: iran, palestinian, afp afp, athens greece, president, nuclear, canadian press, iraqi, afp, iraq 
Sports: players, olympic, baseball, stadium, season, sports, league, team, cup, coach 
Business: business, airlines, corp, tax, company, economic, bank, hellip, economy, oil 
Sci/Tech: reuters reuters, apple, technology, web, microsoft, scientists, software, space, nasa, internet 

The results show that the model captures meaningful domain-specific vocabulary for each category.
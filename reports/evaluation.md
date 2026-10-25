# Clause classifier evaluation

- Test rows: 300 (random paragraphs from 46 policy documents, never used for training)
- Test labels: made by an LLM (Claude), not human-verified; scores mean agreement with those labels
- **Accuracy: 0.8233**
- **Macro-F1: 0.8046**

## Per-class report
```
                precision    recall  f1-score   support

        claims       0.70      1.00      0.82        33
      coverage       0.64      0.62      0.63        29
    definition       0.94      0.97      0.96        69
     exclusion       0.94      0.79      0.86        42
         limit       0.87      0.83      0.85        24
         other       0.91      0.59      0.72        49
       premium       0.75      0.92      0.83        13
       renewal       0.72      0.82      0.77        22
waiting_period       0.74      0.89      0.81        19

      accuracy                           0.82       300
     macro avg       0.80      0.83      0.80       300
  weighted avg       0.84      0.82      0.82       300

```

## Confusion matrix (rows = true label, columns = predicted)
```
                claims  coverage  definition  exclusion  limit  other  premium  renewal  waiting_period
claims              33         0           0          0      0      0        0        0               0
coverage             4        18           0          0      2      0        1        1               3
definition           1         1          67          0      0      0        0        0               0
exclusion            1         4           3         33      0      0        0        0               1
limit                0         3           0          0     20      0        0        1               0
other                8         1           1          2      1     29        1        4               2
premium              0         0           0          0      0      0       12        1               0
renewal              0         0           0          0      0      2        2       18               0
waiting_period       0         1           0          0      0      1        0        0              17
```

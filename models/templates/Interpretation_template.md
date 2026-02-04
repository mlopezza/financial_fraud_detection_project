# Evaluation template

We'll focus on key metrics like classification report, confusion matrix, ROC AUC, PR AUC, F1, precision, recall, and possibly coefficients or feature importance.

## 1. Core classification metrics

These come from `classification_report` and your confusion matrix.

### 1.1 Confusion matrix

You usually have:

- **TP (True Positives):** fraud correctly flagged
- **FP (False Positives):** legit flagged as fraud
- **FN (False Negatives):** fraud missed
- **TN (True Negatives):** legit correctly ignored

**How to interpret:**

- **High FN:** you’re _missing fraud_ → bad for risk
- **High FP:** you’re _annoying legit users_ → bad for UX
- **Tradeoff:** fraud detection is usually more tolerant of FP than FN, but you decide the balance.

**Template language:**

> “The model correctly identified X% of fraud cases (TP) but missed Y frauds (FN). It also incorrectly flagged Z legitimate transactions as fraud (FP), which impacts user experience and manual review load.”

---

### 1.2 Precision, recall, F1

From `classification_report`:

- **Precision (for fraud class):**  
  “Of all transactions we flagged as fraud, how many were actually fraud?”

- **Recall (for fraud class):**  
  “Of all actual frauds, how many did we catch?”

- **F1 score:**  
  Harmonic mean of precision and recall— good when we care about both.

**How to interpret:**

- **High recall, low precision:**  
  You catch most fraud but with many false alarms.
- **High precision, low recall:**  
  When you say “fraud”, you’re usually right—but you miss many frauds.
- **F1:**  
  Good single-number summary, but always interpret alongside precision/recall.

**Interpretation template language:**

> “The model achieves a precision of P and recall of R on the fraud class. This means that when it flags a transaction as fraud, it is correct P% of the time, and it successfully detects R% of all fraud cases. The F1 score of F reflects the balance between these two.”

**Example Usage of Template Language:**
“The model achieves a precision of 0.043 and recall of 0.043 on the fraud class. This means that when it flags a transaction as fraud, it is correct 0.043% of the time, and it successfully detects 0.043% of all fraud cases. The F1 score of 0.078 reflects the balance between these two.”

---

## 2. Threshold-dependent vs threshold-free metrics

### 2.1 ROC AUC

From `roc_curve` + `auc`:

- **ROC AUC:** probability that the model ranks a random fraud higher than a random legit transaction.
- Range: 0.5 (random) → 1.0 (perfect).

**How to interpret:**

- **0.5–0.6:** weak
- **0.6–0.7:** modest
- **0.7–0.8:** decent
- **0.8–0.9:** strong
- **>0.9:** very strong (check for leakage)

**Template language:**

> “The ROC AUC of A indicates that the model is able to rank fraud cases above legitimate ones with A probability, which reflects [weak/moderate/strong] discriminative power.”

---

### 2.2 Precision–Recall AUC (PR AUC)

From `precision_recall_curve` + `auc`:

- Much more informative than ROC AUC on **highly imbalanced data**.
- Focuses on performance on the positive class (fraud).

**How to interpret:**

- Compare PR AUC to the **baseline positive rate** (e.g., if fraud is 1%, a PR AUC of 0.2 is actually quite good).
- Higher PR AUC = better precision–recall tradeoff across thresholds.

**Template language:**

> “The PR AUC of P indicates that across different thresholds, the model maintains a strong balance between precision and recall on the fraud class, which is particularly meaningful given the low base rate of fraud.”

---

### 2.3 Threshold choice

You’re currently using the default threshold of 0.5. For fraud, that’s rarely optimal.

**Key ideas:**

- Lower threshold → higher recall, lower precision
- Higher threshold → higher precision, lower recall

You can:

- Pick a threshold that maximizes **F1**
- Or one that satisfies a **minimum recall** (e.g., recall ≥ 0.9)
- Or one that controls **FP rate** to match operational capacity

**Template language:**

> “The current results are based on a default threshold of 0.5. In practice, we can adjust this threshold to trade off between catching more fraud (higher recall) and reducing false alarms (higher precision), depending on business constraints.”

---

## 3. Model-specific interpretation

### 3.1 Logistic regression

**What you can interpret:**

- **Coefficients (`coef_`):** effect of each feature on the log-odds of fraud.
- **Sign:**
  - Positive → increases fraud risk
  - Negative → decreases fraud risk
- **Magnitude:** larger absolute value → stronger effect.

**Template language:**

> “In the logistic regression model, a one-unit increase in feature X is associated with a Y× change in the odds of fraud, holding other features constant. Positive coefficients indicate higher fraud risk, while negative coefficients indicate lower risk.”

We could also consider:

- **Multicollinearity:** correlated features can distort coefficients.
- **Regularization (C, penalty):** affects coefficient shrinkage and stability.

---

### 3.2 Tree-based models (RandomForest, XGBoost, etc.)

**What you can interpret:**

- **Feature importance:**
  - Gini importance / gain / split frequency
  - Tells you which features the model relies on most.
- **Global view:** which features matter overall.
- **Local view:** SHAP values or similar to explain individual predictions.

**How to interpret:**

- High importance ≠ causal, just predictive.
- Check if top features make domain sense (amount, velocity, geo_anomaly, device, etc.).

**Template language:**

> “The tree-based model relies most heavily on features such as A, B, and C. These features contribute the most to splitting decisions and therefore to the model’s fraud predictions. This aligns with domain expectations that [e.g., unusual amounts and high velocity] are strong indicators of fraud.”

---

## 4. Calibration and probabilities

Especially important for logistic regression, but also relevant for trees.

- **Well-calibrated model:** predicted probability ≈ actual frequency.
  - e.g., among transactions with predicted fraud probability ~0.8, about 80% are actually fraud.
- Tree models often need **calibration** (Platt scaling, isotonic regression).

**Template language:**

> “The model’s predicted probabilities are [well / poorly] calibrated. This means that a predicted fraud probability of p can [reasonably / not reliably] be interpreted as ‘p% chance of fraud’, which affects how thresholds and risk scores are used operationally.”

## 5. Putting it all together (narrative template)

Here’s a compact narrative you can reuse and adapt:

> **Performance summary**  
> The model achieves a precision of P and recall of R on the fraud class, with an F1 score of F. This means that when it flags a transaction as fraud, it is correct P% of the time, and it successfully detects R% of all fraud cases. The confusion matrix shows that this corresponds to TP true frauds caught, FN frauds missed, FP legitimate transactions incorrectly flagged, and TN legitimate transactions correctly ignored.
>
> **Ranking ability**  
> The ROC AUC of A indicates [weak/moderate/strong] ability to rank fraud cases above legitimate ones. The PR AUC of PR, compared to the base fraud rate of B, suggests that the model maintains a [good/strong/modest] precision–recall tradeoff across thresholds, which is particularly important given the class imbalance.
>
> **Threshold considerations**  
> These results are based on a default threshold of 0.5. In practice, we can adjust this threshold to prioritize either catching more fraud (higher recall) or reducing false positives (higher precision), depending on operational and business constraints.
>
> **Model interpretation**  
> For the logistic regression model, coefficients indicate that features X, Y, and Z are most strongly associated with increased fraud risk, while features A and B are associated with lower risk. For the tree-based model, feature importance shows that the model relies heavily on [top features], which aligns with domain expectations around fraud behavior.
>
> **Operational takeaway**  
> Overall, the model provides [useful / strong / promising] discriminative power for fraud detection, with clear tradeoffs between recall and precision that can be tuned via the decision threshold. The most influential features are consistent with domain knowledge, increasing confidence in the model’s behavior and its suitability for deployment with appropriate monitoring.

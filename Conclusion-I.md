# Conclusions (Part 1): Cells 0–22

- **Problem & Data**
  - Trained a binary classifier to predict `income` (`>50K` vs `<=50K`) using the Adult dataset (48,842 rows) after dropping `fnlwgt`, `education_num`, `race`, `native_country` for clearer explainability.
  - Mixed feature types (categorical: workclass, education, marital_status, occupation, relationship, sex; numeric: age, capital_gain, capital_loss, hours_per_week).

- **Model Training (LightGBM)**
  - Proper preprocessing: categorical casting and numeric float conversion enabled native handling by LightGBM.
  - Train/validation split (80/20) with AUC metric and early stopping configured.
  - Best validation AUC ≈ 0.929 on real-data split, indicating strong class separation.

- **Synthetic Data Generation (MostlyAI)**
  - A tabular generator was trained on the real table (time-limited: `max_training_time: 1`).
  - Validation loss consistently decreased across epochs, meaning the generator learned the real data distribution.
  - Training likely stopped due to the time budget (not lack of improvement).
  - Generated synthetic dataset matches real structure and size (48,842 x 11) but contains privacy-safe, newly sampled rows.

- **Evaluate Real-trained Model on Synthetic (TRTS)**
  - On synthetic data, model produced: Accuracy ≈ 83.0%, AUC ≈ 88.2% (from attached histogram figure).
  - The probability histogram shows clear separation: most `<=50K` near low probabilities; `>50K` concentrated at higher probabilities. This suggests synthetic data preserves key predictive patterns.

- **Global Explainability (SHAP Summary Bar Plot)**
  - Feature importance by mean(|SHAP|), highest to lowest:
    1) age
    2) marital_status
    3) capital_gain
    4) relationship
    5) education
    6) occupation
    7) hours_per_week
    8) capital_loss
    9) sex
    10) workclass
  - Interpretation: age and marital_status dominate global influence; income-related signals like capital_gain and work patterns/features follow.

- **Feature Behavior (SHAP Dependency Plots)**
  - age: Very negative SHAP at ~18–20; sharp increase through 20s; positive and plateau from ~30s–60s. Older age generally increases odds of `>50K`, with spread indicating interactions.
  - marital_status: Ordered (most negative → most positive) roughly: Never-married → Separated/Divorced/Married-spouse-absent → Widowed → Married-civ-spouse. Being Married-civ-spouse strongly pushes toward `>50K`; Never-married strongly toward `<=50K`.
  - hours_per_week: Below ~35 hours is negative; crosses near ~38–40; steadily more positive beyond 40 with increasing variance. More hours typically increase odds of `>50K`.

- **Takeaways so far**
  - The real-data-trained model generalizes to synthetic data with good AUC/accuracy, supporting the use of synthetic data for auditing/explanation.
  - Key drivers of higher predicted income: being older (post-20s), being married-civ-spouse, working longer hours, and having capital gains; patterns are plausible and align with domain intuition.
  - Variance in SHAP at the same feature values suggests interactions (e.g., age × occupation/education), motivating further per-sample inspection next.

- **Caveats/Limitations**
  - SHAP values indicate associations, not causation; categorical effects may proxy other socio-economic variables.
  - Metrics shown are on synthetic data; while fidelity looks strong, final validation on held-out real data is recommended for deployment decisions.
  - The chosen 0.5 threshold for accuracy is arbitrary; operating threshold should be tuned to the use case.

# Mortality Models

Created by **Jason Abi Chebli**  
© 2025 Jason Abi Chebli. All rights reserved.

## Description
<div style="text-align: justify;">This project applies advanced machine learning techniques to the MIMIC-III clinical dataset to predict ICU patient mortality. The goal was to model the <code>HOSPITAL_EXPIRE_FLAG</code> — an indicator of death during hospital stay — using features collected at ICU admission, including vitals, demographics, comorbidities, and diagnostic codes. I implemented multiple supervised learning algorithms, evaluated their performance using AUC-ROC, and explored the data to understand key predictors and patterns underlying ICU outcomes.</div>

## Data Source
[MIMIC-III Data](https://github.com/jabichebli/MortalityModels/tree/main/data)

## Data Exploration and Visualisation
<div style="text-align: justify;">I began with a thorough visual inspection of the dataset. Histograms revealed that features like glucose were highly skewed with outliers, while heart rate and temperature followed more normal distributions. UMAP projections showed distinct separation between ICU survivors and non-survivors, indicating strong multivariate structure. While no single variable was a strong univariate predictor, diagnosis codes and comorbidity information proved valuable.</div>

## Data Preprocessing
- Merged training features and labels
- Dropped irrelevant columns and extracted age from ADMITTIME and DOB
- One-hot encoded categorical variables
- Appended secondary ICD9 codes to capture comorbidities
- Applied z-score standardisation to numerical features
- Oversampled the minority class (deceased patients) to address class imbalance
- Ensured test data matched training structure exactly
Final feature matrix size: 37,080 × 7,037

## Models and Performance
### Logistic Regression
- Out-of-Sample AUC-ROC: 0.629
- A baseline linear model with limited predictive power.

### Decision Trees
- #### Regularised Tree
  - Tuned via 3×3 cross-validation
  - Out-of-Sample AUC-ROC: 0.796
- #### Random Forest
  - Tuned with 5×5 grid, 1000 trees (ranger engine)
  - Out-of-Sample AUC-ROC: 0.845
- #### Boosted Tree
  - Extensive tuning (mtry, depth, learning rate, etc.), 1000 trees
  - Out-of-Sample AUC-ROC: 0.928
- Ensemble models showed significantly improved performance.

### Neural Networks
- #### Simple NN
  - 1 hidden layer (2 nodes, ReLU)
  - Out-of-Sample AUC-ROC: 0.917
- #### Complex NN
  - 2 hidden layers (128 → 64), batch norm, dropout, L2 regularisation
  - Early stopping, Adam optimiser
  - Out-of-Sample AUC-ROC: 0.950
- The deep learning approach outperformed all other models on the validation set.

### Key Techniques
- UMAP for visualisation
- SMOTE-style oversampling
- Hyperparameter tuning with cross-validation
- Neural network regularisation strategies
- Use of secondary ICD9 codes for comorbidity features

## Feedback
If you have any questions, suggestions, or feedback, feel free to [contact me](https://jabichebli.github.io/jabichebli/contact/). Your input is valuable and will help improve my understanding of ML.

################################################################################
################################################################################
# The following script was written by: Jason Abi Chebli (31444059)
# The code was written for: ETC3250: Introduction to Machine Learning, Project Assessment (20%)
# Date submitted: 23-May-2025 
# Best Performing Model: Complex Neural Network (AUC-ROC of 0.950)
################################################################################
################################################################################


################################################################################
################################################################################
######################## Load the Packages & Data In ###########################
################################################################################
################################################################################

# Load all the necessary packages for this assignment
library(boot)
library(crosstalk)
library(detourr)
library(discrim)
library(ggpubr)
library(ggthemes)
library(GGally)
library(kableExtra)
library(knitr)
library(patchwork)
library(plotly)
library(randomForest)
library(readr)
library(rpart.plot)
library(tidymodels)
library(tidyverse)
library(tourr)
library(viridisLite)
library(xgboost)

# Read the necessary data in
mimic_train_X <- read_csv("Assignment 3/data/mimic_train_X.csv")
mimic_train_Y <- read_csv("Assignment 3/data/mimic_train_Y.csv")
mimic_test_X <- read_csv("Assignment 3/data/mimic_test_X.csv")
MIMIC_diagnoses <- read_csv("Assignment 3/data/MIMIC_diagnoses.csv")

################################################################################
################################################################################
############################ Pre-Process the Data ##############################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#------------------------ Pre-Process the Training Data ------------------------
#-------------------------------------------------------------------------------
# Merge main table with outcome
mimic_train <- mimic_train_X |>
  mutate(HOSPITAL_EXPIRE_FLAG = mimic_train_Y$HOSPITAL_EXPIRE_FLAG)

# Calculate the age of patients
mimic_train <- mimic_train |> 
  mutate(age = round(as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25, 1)) |> 
  mutate(age = ifelse(age > 89, 90, age))

# Drop unnecessary columns
mimic_train_clean <- mimic_train |>
  select(
    -`...1`, -icustay_id, -DOB, -ADMITTIME, -Diff, -DIAGNOSIS, -ICD9_diagnosis
  )

# One-hot encode categorical vars
categorical_vars <- c(
  "GENDER", "ETHNICITY", "MARITAL_STATUS", "RELIGION", "INSURANCE",
  "ADMISSION_TYPE", "FIRST_CAREUNIT"
)

mimic_train_encoded <- mimic_train_clean |>
  mutate(across(all_of(categorical_vars), as.factor)) |>
  fastDummies::dummy_cols(
    select_columns = categorical_vars,
    remove_selected_columns = TRUE,
    remove_first_dummy = TRUE
  )

# Clean and one-hot encode secondary ICD9 codes
MIMIC_diagnoses_clean <- MIMIC_diagnoses |>
  rename(subject_id = SUBJECT_ID, hadm_id = HADM_ID) |>
  filter(!is.na(ICD9_CODE)) |>
  distinct(subject_id, hadm_id, ICD9_CODE) |>
  mutate(ICD9_CODE = paste0("ICD9_diagnosis_", ICD9_CODE), value = 1)

diagnoses_onehot <- MIMIC_diagnoses_clean |>
  pivot_wider(
    names_from = ICD9_CODE,
    values_from = value,
    values_fill = list(value = 0)
  )

# Join diagnosis one-hot codes
mimic_train_with_diag <- mimic_train_encoded |>
  left_join(diagnoses_onehot, by = c("subject_id", "hadm_id")) |>
  mutate(across(starts_with("ICD9_"), ~ replace_na(., 0)))

# Drop IDs for final training set
mimic_train_final <- mimic_train_with_diag |> select(-subject_id, -hadm_id)

# Standardise numerical predictors
numeric_vars <- c(
  "HeartRate_Min", "HeartRate_Max", "HeartRate_Mean",
  "SysBP_Min", "SysBP_Max", "SysBP_Mean",
  "DiasBP_Min", "DiasBP_Max", "DiasBP_Mean",
  "MeanBP_Min", "MeanBP_Max", "MeanBP_Mean",
  "RespRate_Min", "RespRate_Max", "RespRate_Mean",
  "TempC_Min", "TempC_Max", "TempC_Mean",
  "SpO2_Min", "SpO2_Max", "SpO2_Mean",
  "Glucose_Min", "Glucose_Max", "Glucose_Mean",
  "age"
)

mimic_train_final <- mimic_train_final |>
  mutate(across(all_of(numeric_vars), ~ scale(.)[, 1]))

# Ensure outcome variable is a factor
mimic_train_final <- mimic_train_final |>
  mutate(HOSPITAL_EXPIRE_FLAG = as.factor(HOSPITAL_EXPIRE_FLAG))

# Balance Classes

# Identify majority and minority classes
majority_class <- mimic_train_final |> filter(HOSPITAL_EXPIRE_FLAG == "0")
minority_class <- mimic_train_final |> filter(HOSPITAL_EXPIRE_FLAG == "1")

# Oversample the minority class
set.seed(42)
oversampled_minority <- minority_class |>
  slice_sample(n = nrow(majority_class), replace = TRUE)

# Combine and shuffle
mimic_train_ready <- bind_rows(majority_class, oversampled_minority) |>
  slice_sample(prop = 1)

# Check class balance
table(mimic_train_ready$HOSPITAL_EXPIRE_FLAG)

#-------------------------------------------------------------------------------
#------------------------ Pre-Process the Testing Data -------------------------
#-------------------------------------------------------------------------------

# Calculate the age of patients
mimic_test_X <- mimic_test_X |> 
  mutate(age = round(as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25, 1)) |> 
  mutate(age = ifelse(age > 89, 90, age))


# Drop unnecessary columns
mimic_test_clean <- mimic_test_X |>
  select(
    -`...1`, -icustay_id, -DOB, -ADMITTIME, -Diff, -DIAGNOSIS, -ICD9_diagnosis
  )

# One-hot encode categorical variables
mimic_test_encoded <- mimic_test_clean |>
  mutate(across(all_of(categorical_vars), as.factor)) |>
  fastDummies::dummy_cols(
    select_columns = categorical_vars,
    remove_selected_columns = TRUE,
    remove_first_dummy = TRUE
  )

# Clean and one-hot encode secondary ICD9 codes for test set
MIMIC_diagnoses_clean_test <- MIMIC_diagnoses |>
  rename(subject_id = SUBJECT_ID, hadm_id = HADM_ID) |>
  filter(!is.na(ICD9_CODE)) |>
  distinct(subject_id, hadm_id, ICD9_CODE) |>
  mutate(ICD9_CODE = paste0("ICD9_diagnosis_", ICD9_CODE), value = 1)

diagnoses_onehot_test <- MIMIC_diagnoses_clean_test |>
  pivot_wider(
    names_from = ICD9_CODE,
    values_from = value,
    values_fill = list(value = 0)
  )

# Join ICD9 diagnosis codes to test data
mimic_test_with_diag <- mimic_test_encoded |>
  left_join(diagnoses_onehot_test, by = c("subject_id", "hadm_id")) |>
  mutate(across(starts_with("ICD9_"), ~ replace_na(., 0)))

# Drop IDs
mimic_test_final <- mimic_test_with_diag |> select(-subject_id, -hadm_id)

# Standardise numeric variables
mimic_test_final <- mimic_test_final |>
  mutate(across(all_of(numeric_vars), ~ scale(.)[, 1]))

# Ensure same columns as training data
# Add missing columns (present in train, missing in test)
missing_cols <- setdiff(names(mimic_train_ready), names(mimic_test_final))
mimic_test_final[missing_cols] <- 0

# Remove extra columns (present in test, not in train)
extra_cols <- setdiff(names(mimic_test_final), names(mimic_train_ready))
mimic_test_final <- mimic_test_final |> select(-all_of(extra_cols))

# Reorder to match training columns
mimic_test_ready <- mimic_test_final |>
  select(all_of(names(mimic_train_ready)))


################################################################################
################################################################################
######################## Train a Logistic Regression ###########################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# Logistic Regresssion takes a long time to run. As such, we will train it on only 8% of the very large data
set.seed(123)
mimic_subset <- mimic_train_ready |>
  initial_split(prop = 0.08, strata = HOSPITAL_EXPIRE_FLAG) |>
  training()

# Define the logistic regression model
logistic_mod <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification") 

# Fit logistic regression model
logistic_fit <- logistic_mod |>
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = mimic_subset)

#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Create a Confusion Matrix

# Make Predictions 
logistic_preds <- predict(logistic_fit, new_data = mimic_subset)

# Combine predictions with true labels
logistic_results <- bind_cols(logistic_preds, mimic_subset)

# Make Confusion Matrix
logistic_cm <- logistic_results |> select(HOSPITAL_EXPIRE_FLAG, .pred_class) |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |> 
  group_by(HOSPITAL_EXPIRE_FLAG) |> 
  mutate(cl_acc = n[.pred_class ==  HOSPITAL_EXPIRE_FLAG]/sum(n)) |> pivot_wider(names_from = .pred_class, values_from = n) |> 
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output confusion matrix
logistic_cm

# Determine Accuracy & Balanced Accuracy

# Augment test set with predicted probabilities and predicted classes (Logistic)
logistic_probs <- logistic_fit |> 
  augment(new_data = mimic_subset, type.predict = "prob") |>
  mutate(.pred_correct = if_else(HOSPITAL_EXPIRE_FLAG == "0", .pred_0, .pred_1),
         .pred_predicted = if_else(.pred_class == "0", .pred_0, .pred_1))

# Determine accuracy and balanced accuracy
logistic_accuracy <- accuracy(logistic_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
logistic_bal_accuracy <- bal_accuracy(logistic_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class) |> pull(.estimate)

# Create ROC Curve
logistic_probs |>
  roc_curve(truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |>
  autoplot()

# Determine AUC ROC
roc_auc(logistic_probs, truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |> pull(.estimate)

#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------

# Get predicted classes
logistic_test_preds <- predict(logistic_fit, new_data = mimic_test_ready)

# Bind the IDs from your original test set
logistic_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  bind_cols(logistic_test_preds) |>
  rename(HOSPITAL_EXPIRE_FLAG = .pred_class)

# Save as CSV
write.csv(logistic_predictions, "Assignment 3/logistic_predictions_final.csv", row.names = FALSE)



################################################################################
################################################################################
########################### Train a Decision Tree ##############################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# A regular Decision Tree takes a long time to run. As such, we will train it on only 8% of the very large data
set.seed(123)
mimic_subset <- mimic_train_ready |>
  initial_split(prop = 0.08, strata = HOSPITAL_EXPIRE_FLAG) |>
  training()

# Define the tunable decision tree spec
tune_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) |> 
  set_engine("rpart") |> 
  set_mode("classification")

# Create a grid of parameters to tune
tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 3
)

# Set up cross-validation folds
set.seed(234)
mimic_train_folds <- vfold_cv(mimic_subset, v = 3, strata = HOSPITAL_EXPIRE_FLAG)

# Create a workflow
tree_wf <- workflow() |>
  add_model(tune_spec) |>
  add_variables(outcomes = HOSPITAL_EXPIRE_FLAG, predictors = everything())

# Perform the tuning
set.seed(345)
tree_res <- tree_wf |> 
  tune_grid(
    resamples = mimic_train_folds,
    grid = tree_grid,
    metrics = NULL  # Computes a standard set of metrices
  )

# View and summarise results
tree_res |> collect_metrics() |> slice_head(n = 6)

# Find the best combination (based on AUC)
tree_top_5 <- tree_res |> show_best(metric = "roc_auc")
tree_best <- tree_res |> select_best(metric = "roc_auc")

# Finalise workflow and fit on training data
tuned_tree_wf <- tree_wf |> finalize_workflow(tree_best)
tuned_tree_fit <- tuned_tree_wf |> fit(data = mimic_subset)


#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Plot Decision Tree
tuned_tree_fit |> extract_fit_engine() |> rpart.plot(type = 3, extra = 1)

# Create a Confusion Matrix

# Make Predictions 
tuned_tree_train_preds <- predict(tuned_tree_fit, new_data = mimic_subset)

# Combine predictions with true labels
tuned_tree_train_results <- bind_cols(tuned_tree_train_preds, mimic_subset)

# Make Confusion Matrix
tuned_tree_train_cm <- tuned_tree_train_results |> select(HOSPITAL_EXPIRE_FLAG, .pred_class) |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |> 
  group_by(HOSPITAL_EXPIRE_FLAG) |> 
  mutate(cl_acc = n[.pred_class ==  HOSPITAL_EXPIRE_FLAG]/sum(n)) |> pivot_wider(names_from = .pred_class, values_from = n) |> 
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output confusion matrix
tuned_tree_train_cm

# Determine Accuracy & Balanced Accuracy

# Augment test set with predicted probabilities and predicted classes (Logistic)
tuned_tree_train_probs <- tuned_tree_fit |> 
  augment(new_data = mimic_subset, type.predict = "prob") |>
  mutate(.pred_correct = if_else(HOSPITAL_EXPIRE_FLAG == "0", .pred_0, .pred_1),
         .pred_predicted = if_else(.pred_class == "0", .pred_0, .pred_1))

# Determine accuracy and balanced accuracy
tuned_tree_train_accuracy <- accuracy(tuned_tree_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
tuned_tree_train_bal_accuracy <- bal_accuracy(tuned_tree_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class) |> pull(.estimate)

# Create ROC Curve
tuned_tree_train_probs |>
  roc_curve(truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |>
  autoplot()

# Determine AUC ROC
roc_auc(tuned_tree_train_probs, truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |> pull(.estimate)


#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------

# Get predicted classes
tuned_tree_test_preds <- predict(tuned_tree_fit, new_data = mimic_test_ready)

# Bind the IDs from your original test set
tuned_tree_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  bind_cols(tuned_tree_test_preds) |>
  rename(HOSPITAL_EXPIRE_FLAG = .pred_class)

# Save as CSV
write.csv(tuned_tree_predictions, "Assignment 3/tuned_tree_predictions_final.csv", row.names = FALSE)


################################################################################
################################################################################
########################### Train a Random Forest ##############################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# A Random Forest takes a long time to run. As such, we will train it on only 8% of the very large data
set.seed(123)
mimic_subset <- mimic_train_ready |>
  initial_split(prop = 0.08, strata = HOSPITAL_EXPIRE_FLAG) |>
  training()

mimic_subset <- mimic_subset |> 
  mutate(HOSPITAL_EXPIRE_FLAG = as.factor(HOSPITAL_EXPIRE_FLAG))

# remove near zero variance features because 7000+ variables is too much for Random Forest
library(caret)
nzv <- nearZeroVar(mimic_subset, saveMetrics = TRUE)
mimic_subset <- mimic_subset[, !nzv$nzv]


# Define the tunable random forest spec
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

# Create a grid of parameters to tune
rf_grid <- grid_regular(
  mtry(range = c(1, 5)),
  min_n(range = c(2, 20)),
  levels = 5
)

# Set up cross-validation folds
set.seed(234)
rf_folds <- vfold_cv(mimic_subset, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

# Create a workflow
rf_wf <- workflow() |>
  add_model(rf_spec) |>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

# Perform the tuning sequentially
set.seed(345)
rf_res <- rf_wf |>
  tune_grid(
    resamples = rf_folds,
    grid = rf_grid,
    metrics = NULL
  )

# View and summarise results
rf_res |> collect_metrics() |> slice_head(n = 6)

# Find the best combination (based on AUC)
rf_top_5 <- rf_res |> show_best(metric = "roc_auc")
rf_best <- rf_res |> select_best(metric = "roc_auc")

# Finalise workflow and fit on training data
final_rf_wf <- rf_wf |> finalize_workflow(rf_best)
final_rf_fit <- final_rf_wf |> fit(data = mimic_subset)

#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Create a Confusion Matrix

# Make Predictions 
rf_train_preds <- predict(final_rf_fit, new_data = mimic_subset)

# Combine predictions with true labels
rf_train_results <- bind_cols(rf_train_preds, mimic_subset)

# Make Confusion Matrix
rf_train_cm <- rf_train_results |> select(HOSPITAL_EXPIRE_FLAG, .pred_class) |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |> 
  group_by(HOSPITAL_EXPIRE_FLAG) |> 
  mutate(cl_acc = n[.pred_class ==  HOSPITAL_EXPIRE_FLAG]/sum(n)) |> pivot_wider(names_from = .pred_class, values_from = n) |> 
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output confusion matrix
rf_train_cm

# Determine Accuracy & Balanced Accuracy

# Augment test set with predicted probabilities and predicted classes (Logistic)
rf_train_probs <- final_rf_fit |> 
  augment(new_data = mimic_subset, type.predict = "prob") |>
  mutate(.pred_correct = if_else(HOSPITAL_EXPIRE_FLAG == "0", .pred_0, .pred_1),
         .pred_predicted = if_else(.pred_class == "0", .pred_0, .pred_1))

# Determine accuracy and balanced accuracy
rf_train_accuracy <- accuracy(rf_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
rf_train_bal_accuracy <- bal_accuracy(rf_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class) |> pull(.estimate)

# Create ROC Curve
rf_train_probs |>
  roc_curve(truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |>
  autoplot()

# Determine AUC ROC
roc_auc(rf_train_probs, truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |> pull(.estimate)

#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------

# Get predicted classes
rf_test_preds <- predict(final_rf_fit, new_data = mimic_test_ready)

# Bind the IDs from your original test set
rf_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  bind_cols(rf_test_preds) |>
  rename(HOSPITAL_EXPIRE_FLAG = .pred_class)

# Save as CSV
write.csv(rf_predictions, "Assignment 3/random_forest_predictions_final.csv", row.names = FALSE)



################################################################################
################################################################################
############################ Train a Boosted Tree ##############################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# Define the tunable boosted tree spec
boost_spec <- boost_tree(
  mtry = tune(),       
  min_n = tune(),       
  tree_depth = tune(),      
  learn_rate = tune(), 
  loss_reduction = tune(), 
  sample_size = tune(),    
  trees = 1000       
) |>
  set_engine("xgboost", nthread = parallel::detectCores() - 1) |>
  set_mode("classification")

# Create a grid of parameters to tune using grid_regular
boost_grid <- grid_max_entropy(
  mtry(range = c(1, 20)),
  min_n(range = c(2, 40)),              
  tree_depth(range = c(2, 10)),         
  learn_rate(range = c(0.001, 0.3)), 
  loss_reduction(range = c(0, 10)),     
  sample_size = sample_prop(range = c(0.5, 1.0)),  
  size = 50                         
)

# Set up cross-validation folds
set.seed(234)
boost_folds <- vfold_cv(mimic_train_ready, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

# Create a workflow
boost_wf <- workflow() |>
  add_model(boost_spec) |>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

# Perform the tuning using multi-core processing to speed up compute
library(doParallel)
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

set.seed(345)
boost_res <- boost_wf |>
  tune_grid(
    resamples = boost_folds,
    grid = boost_grid,
    metrics = NULL
  )

stopCluster(cl)
registerDoSEQ()

# View and summarise results
boost_res |> collect_metrics() |> slice_head(n = 6)
autoplot(boost_res)


# Find the best combination (based on AUC)
boosted_top_5 <- boost_res |> show_best(metric ="roc_auc")
boost_best <- boost_res |> select_best(metric = "roc_auc")

# Finalise workflow and fit on training data
final_boost_wf <- boost_wf |> finalize_workflow(boost_best)
final_boost_fit <- final_boost_wf |> fit(data = mimic_train_ready)


#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Create a Confusion Matrix

# Make Predictions 
boost_train_preds <- predict(final_boost_fit, new_data = mimic_train_ready)

# Combine predictions with true labels
boost_train_results <- bind_cols(boost_train_preds, mimic_train_ready)

# Make Confusion Matrix
boost_train_cm <- boost_train_results |> select(HOSPITAL_EXPIRE_FLAG, .pred_class) |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |> 
  group_by(HOSPITAL_EXPIRE_FLAG) |> 
  mutate(cl_acc = n[.pred_class ==  HOSPITAL_EXPIRE_FLAG]/sum(n)) |> pivot_wider(names_from = .pred_class, values_from = n) |> 
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output confusion matrix
boost_train_cm

# Determine Accuracy & Balanced Accuracy

# Augment test set with predicted probabilities and predicted classes (Logistic)
boost_train_probs <- final_boost_fit |> 
  augment(new_data = mimic_train_ready, type.predict = "prob") |>
  mutate(.pred_correct = if_else(HOSPITAL_EXPIRE_FLAG == "0", .pred_0, .pred_1),
         .pred_predicted = if_else(.pred_class == "0", .pred_0, .pred_1))

# Determine accuracy and balanced accuracy
boost_train_accuracy <- accuracy(boost_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
boost_train_bal_accuracy <- bal_accuracy(boost_train_probs, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class) |> pull(.estimate)

# Create ROC Curve
boost_train_probs |>
  roc_curve(truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |>
  autoplot()

# Determine AUC ROC
roc_auc(boost_train_probs, truth = HOSPITAL_EXPIRE_FLAG, .pred_0) |> pull(.estimate)

#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------

# Get predicted classes
boost_test_preds <- predict(final_boost_fit, new_data = mimic_test_ready, type = "prob") |> select(.pred_1)

# Bind the IDs from your original test set
boost_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  bind_cols(boost_test_preds) |>
  rename(HOSPITAL_EXPIRE_FLAG = .pred_1)

# Save as CSV
write.csv(boost_predictions, "Assignment 3/boost_predictions_final.csv", row.names = FALSE)


################################################################################
################################################################################
######################## Train a Simple Neural Network #########################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# Split the tidy training data into x and y again
mimic_train_x <- mimic_train_ready |> select(-HOSPITAL_EXPIRE_FLAG)
mimic_train_y <- mimic_train_ready$HOSPITAL_EXPIRE_FLAG |> as.integer() - 1 # mimic_train_y <- mimic_train_ready |> select(HOSPITAL_EXPIRE_FLAG) 

mimic_train_x <- as.matrix(mimic_train_x)
mimic_train_y <- as.numeric(mimic_train_y)

# Set the seed for reproducability
library(keras)
tensorflow::set_random_seed(211)

# Define the feed forward neural network model (the number of layers, number of inputs, number of nodes, number of outputs, activation function, output function)
nn_mimic_model <- keras_model_sequential() |>
  layer_dense(units = 2, activation = 'relu', input_shape = ncol(mimic_train_x)) |> 
  layer_dense(units = 1, activation = 'sigmoid')

nn_mimic_model |> summary()

# Define the optimizer, loss function and compile the neural network model
nn_mimic_model |> compile(
  optimizer = "adam",
  loss      = "binary_crossentropy",
  metrics   = c('accuracy')
)

# Fit the neural network model
nn_mimic_fit <- nn_mimic_model |> 
  keras::fit(
    x = mimic_train_x, 
    y = mimic_train_y,
    epochs = 200,
    batch_size = 32, 
    verbose = 0
  )

# Determine the final Loss and Accuracy
nn_mimic_model |> 
  evaluate(mimic_train_x, mimic_train_y, verbose = 0)

#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Confusion Matrix

# Get predicted probabilities (between 0 and 1)
nn_train_probs <- nn_mimic_model |> predict(mimic_train_x)

# Convert probabilities to predicted class (threshold = 0.5)
nn_train_preds <- ifelse(nn_train_probs > 0.5, 1, 0)

# Combine with true labels
nn_train_results <- tibble(
  HOSPITAL_EXPIRE_FLAG = mimic_train_y,
  .pred_class = nn_train_preds,
  .pred_prob = nn_train_probs
)

# Create a confusion matrix
nn_cm <- nn_train_results |>
  count(HOSPITAL_EXPIRE_FLAG, .pred_class) |>
  group_by(HOSPITAL_EXPIRE_FLAG) |>
  mutate(cl_acc = n[.pred_class == HOSPITAL_EXPIRE_FLAG] / sum(n)) |>
  pivot_wider(names_from = .pred_class, values_from = n) |>
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output the confusion matrix
nn_cm

# Determine Accuracy & Balanced Accuracy

# Convert true class to factor for yardstick
nn_train_results <- nn_train_results |>
  mutate(
    HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1)),
    .pred_class = factor(.pred_class, levels = c(0, 1))
  )

# Determine accuracy and balanced accuracy 
nn_accuracy <- accuracy(nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
nn_bal_accuracy <- bal_accuracy(nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate

# Determine AUC ROC
nn_train_results <- nn_train_results |>
  mutate(.pred_prob = as.numeric(.pred_prob[,1]))

nn_train_results <- nn_train_results |>
  mutate(.pred_prob = 1 - .pred_prob)

roc_auc(nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, .pred_prob) |> pull(.estimate)

#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------

# Predict probabilities (returns values between 0 and 1)
mimic_test_matrix <- mimic_test_ready[, colnames(mimic_train_x)]
mimic_test_matrix <- as.matrix(mimic_test_matrix)
nn_test_probs <- predict(nn_mimic_model, mimic_test_matrix)

# Bind with original test IDs
nn_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  mutate(HOSPITAL_EXPIRE_FLAG = nn_test_probs)

# Save to CSV
write.csv(nn_predictions, "Assignment 3/simple_nn_predictions_final.csv", row.names = FALSE)


################################################################################
################################################################################
####################### Train a Complex Neural Network #########################
################################################################################
################################################################################

#-------------------------------------------------------------------------------
#---------------------- Fit the Model on Training Data -------------------------
#-------------------------------------------------------------------------------

# Split the oversampled data into x and y
mimic_train_x <- mimic_train_ready |> select(-HOSPITAL_EXPIRE_FLAG)
mimic_train_y <- mimic_train_ready$HOSPITAL_EXPIRE_FLAG |> as.integer() - 1  

# Convert to matrix and numeric as needed
mimic_train_x <- as.matrix(mimic_train_x)
mimic_train_y <- as.numeric(mimic_train_y)

# Set the seed for reproducibility
library(keras)
tensorflow::set_random_seed(211)

# Define the feed-forward neural network model
complex_nn_mimic_model <- keras_model_sequential() |>
  layer_dense(units = 128, 
              kernel_regularizer = regularizer_l2(0.001), 
              input_shape = ncol(mimic_train_x)) |>
  layer_batch_normalization() |> 
  layer_activation("relu") |>
  layer_dropout(rate = 0.4) |>
  
  layer_dense(units = 64, 
              kernel_regularizer = regularizer_l2(0.001)) |>
  layer_batch_normalization() |>
  layer_activation("relu") |> 
  
  layer_dense(units = 1, activation = 'sigmoid')

# Show model summary
complex_nn_mimic_model |> summary()

# Compile the model
complex_nn_mimic_model |> compile(
  optimizer = "adam",
  loss      = "binary_crossentropy",
  metrics   = c('accuracy')
)

# Early stopping callback
early_stop_cb <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE  # restores the best model found
)

# Fit the model using the oversampled data
complex_nn_mimic_fit <- complex_nn_mimic_model |> 
  fit(
    x = mimic_train_x, 
    y = mimic_train_y,
    epochs = 75,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = list(early_stop_cb),
    verbose = 1
  )

# Determine the final Loss and Accuracy
complex_nn_mimic_model |>
  evaluate(mimic_train_x, mimic_train_y, verbose = 0)


#-------------------------------------------------------------------------------
#--------------------- Determine In-Sample Performance -------------------------
#-------------------------------------------------------------------------------

# Confusion Matrix

# Get predicted probabilities (between 0 and 1)
complex_nn_train_probs <- complex_nn_mimic_model |> predict(mimic_train_x)

# Convert probabilities to predicted class (threshold = 0.5)
complex_nn_train_preds <- ifelse(complex_nn_train_probs > 0.5, 1, 0)

# Combine with true labels
complex_nn_train_results <- tibble(
  HOSPITAL_EXPIRE_FLAG = mimic_train_y,
  .pred_class = complex_nn_train_preds,
  .pred_prob = complex_nn_train_probs
)

# Create a confusion matrix
complex_nn_cm <- complex_nn_train_results |>
  count(HOSPITAL_EXPIRE_FLAG, .pred_class) |>
  group_by(HOSPITAL_EXPIRE_FLAG) |>
  mutate(cl_acc = n[.pred_class == HOSPITAL_EXPIRE_FLAG] / sum(n)) |>
  pivot_wider(names_from = .pred_class, values_from = n) |>
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc)

# Output the confusion matrix
complex_nn_cm

# Determine Accuracy & Balanced Accuracy

# Convert true class to factor for yardstick
complex_nn_train_results <- complex_nn_train_results |>
  mutate(
    HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1)),
    .pred_class = factor(.pred_class, levels = c(0, 1))
  )

# Determine accuracy and balanced accuracy 
complex_nn_accuracy <- accuracy(complex_nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate
complex_nn_bal_accuracy <- bal_accuracy(complex_nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, estimate = .pred_class)$.estimate

# Determine AUC ROC
complex_nn_train_results <- complex_nn_train_results |>
  mutate(.pred_prob = as.numeric(.pred_prob[,1]))

complex_nn_train_results <- complex_nn_train_results |>
  mutate(.pred_prob = 1 - .pred_prob)

roc_auc(complex_nn_train_results, truth = HOSPITAL_EXPIRE_FLAG, .pred_prob) |> pull(.estimate)

#-------------------------------------------------------------------------------
#----------------------- Predict Out of Sample Values --------------------------
#-------------------------------------------------------------------------------


# Predict probabilities (returns values between 0 and 1)
mimic_test_matrix <- mimic_test_ready[, colnames(mimic_train_x)]
mimic_test_matrix <- as.matrix(mimic_test_matrix)
complex_nn_test_probs <- predict(complex_nn_mimic_model, mimic_test_matrix)

# Bind with original test IDs
complex_nn_predictions <- mimic_test_X |>
  select(icustay_id) |>
  rename(ID = icustay_id) |>
  mutate(HOSPITAL_EXPIRE_FLAG = complex_nn_test_probs)

# Save to CSV
write.csv(complex_nn_predictions, "Assignment 3/complex_nn_predictions_final.csv", row.names = FALSE)




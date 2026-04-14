# Network Anomaly Detection

## Project Overview
This project implements a comprehensive machine learning pipeline for detecting anomalous network traffic. It processes raw network data, handles class imbalances, and trains multiple supervised algorithms alongside K-Means clustering to classify network events accurately. 

## Pipeline Architecture
1. **Data Loading & Setup**: Integrates directly with Google Drive to load network logs seamlessly in a cloud environment.
2. **Exploratory Data Analysis (EDA)**: Generates dynamic visualizations including target class distributions, feature boxplots, and scatter plots to analyze network behavior.
3. **Robust Preprocessing**: 
   - Strict 70/15/15 Data Split (Train/Validation/Test).
   - Target encoding for binary classification.
   - One-Hot Encoding for categorical features to prevent mathematical hierarchies.
   - Standard scaling for continuous variables.
   - SMOTE (Synthetic Minority Over-sampling Technique) applied strictly within the cross-validation folds to handle class imbalance without data leakage.
4. **Supervised Learning**: Evaluates 11 distinct algorithms (including Random Forest, XGBoost, LightGBM, and Support Vector Machines) using 5-Fold Stratified Cross-Validation optimized for the F1-weighted metric.
5. **Hyperparameter Tuning**: Automatically applies RandomizedSearchCV to the top-performing baseline model to maximize predictive accuracy.
6. **Unsupervised Learning**: Utilizes K-Means clustering for baseline anomaly discovery and topological reality checks.
7. **Final Evaluation**: Tests the tuned model against the untouched 15% Test Vault, outputting a comprehensive 4-panel visual dashboard (Confusion Matrix, Classification Report Heatmap, ROC-AUC Curve, and Feature Importances).

## Prerequisites and Dependencies
The project is built to run in a Google Colab environment. The required libraries are installed automatically in the first cell of the notebook, including:
* pandas
* numpy
* scikit-learn
* imbalanced-learn
* xgboost
* lightgbm
* matplotlib
* seaborn

## Instructions to Run the Code

### Step 1: Prepare the Dataset
1. Download the [Network Anomaly Dataset](https://www.kaggle.com/datasets/amineipad/network-anoamly-dataset/data) from Kaggle.
2. Ensure the dataset is in CSV format.
3. Upload the CSV file to your Google Drive.

### Step 2: Setup Google Colab
1. Ensure your Colab runtime is utilizing a standard Python 3 environment.

### Step 3: Execution
1. Run the first cell to install all required dependencies.
2. Run the data loading cell. A prompt will appear asking for permission to mount your Google Drive. Accept the prompt and allow Colab to connect to your Drive.
3. Once the Drive is mounted, the script will automatically locate the CSV file in the folder.
4. Execute the remaining cells sequentially. 
5. During the Supervised Modeling step, allow several minutes for the 5-Fold Cross Validation and Hyperparameter Tuning processes to complete.

## Results Output
Upon completion, the notebook will output a detailed comparison table of all 11 models based on their Validation and Cross-Validation F1-Scores.

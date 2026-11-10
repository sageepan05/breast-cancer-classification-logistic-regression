A machine learning pipeline in Python that classifies breast tumours as benign or malignant from measurements of cell nuclei. Built with scikit-learn as an educational biomedical-engineering project.

Results

Evaluated on a held-out test set of 114 samples (20% of the data):

Metric	Value
Accuracy	96.49%
ROC AUC	0.996
Malignant recall	0.93 (39 of 42 caught)
Malignant precision	0.97

Confusion matrix: 71 benign and 39 malignant correctly classified, 1 false alarm, 3 missed malignant cases.

Confusion matrix	ROC curve
Confusion matrix	ROC curve

Dataset

Breast Cancer Wisconsin (Diagnostic) dataset, downloaded from Kaggle,
569 samples, 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses,
Target: Malignant (M) = 1 (212 samples), Benign (B) = 0 (357 samples),
Licence: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)



Pipeline

Load the CSV with Pandas and inspect it (info, head, describe)
Drop the id and empty Unnamed: 32 columns (no missing or duplicate values in the rest)
Encode the label: M = 1, B = 0, and plot the class balance
Split 80% train / 20% test with stratification
Scale features with StandardScaler (fitted on training data only to avoid data leakage)
Train a LogisticRegression model (max_iter=5000)
Evaluate with accuracy, classification report, confusion matrix, and ROC curve with AUC (using predict_proba)
Predict a single unseen sample with its probability of malignancy
Save the model and scaler with joblib

Why logistic regression? The data is tabular and nearly linearly separable, and the model is fast and interpretable. A CNN would suit raw images rather than pre-extracted features.

Running the script shows the plots and creates Breast_Cancer_Detection_model.pkl and scaler.pkl.

Limitations and future work

Small dataset (569 samples) and a single train/test split, so results can vary. Cross-validation would give a more reliable estimate.

Three malignant cases were missed in testing. In a medical setting, recall for the malignant class matters most, so threshold tuning would be worth exploring.

Try other models (random forest, SVM) and compare them, and add feature selection

Wrap the saved model in a simple web app for demonstration

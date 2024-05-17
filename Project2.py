#Data Mining Programming Assignment -2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load Data
try:
    df = pd.read_csv('nba2021.csv')
except FileNotFoundError:
    print("The file 'nba2021.csv' was not found in the current directory. Please add it to the same directory as this script.")

# Data Preprocessing
for col in ['3P%', 'FG%', 'FT%']:
    df[col] = df.apply(lambda row: 0 if row[col.replace('%', 'A')] == 0 else row[col], axis=1)

imputer = SimpleImputer(strategy='mean')
df.iloc[:, 4:] = imputer.fit_transform(df.iloc[:, 4:])

# Feature Selection
X = df.drop(['Player', 'Pos', 'Tm'], axis=1)
y = df['Pos']
X = X.loc[:, X.var() != 0]

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Addressing Class Imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=0)

# Feature Selection with RFECV and Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
selector = RFECV(forest, step=1, cv=StratifiedKFold(5))
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Model with SVC (SVM)
svm = SVC(random_state=0)

# Hyperparameter Tuning with Grid Search
parameters = {
    'C': np.logspace(-2, 2, 10),
    'kernel': ['linear']
}
clf = GridSearchCV(svm, parameters, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
clf.fit(X_train_selected, y_train)

# Best Parameters
print("Best parameters set found on development set:")
print(clf.best_params_)
best_svm = clf.best_estimator_

# Predictions and Accuracy
y_pred_train = best_svm.predict(X_train_selected)
y_pred_test = best_svm.predict(X_test_selected)
print('Training set accuracy:', accuracy_score(y_train, y_pred_train))
print('Test set accuracy:', accuracy_score(y_test, y_pred_test))

# Confusion matrix 
print("\nConfusion Matrix:\n")
unique_labels = np.unique(y)
cm = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
cm_df.loc['All', :] = cm_df.sum(axis=0)
cm_df.loc[:, 'All'] = cm_df.sum(axis=1)
print(cm_df)
# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

# Stratified K-Fold Cross-Validation with Best Estimator
skf = StratifiedKFold(n_splits=10)
fold_accuracies = []

for train_index, test_index in skf.split(X_res, y_res):
    X_train_fold, X_test_fold = X_res[train_index], X_res[test_index]
    y_train_fold, y_test_fold = y_res[train_index], y_res[test_index]

    X_train_fold_selected = selector.transform(X_train_fold)
    X_test_fold_selected = selector.transform(X_test_fold)

    best_svm.fit(X_train_fold_selected, y_train_fold)
    fold_accuracy = best_svm.score(X_test_fold_selected, y_test_fold)
    fold_accuracies.append(fold_accuracy)

print('Fold accuracies:', fold_accuracies)
print('Average accuracy:', np.mean(fold_accuracies))

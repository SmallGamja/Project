import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as impipeline
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

# Load the data into a Pandas DataFrame
data = pd.read_csv('data.csv')

# Scale the data 
X = data.drop('Bankrupt?', axis = 1)
y = data['Bankrupt?']

#Scaling the data (just in case)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dimension Reduction PCA
pca = PCA(n_components= 0.80).fit(X_scaled)
pca_var = pca.n_components_
print(pca_var)
# will use 34 components to capture at least 80% of the variance in our data
Xpca = PCA(n_components=0.80).fit_transform(X_scaled)

# Train Test Splitting with stratify and Stratified K Fold CV
X_train, X_test, y_train, y_test = train_test_split(Xpca, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)

"""
##################
SVC Classification
##################
"""

classifier = SVC()
#Establishing pipeline for SMOTE in 5 fold cross validation
pipeline = impipeline(steps = [['smote', smote], ['classifier', classifier]])

stratkfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)

#Performing Grid Search with SVC hyperparameter tuning 
param_grid = {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf'], 'classifier__gamma': [1, 0.1, 0.001, 'scale']}
grid_search = GridSearchCV(estimator=pipeline, param_grid = param_grid, cv=stratkfold, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
#saving our best performing svm and getting predictions
best_svm = grid_search.best_estimator_
test_score = grid_search.score(X_test, y_test)
print(f'Test Score:{test_score}')

#Printing Metrics 
svc_y_pred = best_svm.predict(X_test)
print("SVM")
print("ORIGINAL TEST SET")
print("Accuracy:", accuracy_score(y_test, svc_y_pred))
print("Precision:", precision_score(y_test, svc_y_pred))
print("Recall:", recall_score(y_test, svc_y_pred))
print("F1-score:", f1_score(y_test, svc_y_pred))
print("ROC AUC:", roc_auc_score(y_test, svc_y_pred))

# making confusion matrix for svm results
conf_mat = confusion_matrix(y_test, svc_y_pred)

# Plot the confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("svc_startkfold_smote_correct.png")
plt.close()

# Testing our classifier with balanced smote generated test data
#(This will confirm whether our classifier would perform well with better data)
X_test_balanced, y_test_balanced = smote.fit_resample(X_test, y_test)
svc_y_predbalanced = best_svm.predict(X_test_balanced)
print("METRICS FOR BALANCED TEST SET")
print("Accuracy:", accuracy_score(y_test_balanced, svc_y_predbalanced))
print("Precision:", precision_score(y_test_balanced, svc_y_predbalanced))
print("Recall:", recall_score(y_test_balanced, svc_y_predbalanced))
print("F1-score:", f1_score(y_test_balanced, svc_y_predbalanced))
print("ROC AUC:", roc_auc_score(y_test_balanced, svc_y_predbalanced))
print('---'*20)

# making confusion matrix for svm results
conf_mat_balanced = confusion_matrix(y_test_balanced, svc_y_predbalanced)

# Plot the confusion matrix
sns.heatmap(conf_mat_balanced, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Balanced')
plt.savefig("svc_stratkfold_smote_test_set_balanced.png")
plt.close()

"""
##################################
Logistic Regression Classification
##################################
"""
classifier = LogisticRegression(solver = 'saga')
#Establishing pipeline for SMOTE in 5 fold cross validation
pipeline = impipeline(steps = [['smote', smote], ['classifier', classifier]])

#Performing Grid Search with hyperparameter tuning 
log_reg_params = {"classifier__penalty": ['l2'],
                  'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator=pipeline, param_grid = log_reg_params, cv=stratkfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
#saving our best performing model and getting predictions
best_log = grid_search.best_estimator_
test_score = grid_search.score(X_test, y_test)
print(f'Test Score:{test_score}')

#Printing Metrics 
log_y_pred = best_log.predict(X_test)
print("Logistic Regression")
print("ORIGINAL TEST SET")
print("Accuracy:", accuracy_score(y_test, log_y_pred))
print("Precision:", precision_score(y_test, log_y_pred))
print("Recall:", recall_score(y_test, log_y_pred))
print("F1-score:", f1_score(y_test, log_y_pred))
print("ROC AUC:", roc_auc_score(y_test, log_y_pred))

# making confusion matrix for log_reg results
conf_mat2 = confusion_matrix(y_test, log_y_pred)

# Plot the confusion matrix
sns.heatmap(conf_mat2, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("log_reg_smote.png")
plt.close()

# Testing our classifier with balanced smote generated test data
#(This will confirm whether our classifier would perform well with better data)
log_y_predbalanced = best_log.predict(X_test_balanced)
print("METRICS FOR BALANCED TEST SET")
print("Accuracy:", accuracy_score(y_test_balanced, log_y_predbalanced))
print("Precision:", precision_score(y_test_balanced, log_y_predbalanced))
print("Recall:", recall_score(y_test_balanced, log_y_predbalanced))
print("F1-score:", f1_score(y_test_balanced, log_y_predbalanced))
print("ROC AUC:", roc_auc_score(y_test_balanced, log_y_predbalanced))
print('---'*20)

# making confusion matrix for log_reg results
conf_mat_balanced2 = confusion_matrix(y_test_balanced, log_y_predbalanced)

# Plot the confusion matrix
sns.heatmap(conf_mat_balanced2, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Balanced')
plt.savefig("log_reg_smote_balanced.png")
plt.close()

""""
#############################
Random Forest Classsification
#############################
"""
classifier = RandomForestClassifier(criterion = 'gini', random_state=42)
#Establishing pipeline for SMOTE in 5 fold cross validation
pipeline = impipeline(steps = [['smote', smote], ['classifier', classifier]])

#Performing Grid Search with hyperparameter tuning 
rf_params = {
    'classifier__max_depth':[2, 3, 4],
    'classifier__max_features':[2,4, 5, 6]}
grid_search = GridSearchCV(estimator=pipeline, param_grid = rf_params, cv=stratkfold, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
#saving our best performing model and getting predictions
best_rf = grid_search.best_estimator_
test_score = grid_search.score(X_test, y_test)
print(f'Test Score:{test_score}')

#Printing Metrics 
rf_y_pred = best_rf.predict(X_test)
print("Random Forest")
print("ORIGINAL TEST SET")
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Precision:", precision_score(y_test, rf_y_pred))
print("Recall:", recall_score(y_test, rf_y_pred))
print("F1-score:", f1_score(y_test, rf_y_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_y_pred))

# making confusion matrix for rf results
conf_mat3 = confusion_matrix(y_test, rf_y_pred)

# Plot the confusion matrix
sns.heatmap(conf_mat3, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("rf_smote.png")
plt.close()

# Testing our classifier with balanced smote generated test data
#(This will confirm whether our classifier would perform well with better data)
rf_y_predbalanced = best_rf.predict(X_test_balanced)
print("METRICS FOR BALANCED TEST SET")
print("Accuracy:", accuracy_score(y_test_balanced, rf_y_predbalanced))
print("Precision:", precision_score(y_test_balanced, rf_y_predbalanced))
print("Recall:", recall_score(y_test_balanced, rf_y_predbalanced))
print("F1-score:", f1_score(y_test_balanced, rf_y_predbalanced))
print("ROC AUC:", roc_auc_score(y_test_balanced, rf_y_predbalanced))
print('---'*20)

# making confusion matrix for rf results
conf_mat_balanced3 = confusion_matrix(y_test_balanced, rf_y_predbalanced)

# Plot the confusion matrix
sns.heatmap(conf_mat_balanced3, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Balanced')
plt.savefig("rf_smote_balanced.png")
plt.close()

"""
##################################
K-Nearest Neighbors Classification
##################################
"""
classifier = KNeighborsClassifier()
#Establishing pipeline for SMOTE in 5 fold cross validation
pipeline = impipeline(steps = [['smote', smote], ['classifier', classifier]])

#Performing Grid Search with hyperparameter tuning 
#Range is only odd numbers to avoid ties
k_range = list(range(9,71,2)) 
#note knn will keep choosing lowest k-value even after increasing range
#This makes it VERY prone to overfitting, so we go with k=9
knn_params = dict(classifier__n_neighbors = k_range)
grid_search = GridSearchCV(estimator=pipeline, param_grid = knn_params, cv=stratkfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
#saving our best performing model and getting predictions
best_knn = grid_search.best_estimator_
test_score = grid_search.score(X_test, y_test)
print(grid_search.best_params_)
print(f'Test Score:{test_score}')

#Printing Metrics 
knn_y_pred = best_knn.predict(X_test)
print("K-Nearest Neighbors")
print("ORIGINAL TEST SET")
print("Accuracy:", accuracy_score(y_test, knn_y_pred))
print("Precision:", precision_score(y_test, knn_y_pred))
print("Recall:", recall_score(y_test, knn_y_pred))
print("F1-score:", f1_score(y_test, knn_y_pred))
print("ROC AUC:", roc_auc_score(y_test, knn_y_pred))

# making confusion matrix for knn results
conf_mat4 = confusion_matrix(y_test, knn_y_pred)

# Plot the confusion matrix
sns.heatmap(conf_mat4, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("knn_smote.png")
plt.close()

# Testing our classifier with balanced smote generated test data
#(This will confirm whether our classifier would perform well with better data)
knn_y_predbalanced = best_knn.predict(X_test_balanced)
print("METRICS FOR BALANCED TEST SET")
print("Accuracy:", accuracy_score(y_test_balanced, knn_y_predbalanced))
print("Precision:", precision_score(y_test_balanced, knn_y_predbalanced))
print("Recall:", recall_score(y_test_balanced, knn_y_predbalanced))
print("F1-score:", f1_score(y_test_balanced, knn_y_predbalanced))
print("ROC AUC:", roc_auc_score(y_test_balanced, knn_y_predbalanced))

# making confusion matrix for svm results
conf_mat_balanced4 = confusion_matrix(y_test_balanced, knn_y_predbalanced)

# Plot the confusion matrix
sns.heatmap(conf_mat_balanced4, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Balanced')
plt.savefig("knn_smote_balanced.png")
plt.close()

"""
#####################
Generating ROC Curves
#####################
"""
# We will generate a figure to plot the ROC curves of the different classifiers 
# Using the predicted values and the false/true positive rates

# creating the curves for each classifier for unbalanced and balanced case

#unbalanced test set case
svm_fpr, svm_tpr, svm_thresh = roc_curve(y_test, svc_y_pred)
log_fpr, log_tpr, log_thresh = roc_curve(y_test, log_y_pred)
rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test, rf_y_pred)
knn_fpr, knn_tpr, knn_thresh = roc_curve(y_test, knn_y_pred)

plt.figure(figsize=(14,10))
plt.title('ROC Curve Unbalanced Test', fontsize=14)
plt.plot(svm_fpr, svm_tpr, label='SVC Classifier Score: {:.4f}'.format(roc_auc_score(y_test, svc_y_pred)))
plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test, log_y_pred)))
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test, rf_y_pred)))
plt.plot(knn_fpr, knn_tpr, label='K-Nearest-Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_test, knn_y_pred)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
plt.legend()
plt.savefig("ROC_curves.png")
plt.close()

#balanced test set case
balancedsvm_fpr, balancedsvm_tpr, balancedsvm_thresh = roc_curve(y_test_balanced, svc_y_predbalanced)
balancedlog_fpr, balancedlog_tpr, balancedlog_thresh = roc_curve(y_test_balanced, log_y_predbalanced)
balancedrf_fpr, balancedrf_tpr, balancedrf_thresh = roc_curve(y_test_balanced, rf_y_predbalanced)
balancedknn_fpr, balancedknn_tpr, balancedknn_thresh = roc_curve(y_test_balanced, knn_y_predbalanced)

plt.figure(figsize=(14,10))
plt.title('ROC Curve Balanced Test', fontsize=14)
plt.plot(balancedsvm_fpr, balancedsvm_tpr, label='SVC Classifier Score: {:.4f}'.format(roc_auc_score(y_test_balanced, svc_y_predbalanced)))
plt.plot(balancedlog_fpr, balancedlog_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test_balanced, log_y_predbalanced)))
plt.plot(balancedrf_fpr, balancedrf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test_balanced, rf_y_predbalanced)))
plt.plot(balancedknn_fpr, balancedknn_tpr, label='K-Nearest-Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_test_balanced, knn_y_predbalanced)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
plt.legend()
plt.savefig("ROC_curves_balanced.png")
plt.close()


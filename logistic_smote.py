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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

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
classifier = LogisticRegression()
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
plt.show()

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

# making confusion matrix for svm results
conf_mat_balanced = confusion_matrix(y_test_balanced, svc_y_predbalanced)

# Plot the confusion matrix
sns.heatmap(conf_mat_balanced, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Balanced')
plt.show()
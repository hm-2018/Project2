import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
df = pd.read_csv("heart.csv")
x = df.iloc[:, 0:12].values
y= df.iloc[: ,13].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
kfold=KFold(n_splits=5, shuffle=True, random_state=0)
linear_svc=SVC(kernel='linear')
linear_scores = cross_val_score(linear_svc, x, y, cv=kfold)
print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))
print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))
#Stratified k-Fold Cross Validation with shuffle split with rbf kernel
rbf_svc=SVC(kernel='rbf')
rbf_scores = cross_val_score(rbf_svc, x, y, cv=kfold)
print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))
print('Average stratified cross-validation score with rbf kernel:{:.4f}'.format(rbf_scores.mean()))
#========================================Cross validation==========================================================
svc=SVC()
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]}]
grid_search = GridSearchCV(estimator = svc, param_grid = parameters,scoring = 'accuracy', cv = 5, verbose=0)
grid_search.fit(X_train, y_train)
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
#========================result=====================================================
print("FFFFFFFFFFFFFFFFFFFFFFFFINALLLLLLLLLLLLLLLLLLLLLLLLLLLL")
linear_svc=SVC(kernel='linear' , C=10 )
linear_svc.fit(X_train,y_train)
y_pred=linear_svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))




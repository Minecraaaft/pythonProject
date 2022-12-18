# Download the Breast Cancer Dataset
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer;

cancer = load_breast_cancer();
X = cancer.data;
y = cancer.target;
#%%
# Divide the data into train (80%) and test (20%)
from sklearn.model_selection import train_test_split;

X_train , X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state = 0);

print(X_train.shape);
print(y_train.shape);
print("\r\n");
print(X_test.shape);
print(y_test.shape);
#%%
# Decision Tree Classifier* using all the features of the data. Model tested on the test data
from sklearn.tree import DecisionTreeClassifier;

tree = DecisionTreeClassifier(criterion    =  'entropy',
                              max_depth    =  3,
                              random_state =  0 );
tree.fit(X_train, y_train)
#%%
from sklearn.tree import plot_tree;

plot_tree(tree,
          feature_names = cancer.feature_names,
          fontsize      = 8 )
#%%
# see more about decision trees here:
# https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
#%%
from sklearn.metrics import confusion_matrix
y_true = y_train
y_pred = tree.predict(X_train)
print("Precision on training set")
confusion_matrix(y_true, y_pred)
#%%
y_true = y_test
y_pred = tree.predict(X_test)
print("Precision on test set")
confusion_matrix(y_true, y_pred)
#%%
# Exercise 1 : Improve precision on training set.
# Make relevant changes to the code above.
#%%
from sklearn.metrics import accuracy_score;
print("Train Set Accuracy : ", accuracy_score(y_train, tree.predict(X_train)))
y_pred_test = tree.predict(X_test);
print("Test Set Accuracy  : ", accuracy_score(y_test, tree.predict(X_test)))
#%%
# Gini statistics
#%%
# GINI IMPURITY
tree_gin_d1 = DecisionTreeClassifier(criterion    =  'gini',
                                     max_depth    =  1,
                                     random_state =  0 );
tree_gin_d1.fit(X_train, y_train)

y_pred_train_gin_d1 = tree_gin_d1.predict(X_train);
y_pred_test_gin_d1  = tree_gin_d1.predict(X_test);


# ENTROPY IMPURITY
tree_ent_d1 = DecisionTreeClassifier(criterion    =  'entropy',
                                     max_depth    =  1,
                                     random_state =  0 );
tree_ent_d1.fit(X_train, y_train)

y_pred_train_ent_d1 = tree_ent_d1.predict(X_train);
y_pred_test_ent_d1  = tree_ent_d1.predict(X_test);
#%%
# GINI IMPURITY
tree_gin_d6 = DecisionTreeClassifier(criterion    =  'gini',
                                     max_depth    =  6,
                                     random_state =  0 );
tree_gin_d6.fit(X_train, y_train)

y_pred_train_gin_d6 = tree_gin_d6.predict(X_train);
y_pred_test_gin_d6  = tree_gin_d6.predict(X_test);


# ENTROPY IMPURITY
tree_ent_d6 = DecisionTreeClassifier(criterion    =  'entropy',
                                     max_depth    =  6,
                                     random_state =  0 );
tree_ent_d6.fit(X_train, y_train)

y_pred_train_ent_d6 = tree_ent_d6.predict(X_train);
y_pred_test_ent_d6  = tree_ent_d6.predict(X_test);
#%%
print("\r\nDEPTH = 1");
print("\tGINI : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_gin_d1));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_gin_d1));
print("\tENTROPY : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_ent_d1));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_ent_d1));
#%%
print("\r\nDEPTH = 6");
print("\tGINI : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_gin_d6));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_gin_d6));
print("\tENTROPY : ");
print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, y_pred_train_ent_d6));
print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, y_pred_test_ent_d6));
#%%
# Exercise 2
# Make similar code as above for depth 2,3 and 7 (8)
#%%
print("Test Set Accuracy d = 2 : ", accuracy_score(y_test, y_pred_test_gin_d1));
print("Test Set Accuracy d = 3 : ", accuracy_score(y_test, y_pred_test_gin_d6));
#%%
# Exercise 3
# For which tree height (1 -7) do we find the highest accuracy for test set ?
#%%
from sklearn.svm import SVC;
svm = SVC(kernel = "rbf", C = 1.0, random_state = 0);
svm.fit(X_train, y_train);

Y_pred_train = svm.predict(X_train);
print("SVM Train Set Accuracy : ", accuracy_score(y_train, Y_pred_train));

Y_pred_test = svm.predict(X_test);
print("SVM Test Set Accuracy  : ", accuracy_score(y_test, Y_pred_test));
#%%
# Exercise 4
# Improve the accuracy of the SVM classifier.
#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l2', solver = "lbfgs", C= 1.0, random_state = 0, max_iter=100000);
lr.fit(X_train, y_train);

Y_pred_train = lr.predict(X_train);
print("LR Train Set Accuracy : ", accuracy_score(y_train, Y_pred_train));

Y_pred_test = lr.predict(X_test);
print("LR Test Set Accuracy  : ", accuracy_score(y_test, Y_pred_test));
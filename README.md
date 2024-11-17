# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Load Dataset: Load the Iris dataset using load_iris() and convert it to a DataFrame.

2 Preprocess Data: Separate the features (X) and target (y) from the DataFrame.

3 Split Data: Split the data into training and test sets (80%-20%) using train_test_split().

4 Train Model: Initialize and train an SGDClassifier on the training data (X_train, y_train).

5 Make Predictions: Predict the target values for the test set (X_test) using the trained model.

6 Evaluate Model: Calculate and print the accuracy score and confusion matrix to evaluate the model's performance.
## Program:

/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S.PARTHASARATHI
RegisterNumber:  212223040144
*/
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
![367656537-278ac232-c82e-451b-80f5-24c487bdb483](https://github.com/user-attachments/assets/8ba273af-5c85-4ef6-9076-e8e161551212)
## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

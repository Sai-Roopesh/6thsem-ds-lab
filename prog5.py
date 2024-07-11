import numpy as np  
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 

# Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Define the hyperparameters to try 
kernels = ['rbf'] 
gammas = [0.5] 
Cs = [0.01, 1, 10] 

best_accuracy = 0 
best_parameters = None 
best_support_vectors = None 

# Iterate over different combinations of hyperparameters 
for kernel in kernels: 
    for gamma in gammas: 
        for C in Cs: 
            # Train the SVM classifier 
            svm_clf = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape='ovr') 
            svm_clf.fit(X_train, y_train) 

            # Predict on the test set 
            y_pred = svm_clf.predict(X_test) 

            # Calculate classification accuracy 
            accuracy = accuracy_score(y_test, y_pred) 

            if accuracy > best_accuracy: 
                best_accuracy = accuracy 
                best_parameters = (kernel, gamma, C) 
                best_support_vectors = svm_clf.support_vectors_ 

print("Best Classification Accuracy:", best_accuracy) 
print("Best Parameters (kernel, gamma, C):", best_parameters) 
print("Number of Support Vectors:", len(best_support_vectors))

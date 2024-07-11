from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression classifier with regularization (L2 penalty) and C=1e4
log_reg = LogisticRegression(C=1e4, penalty='l2', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Classification Accuracy:", accuracy)

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Define the dataset
data = [
    ['Low', 'Low', 2, 'No', 'Yes'],
    ['Low', 'Med', 4, 'Yes', 'Yes'],
    ['Low', 'Low', 4, 'No', 'Yes'],
    ['Low', 'Med', 4, 'No', 'No'],
    ['Low', 'High', 4, 'No', 'No'],
    ['Med', 'Med', 4, 'No', 'No'],
    ['Med', 'Med', 4, 'Yes', 'Yes'],
    ['Med', 'High', 2, 'Yes', 'No'],
    ['Med', 'High', 5, 'No', 'Yes'],
    ['High', 'Med', 4, 'Yes', 'Yes'],
    ['High', 'Med', 2, 'Yes', 'Yes'],
    ['High', 'High', 2, 'Yes', 'No'],
    ['High', 'High', 5, 'Yes', 'Yes']
]

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['Price', 'Maintenance', 'Capacity', 'Airbag', 'Profitable'])

# Encode categorical features
label_encoders = {}
for column in ['Price', 'Maintenance', 'Airbag', 'Profitable']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target variable
X = df.drop(columns=['Profitable'])
y = df['Profitable']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')  # ID3 algorithm uses information gain (entropy) for splitting
clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = clf.predict(X_test)

# Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)

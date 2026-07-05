import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("dataset/health.csv")

# Convert HDI text labels to numbers
le = LabelEncoder()
data['HDI'] = le.fit_transform(data['HDI'])

# Input columns
X=data[['LifeExpectancy',
        'MeanSchooling',
        'ExpectedSchooling',
        'GNI']]

y=data['HDI']
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model Saved Successfully")
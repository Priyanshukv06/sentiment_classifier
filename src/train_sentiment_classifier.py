import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('data/imdb_reviews.csv')

# Load sentence vectors (generated from Word2Vec)
sentence_vectors = pd.read_csv('data/sentence_vectors.csv')

# Encode the sentiment labels (positive and negative) into numerical values
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

# Split the data into training and testing sets
X = sentence_vectors
y = df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the trained model to a file
import joblib
joblib.dump(model, 'sentiment_classifier_model.pkl')
print("Model saved to sentiment_classifier_model.pkl")

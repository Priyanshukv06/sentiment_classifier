import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec  # Import Word2Vec

# Load the trained sentiment classifier
model = joblib.load('sentiment_classifier_model.pkl')

# Load the trained Word2Vec model (you should have saved this model during training)
word2vec_model = Word2Vec.load('word2vec_model.model')  # Assuming it's saved with this name

# Load the test data
df_test = pd.read_csv(r'C:\Users\priya\sentiment_classification\data\imdb_reviews.csv')

# Preprocess the test data (tokenization and vector generation)
def get_average_word2vec(sentence, model):
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model.wv:  # Use the Word2Vec model's word vectors
            word_vectors.append(model.wv[word])
    if len(word_vectors) == 0:
        return [0] * model.vector_size  # Return a zero vector if no words are found
    return sum(word_vectors) / len(word_vectors)

# Generate sentence vectors for the test data
X_test_vectors = df_test['review'].apply(lambda x: get_average_word2vec(x, word2vec_model))

# Load sentiment labels for test data
le = LabelEncoder()
df_test['sentiment_encoded'] = le.fit_transform(df_test['sentiment'])

# Convert to array for prediction
X_test_vectors = list(X_test_vectors)
y_test = df_test['sentiment_encoded']

# Make predictions on the test data
y_pred = model.predict(X_test_vectors)

# Evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

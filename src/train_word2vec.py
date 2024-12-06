import gensim
from gensim.models import Word2Vec
import nltk
import pandas as pd

# Load preprocessed data (CSV file)
df = pd.read_csv('data/imdb_reviews.csv')

# Tokenize reviews (already tokenized during data preparation)
reviews = df['review'].apply(lambda x: x.split())

# Train Word2Vec model
model = Word2Vec(sentences=reviews, vector_size=100, window=5, min_count=1, workers=4)

# Save the model to file
model.save("word2vec_model.model")
print("Word2Vec model trained and saved successfully!")

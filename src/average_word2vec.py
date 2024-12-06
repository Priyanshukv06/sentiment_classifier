import gensim
import pandas as pd
import numpy as np

# Load the pre-trained Word2Vec model
model = gensim.models.Word2Vec.load("word2vec_model.model")

# Load the data (CSV file with tokenized reviews)
df = pd.read_csv('data/imdb_reviews.csv')

# Function to compute average word2vec for a sentence
def get_average_word2vec(sentence, model):
    # Tokenize the sentence into words
    words = sentence.split()
    
    # Get vectors for each word in the sentence
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])

    # If the sentence contains no words in the vocabulary, return a zero vector
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)

    # Average the word vectors
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector

# Generate sentence vectors using the average Word2Vec vectors
sentence_vectors = df['review'].apply(lambda x: get_average_word2vec(x, model))

# Convert sentence vectors to a DataFrame
sentence_vectors_df = pd.DataFrame(list(sentence_vectors))

# Save the sentence vectors to a CSV file
sentence_vectors_df.to_csv('data/sentence_vectors.csv', index=False)

print("Sentence vectors generated and saved to 'data/sentence_vectors.csv'")

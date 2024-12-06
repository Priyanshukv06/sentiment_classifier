import nltk
from nltk.corpus import movie_reviews
import random
import pandas as pd
import os

# Download dataset
nltk.download('movie_reviews')
nltk.download('punkt')

def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    
    data = []
    for words, sentiment in documents:
        data.append((' '.join(words), sentiment))
        
    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    return df

if __name__ == "__main__":
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')  # Create the directory if it doesn't exist

    print("Loading data...")  # Debug print
    df = load_data()
    
    # Check if data is loaded
    print(f"Data loaded: {len(df)} rows")  # Debug print
    
    # Save the file to 'data' directory
    try:
        print("Saving CSV file...")  # Debug print
        df.to_csv('data/imdb_reviews.csv', index=False)
        print("CSV file saved successfully")  # Debug print
    except Exception as e:
        print(f"Error: {e}")  # Debug print

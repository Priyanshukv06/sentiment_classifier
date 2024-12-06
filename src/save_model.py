import joblib
from joblib import dump

model = joblib.load('sentiment_classifier_model.pkl')

dump(model, 'C:/Users/priya/sentiment_classification/sentiment_classifier.joblib')  # Replace with your path


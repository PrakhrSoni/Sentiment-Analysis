from flask import Flask, request, render_template
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import re
from nltk.corpus import stopwords
# Load your trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')  # Load the saved model (Logistic Regression)
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the saved TF-IDF vectorizer

# Initialize stopwords
stop_words = stopwords.words('english')

# Flask app initialization
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    # Get the text from the form
    text1 = request.form['text1'].lower()  # Convert to lowercase
    
    # Preprocess the text
    text_final = ''.join(c for c in text1 if not c.isdigit())  # Remove digits
    # text_final = ''.join(c for c in text_final if c not in punctuation)  # Remove punctuation
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])  # Remove stopwords
    
    # Transform the input text using the TF-IDF vectorizer
    text_vectorized = vectorizer.transform([processed_doc1])
    
    # Get predictions from the trained model
    sentiment_prediction = model.predict(text_vectorized)
    sentiment_probability = model.predict_proba(text_vectorized)

    # Convert the prediction to a human-readable sentiment (for example: 'Positive', 'Neutral', 'Negative')
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[sentiment_prediction[0]]

    # Optionally, use Vader Sentiment for extra analysis 
    sa = SentimentIntensityAnalyzer()
    vader_scores = sa.polarity_scores(processed_doc1)
    compound_score = round((1 + vader_scores['compound']) / 2, 2)  # Normalize compound score between 0 and 1
    
    # Return the result to the frontend
    return render_template('form.html', final=compound_score, text1=text_final, 
                           sentiment=predicted_sentiment, sentiment_prob=sentiment_probability[0])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
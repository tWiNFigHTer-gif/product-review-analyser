import pandas as pd
import nltk
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Function to collect data via web scraping
def collect_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text1 = []
    
    # Assuming reviews are in <p> tags; adjust as necessary
    for review in soup.find_all('p'):
        text1.append(review.get_text())
    
    return text1

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to analyze sentiment
def analyze_sentiment(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to classify issues
def classify_issue(text):
    keywords = {
        'performance': ['performance', 'speed', 'acceleration'],
        'comfort': ['comfort', 'seat', 'ride'],
        'safety': ['safety', 'hazard', 'risk'],
        'maintenance': ['maintenance', 'repair', 'service'],
    }
    for category, words in keywords.items():
        if any(word in text for word in words):
            return category
    return 'other'

# Endpoint to analyze reviews
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    reviews = pd.DataFrame(data)

    # Preprocess and analyze
    reviews['cleaned_text'] = reviews['text'].apply(preprocess_text)
    reviews['sentiment'] = reviews['cleaned_text'].apply(analyze_sentiment)
    reviews['issue_category'] = reviews['cleaned_text'].apply(classify_issue)

    # Generate insights
    insights = {
        'average_sentiment': reviews['sentiment'].mean(),
        'issue_counts': reviews['issue_category'].value_counts().to_dict(),
    }

    # Visualize trends
    plt.figure(figsize=(10, 5))
    reviews['issue_category'].value_counts().plot(kind='bar')
    plt.title('Issue Categories from Reviews')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.savefig('issue_trends.png')

    return jsonify(insights)

# Endpoint to scrape and analyze reviews from a URL
@app.route('/scrape', methods=['POST'])
def scrape_and_analyze():
    url = request.json['url']
    reviews = collect_data(url)
    reviews_df = pd.DataFrame({'text': reviews})
    return analyze(reviews_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

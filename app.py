from flask import Flask, jsonify
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import random
import os

app = Flask(__name__)

# Define the RSS feed URLs
rss_urls = [
    "https://www.darkreading.com/rss.xml",
    "https://feeds.feedburner.com/TheHackersNews",
    "https://cybersecurityventures.com/feed/",
]

# Function to parse RSS feeds
def parse_rss_feed(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    items = []
    for entry in soup.find_all('item'):
        item = {
            'title': entry.title.text if entry.title else "No title",
            'link': entry.link.text if entry.link else "No link",
            'description': entry.description.text if entry.description else "No description",
            'published': entry.pubDate.text if entry.pubDate else "No date"
        }
        items.append(item)
    return items

# Fetch and prepare the data
def fetch_and_prepare_data():
    all_items = []
    for url in rss_urls:
        items = parse_rss_feed(url)
        all_items.extend(items)
    
    return all_items

# Dummy dataset creation for the model training (random labels for demonstration)
def create_dummy_data(items):
    texts = [item['title'] + ' ' + item['description'] for item in items]
    labels = [random.choice([0, 1, 2]) for _ in range(len(items))]  # 0: green, 1: orange, 2: red
    return texts, labels

# Train the model
def train_model(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

# Function to classify a new article
def classify_article(model, title, description):
    text = title + ' ' + description
    prediction = model.predict([text])[0]
    return prediction

# Fetch and classify the data
@app.route("/feed.json", methods=["GET"])
def get_classified_data():
    data = fetch_and_prepare_data()
    texts, labels = create_dummy_data(data)  # Using dummy labels for demonstration
    model = train_model(texts, labels)  # Train the ML model

    classified_data = []
    for item in data:
        title = item['title']
        description = item['description']
        label = classify_article(model, title, description)
        classified_data.append({
            'title': title,
            'description': description,
            'critical_level': ['green', 'orange', 'red'][label]
        })
    
    return jsonify(classified_data)

if __name__ == "__main__":
    # Use the PORT environment variable set by Render or default to 8000
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

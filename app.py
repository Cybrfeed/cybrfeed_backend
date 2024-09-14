from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import feedparser
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Define CORS allowed origins
allowed_origins = [
    "http://localhost:5173",
    "https://cybersec-feed-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RSSItem(BaseModel):
    title: str
    link: str
    description: str
    published: str

# Define the RSS feed URLs
rss_urls = [
    "https://www.darkreading.com/rss.xml",
    "https://feeds.feedburner.com/TheHackersNews",
    "https://cybersecurityventures.com/feed/",
]

# Function to parse RSS feeds
def parse_rss_feed(url: str) -> List[RSSItem]:
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        item = RSSItem(
            title=entry.get("title", "No title"),
            link=entry.get("link", "No link"),
            description=entry.get("description", "No description"),
            published=entry.get("published", "No date"),
        )
        items.append(item)
    return items

# Fetch and prepare the data
def fetch_and_prepare_data():
    all_items = []
    for url in rss_urls:
        items = parse_rss_feed(url)
        all_items.extend(items)
    
    df = pd.DataFrame([item.dict() for item in all_items])
    df = df[['title', 'description']]
    
    # Dummy labels for demonstration (replace with real labels in practice)
    labels = np.random.randint(0, 3, size=len(df))
    df['label'] = labels
    
    return df

# Prepare the classification model
df = fetch_and_prepare_data()
X_train, X_test, y_train, y_test = train_test_split(df[['title', 'description']], df['label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train['title'] + ' ' + X_train['description'], y_train)
y_pred = pipeline.predict(X_test['title'] + ' ' + X_test['description'])
print(classification_report(y_test, y_pred, target_names=['green', 'orange', 'red']))

# Classify a new article
def classify_article(title, description):
    text = title + ' ' + description
    prediction = pipeline.predict([text])[0]
    return prediction

@app.get("/feed.json")
def get_classified_data():
    classified_data = []
    for _, row in df.iterrows():
        title = row['title']
        description = row['description']
        label = classify_article(title, description)
        classified_data.append({
            'title': title,
            'description': description,
            'critical_level': ['green', 'orange', 'red'][label]
        })
    
    return JSONResponse(content=classified_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

from fastapi import FastAPI
import feedparser
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class RSSItem(BaseModel):
    title: str
    link: str
    description: str
    published: str


allowed_origins = [
    "http://127.0.0.1:8000",
    "https://cybersec-feed-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_rss_feed(url: str) -> List[RSSItem]:
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        item = RSSItem(
            title=entry.get("title", "No title"),
            link=entry.get("link", "No link"),
            description=(
                entry.get("description", "No description")
                if "description" in entry
                else "No description"
            ),
            published=entry.get("published", "No date"),
        )
        items.append(item)
    return items


@app.get("/rss", response_model=List[RSSItem])
def get_rss_feeds():
    urls = [
        "https://www.darkreading.com/rss.xml",
        "https://feeds.feedburner.com/TheHackersNews",
        "https://cybersecurityventures.com/feed/",
    ]
    all_items = []
    for url in urls:
        items = parse_rss_feed(url)
        all_items.extend(items)
    return all_items


if __name__ == "__main__":
    uvicorn.run(app, port=8000)

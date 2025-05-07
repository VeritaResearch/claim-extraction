import re
from tqdm.auto import tqdm
import requests
import time
import tempfile
import json
from urllib.parse import urlparse, urljoin
from trafilatura import extract, fetch_url
from bs4 import BeautifulSoup
from atlas.document_processing import split_sentences, pdf_to_text

SOURCES = {"congressional record": "https://api.congress.gov/v3/"}


def get_congressional_statements(api_key: str) -> list:
    """
    Fetches the most recent congressional record from the official website.
    """
    url = SOURCES["congressional record"]

    records = requests.get(
        urljoin(url, "daily-congressional-record"), params={"api_key": api_key}
    ).content
    records = json.loads(records)["dailyCongressionalRecord"]

    ignore_titles = [
        "PLEDGE OF ALLEGIANCE;",
        "PRAYER;",
        "ADJOURNMENT",
        "ADDITIONAL SPONSORS",
    ]
    all_articles = []
    for record in records:
        record_url = urljoin(record["url"].split("?")[0] + "/", "articles")
        record = None
        while record is None:
            record = requests.get(
                record_url,
                params={
                    "api_key": api_key,
                    "offset": 0,
                    "limit": 250,
                },
            ).content
            if record is None:
                print("Error fetching record, retrying...")
                time.sleep(5)
        record = json.loads(record)["articles"]
        articles = [
            {
                "url": {source["type"]: source["url"] for source in article["text"]}[
                    "Formatted Text"
                ],
                "title": article["title"],
            }
            for section in record
            for article in section["sectionArticles"]
        ]
        articles = [
            article
            for article in articles
            if not any(article["title"].startswith(header) for header in ignore_titles)
        ]
        all_articles.extend(articles)

    for article in tqdm(all_articles):
        text = BeautifulSoup(fetch_url(article["url"])).text
        sentences = [
            re.sub(" +", " ", sentence)
            for paragraph in text.split("\n\n")
            if paragraph.strip()
            for sentence in split_sentences(
                paragraph.replace("\n", " ").strip(), min_words=1
            )
        ]
        article["text"] = text
        article["sentences"] = sentences

    return all_articles

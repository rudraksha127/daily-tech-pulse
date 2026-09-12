import os
import re
import html
import datetime as dt
from typing import List, Dict
import feedparser
from openai import OpenAI

TODAY_UTC = dt.datetime.utcnow().strftime("%Y-%m-%d")

FEEDS = [
    # Tech news
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    # AI era / industry
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/discover/blog/rss.xml",
    # Research papers
    "http://export.arxiv.org/rss/cs.AI",
    "http://export.arxiv.org/rss/cs.LG",
]

MAX_ITEMS_TOTAL = 24
MAX_ITEMS_PER_FEED = 5


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)  # remove HTML tags
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fetch_feed(url: str, limit: int = MAX_ITEMS_PER_FEED) -> List[Dict[str, str]]:
    parsed = feedparser.parse(url)
    items = []
    entries = getattr(parsed, "entries", []) or []
    for e in entries[:limit]:
        title = clean_text(e.get("title", ""))
        link = (e.get("link", "") or "").strip()
        summary = clean_text(e.get("summary", "") or e.get("description", ""))
        if title and link:
            items.append({"title": title, "link": link, "summary": summary})
    return items


def collect_items() -> List[Dict[str, str]]:
    collected = []
    for feed_url in FEEDS:
        try:
            collected.extend(fetch_feed(feed_url))
        except Exception:
            # graceful skip for broken/unavailable feed
            continue

    # dedupe by link
    seen = set()
    unique = []
    for it in collected:
        if it["link"] not in seen:
            seen.add(it["link"])
            unique.append(it)

    return unique[:MAX_ITEMS_TOTAL]


def build_prompt(items: List[Dict[str, str]]) -> str:
    source_lines = []
    for i, it in enumerate(items, 1):
        source_lines.append(
            f"{i}. {it['title']}\nURL: {it['link']}\nSnippet: {it['summary'][:280]}"
        )

    sources_blob = "\n\n".join(source_lines)

    return f"""
You are an accurate and concise tech editor.

Task: Write a daily Markdown brief in Hinglish (Hindi + English), based ONLY on the provided sources.
Date (UTC): {TODAY_UTC}

Provided Sources:
{sources_blob}

Output format exactly:

# Daily Tech Pulse — {TODAY_UTC}

## Top 5 Updates
- ...

## AI Era Trends
- ...

## Research Spotlight
- Pick one notable research item and explain simply for developers.

## Why It Matters
- Short practical takeaway for builders/founders/devs.

## Sources
- [Exact Source Title](Exact URL)

Hard rules:
1) Do NOT invent facts, companies, numbers, or announcements.
2) If uncertain, keep wording cautious.
3) Use only provided URLs in Sources.
4) Keep total length compact (~350-700 words).
5) Keep it readable, insightful, and practical.
""".strip()


def generate_article(items: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(items)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You produce faithful daily tech summaries from source lists."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Model returned empty content")
    return content


def fallback_article(items: List[Dict[str, str]]) -> str:
    lines = [f"# Daily Tech Pulse — {TODAY_UTC}", "", "Automation fallback post.", ""]
    if not items:
        lines.append("No reliable feed items could be fetched today.")
        return "\n".join(lines) + "\n"

    lines.append("## Fetched Sources")
    for it in items[:10]:
        lines.append(f"- [{it['title']}]({it['link']})")
    lines.append("")
    lines.append("AI draft generation failed this run; source links are posted for continuity.")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs("posts", exist_ok=True)
    items = collect_items()
    outfile = f"posts/{TODAY_UTC}.md"

    if os.path.exists(outfile):
        # Prevent duplicate daily overwrite churn
        print(f"{outfile} already exists. Skipping generation.")
        return

    if not items:
        content = fallback_article(items)
    else:
        try:
            content = generate_article(items)
        except Exception:
            content = fallback_article(items)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    print(f"Created: {outfile}")


if __name__ == "__main__":
    main()

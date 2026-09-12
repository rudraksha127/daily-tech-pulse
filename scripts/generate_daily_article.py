import os
import re
import html
import datetime as dt
from typing import List, Dict
import feedparser
from openai import OpenAI

TODAY_UTC = dt.datetime.utcnow().strftime("%Y-%m-%d")

FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/discover/blog/rss.xml",
    "http://export.arxiv.org/rss/cs.AI",
    "http://export.arxiv.org/rss/cs.LG",
]

MAX_ITEMS_TOTAL = 30
MAX_ITEMS_PER_FEED = 6


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:80] if text else "daily-tech-article"


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
            continue

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
            f"{i}. TITLE: {it['title']}\nURL: {it['link']}\nSNIPPET: {it['summary'][:320]}"
        )
    sources_blob = "\n\n".join(source_lines)

    return f"""
You are writing an ORIGINAL long-form tech article for a personal GitHub blog.

Date (UTC): {TODAY_UTC}
Author name: Rudraksha

Use the sources only for factual grounding and trend awareness.
Do NOT output source links in the article body.
Do NOT write like a news roundup.
Write like a human author with a clear personal narrative voice.

INPUT SOURCES:
{sources_blob}

Output requirements:
1) Return valid markdown only.
2) Start with a strong H1 title.
3) 700-1200 words.
4) Hinglish style (natural Hindi + English mix), professional and readable.
5) Structure:
   - Hook/Intro
   - Main discussion (2-4 sections with H2/H3)
   - Practical implications for developers/builders
   - Conclusion with personal take
6) Add 1 small code block or pseudo-example where relevant.
7) No bullet spam; mostly paragraph style.
8) No external URLs in body text.
9) Keep factual claims conservative; no fabricated announcements.
10) At the end, include a short hidden metadata block:
<!--
sources_used:
- title | url
- ...
-->
Use max 8 most relevant sources in metadata block.
""".strip()


def generate_article(items: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(items)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.6,
        messages=[
            {"role": "system", "content": "You write authentic human-like long-form tech essays with factual discipline."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Model returned empty content")
    return content


def fallback_article(items: List[Dict[str, str]]) -> str:
    title = f"AI, Tech aur Research ka Aaj ka Real Signal ({TODAY_UTC})"
    lines = [
        f"# {title}",
        "",
        "Aaj automated long-form generation issue hua, lekin momentum break nahi karte.",
        "Is post ko short manual placeholder ke form me save kiya ja raha hai.",
        "",
        "Kal se regular long-form article continue rahega.",
        "",
        "<!--",
        "sources_used:",
    ]
    for it in items[:8]:
        lines.append(f"- {it['title']} | {it['link']}")
    lines += ["-->",""]
    return "\n".join(lines)


def extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.strip().startswith("# "):
            return line.strip()[2:].strip()
    return f"Daily Tech Essay {TODAY_UTC}"


def main():
    os.makedirs("posts", exist_ok=True)
    items = collect_items()

    if items:
        try:
            article = generate_article(items)
        except Exception:
            article = fallback_article(items)
    else:
        article = fallback_article([])

    title = extract_title(article)
    slug = slugify(title)
    outfile = f"posts/{TODAY_UTC}-{slug}.md"

    # same day duplicate avoid
    existing_same_day = [f for f in os.listdir("posts") if f.startswith(f"{TODAY_UTC}-") and f.endswith(".md")]
    if existing_same_day:
        print(f"Post for {TODAY_UTC} already exists: {existing_same_day[0]}")
        return

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(article.strip() + "\n")

    print(f"Created: {outfile}")


if __name__ == "__main__":
    main()

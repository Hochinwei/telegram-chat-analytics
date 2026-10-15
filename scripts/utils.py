# ============================================================
# SHARED UTILITIES — telegram-chat-analytics
# Extracted from group and private chat scripts to follow DRY.
# Both scripts import from here so any parsing bug only needs
# fixing in one place.
# ============================================================

import glob
import os
import re
import warnings
from datetime import datetime

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ════════════════════════════════════════════════════════════
# STOPWORDS
# Standard English stopwords + Singlish / informal terms
# ════════════════════════════════════════════════════════════

STOP = set(stopwords.words("english"))
STOP.update([
    # Common informal abbreviations
    "u", "ur", "im", "dont", "gonna", "wanna", "ok", "okay",
    "ya", "yea", "yeah", "nah", "na", "hi", "haha", "hahaha",
    "lol", "omg", "wtf", "kk", "ha",
    # Common but low-information words
    "like", "get", "got", "know", "go", "one", "also",
])

# ════════════════════════════════════════════════════════════
# TEXT CLEANING
# ════════════════════════════════════════════════════════════

def clean_text(text):
    """Lowercase, strip URLs, mentions, punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_tokens(text):
    """Tokenize, remove stopwords, and return clean word list."""
    tokens = word_tokenize(clean_text(text))
    return [t for t in tokens if t.isalpha() and t not in STOP and len(t) > 1]

# ════════════════════════════════════════════════════════════
# HTML PARSING
# ════════════════════════════════════════════════════════════

def parse_single_file(filepath):
    """Parse one Telegram export HTML file into a list of message records."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    records = []
    current_sender = None

    for div in soup.find_all("div", class_="message"):
        classes = div.get("class", [])
        if "service" in classes or "default" not in classes:
            continue

        body = div.find("div", class_="body")
        if not body:
            continue

        # Sender name is only printed on the first message in a consecutive block
        from_tag = body.find("div", class_="from_name")
        if from_tag:
            current_sender = from_tag.get_text(strip=True)

        # Timestamp is stored in the 'title' attribute of the date div
        date_tag = body.find("div", class_=lambda c: c and "date" in c)
        date_str = date_tag["title"] if date_tag and date_tag.has_attr("title") else None
        timestamp = None
        if date_str:
            try:
                timestamp = datetime.strptime(
                    re.sub(r"\s*UTC.*", "", date_str).strip(), "%d.%m.%Y %H:%M:%S"
                )
            except Exception:
                pass

        text_tag = body.find("div", class_="text")
        text = text_tag.get_text(separator=" ", strip=True) if text_tag else None
        has_media = bool(body.find("div", class_="media_wrap"))

        if current_sender and text and timestamp:
            records.append({
                "sender":    current_sender,
                "text":      text,
                "timestamp": timestamp,
                "has_media": has_media,
                "msg_len":   len(text),
            })

    return records


def parse_all_files(pattern, timezone_offset=0):
    """Combine all matching HTML files into one clean, sorted DataFrame.

    Args:
        pattern:          Glob string, e.g. '../data/raw/messages*.html'
        timezone_offset:  Integer hours to shift timestamps from UTC (e.g. 8 for SGT).
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found matching: {pattern}\n"
            "Check that --input points to the correct folder."
        )

    print(f"  Found {len(files)} file(s):")
    all_records = []
    for fp in files:
        records = parse_single_file(fp)
        print(f"  {os.path.basename(fp):25s} → {len(records)} messages")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["sender", "timestamp", "text"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Shift from UTC to local timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=timezone_offset)

    # Derived time columns used throughout the analysis
    df["date"]    = df["timestamp"].dt.date
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month"]   = df["timestamp"].dt.to_period("M")
    df["year"]    = df["timestamp"].dt.year
    return df

# ════════════════════════════════════════════════════════════
# COLOUR PALETTE
# Assigns a consistent, distinct colour to each sender so the
# same person always appears in the same colour across charts.
# ════════════════════════════════════════════════════════════

PALETTE = [
    "#01696f", "#da7101", "#7a39bb", "#a12c7b", "#437a22",
    "#006494", "#a13544", "#d19900", "#4f98a3", "#bb653b",
    "#6daa45", "#d163a7", "#dd6974", "#5591c7", "#a86fdf",
    "#e8af34", "#fdab43", "#4a90d9", "#e05c5c", "#52b788",
    "#f4a261", "#2a9d8f", "#e76f51", "#264653", "#8ecae6",
    "#023047", "#ffb703", "#fb8500", "#c77dff", "#48cae4",
]


def sender_colors(senders):
    """Return a dict mapping each sender name to a unique hex colour."""
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(senders))}

# ════════════════════════════════════════════════════════════
# ANALYTICS LOGIC
# ════════════════════════════════════════════════════════════

def compute_response_times(df, max_gap_minutes=1440, return_excluded=False):
    """
    Return a DataFrame with columns: responder, response_min, and optionally month.

    A response is recorded from the LAST message of the previous sender to the FIRST
    message of the new sender. Gaps >= max_gap_minutes are considered new
    conversations, not replies, and are capped.
    """
    cols = ["responder", "response_min"]
    if "month" in df.columns:
        cols.append("month")

    if df.empty or len(df) <= 1:
        empty_df = pd.DataFrame(columns=cols)
        if return_excluded:
            return empty_df, 0
        return empty_df

    # Sort using stable sort to keep ties in original order without mutating input
    df_s = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Vectorized shift
    prev_sender = df_s["sender"].shift(1)
    prev_timestamp = df_s["timestamp"].shift(1)

    # Condition: sender changes
    is_response = (df_s["sender"] != prev_sender) & prev_sender.notna()

    # Response times in minutes
    gap_minutes = (df_s["timestamp"] - prev_timestamp).dt.total_seconds() / 60

    # Mask for all alternating messages
    mask_alternating = is_response

    # Count how many were excluded by the cap
    valid_gap_mask = (gap_minutes >= 0) & (gap_minutes < max_gap_minutes)
    excluded_count = int((mask_alternating & ~valid_gap_mask).sum())

    # Final mask for included responses
    mask_included = mask_alternating & valid_gap_mask

    # Build resulting DataFrame
    res_df = df_s.loc[mask_included].copy()
    res_df["responder"] = res_df["sender"]
    res_df["response_min"] = gap_minutes[mask_included]

    rdf = res_df[cols].reset_index(drop=True)

    if return_excluded:
        return rdf, excluded_count
    return rdf

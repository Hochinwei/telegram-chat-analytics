# ============================================================
#  TELEGRAM PRIVATE CHAT ANALYSIS
#  Analyzes two-person Telegram chats, focusing on conversation
#  dynamics, response times, messaging style, and how the
#  relationship evolved over time.
#
#  INSTALL DEPENDENCIES FIRST:
#      pip install beautifulsoup4 pandas matplotlib seaborn nltk wordcloud
#
#  HOW TO USE:
#      1. Export your Telegram chat via Telegram Desktop > Export Chat History (HTML format)
#      2. Place all messages*.html files in the same folder
#      3. Edit the CONFIG section below — especially YOUR_NAME and KEEP_SENDERS
#      4. Run: python private_chat_analysis.py
# ============================================================

import re
import os
import glob
import math
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import Counter
from datetime import datetime
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
warnings.filterwarnings("ignore")

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ════════════════════════════════════════════════════════════
#  CONFIG — EDIT THESE VALUES BEFORE RUNNING
# ════════════════════════════════════════════════════════════

# Path to your Telegram HTML exports (supports wildcards)
INPUT_PATTERN = "../data/raw/messages*.html"

# Where to save output charts and CSVs
OUTPUT_DIR = "../data/outputs/private"

# ── REQUIRED: your name exactly as it appears in the chat ───
# Check the exported HTML if unsure — it must match exactly.
YOUR_NAME = "Your Name Here"

# ── REQUIRED: both sender names exactly as they appear ──────
# Everything else (bots, channels, etc.) is excluded automatically.
KEEP_SENDERS = {"Your Name Here", "Their Name Here"}

# ── Conversation session gap ─────────────────────────────────
# A silence of this many minutes marks the start of a new
# conversation session. Affects initiator rate and session length.
SESSION_GAP_MINUTES = 60

# ── Timezone offset ─────────────────────────────────────────
# Telegram exports timestamps in UTC. Set your chat's local timezone.
# Examples: SGT = 8, BST = 1, EST = -5, JST = 9
TIMEZONE_OFFSET_HOURS = 0

# ── Colours ─────────────────────────────────────────────────
# You = teal, Them = orange. Change hex codes if preferred.
YOU_COLOR  = "#01696f"
THEM_COLOR = "#da7101"


# ════════════════════════════════════════════════════════════
#  TEXT CLEANING
#  Removes stopwords, URLs, punctuation, and short tokens
#  before any word frequency analysis.
# ════════════════════════════════════════════════════════════

STOP = set(stopwords.words("english"))
STOP.update([
    "la", "lah", "leh", "lor", "sia", "bro", "ah", "eh", "uh",
    "wa", "aiya", "liao", "wah", "de", "le", "ba",
    "u", "ur", "im", "dont", "gonna", "wanna", "ok", "okay",
    "ya", "yea", "yeah", "nah", "na", "hi",
    "haha", "hahaha", "lol", "omg", "wtf", "kk", "ha",
    "like", "get", "got", "know", "go", "one", "also",
])

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
#  SECTION 1: DATA PARSING
#  Reads all HTML files, extracts sender / timestamp / text,
#  deduplicates, sorts chronologically, and assigns sessions.
# ════════════════════════════════════════════════════════════

def parse_single_file(filepath):
    """Parse one Telegram HTML export and return a list of message dicts."""
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

        from_tag = body.find("div", class_="from_name")
        if from_tag:
            current_sender = from_tag.get_text(strip=True)

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

        text_tag  = body.find("div", class_="text")
        text      = text_tag.get_text(separator=" ", strip=True) if text_tag else None
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


def parse_all_files(pattern):
    """Combine all matching HTML files into one cleaned, sorted DataFrame."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found matching: {pattern}\n"
            "Check that INPUT_PATTERN points to the right folder."
        )

    print(f"  Found {len(files)} file(s):")
    all_records = []
    for fp in files:
        records = parse_single_file(fp)
        print(f"    {os.path.basename(fp):25s}  →  {len(records)} messages")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["sender", "timestamp", "text"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Shift timestamps from UTC to local timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)

    df["date"]    = df["timestamp"].dt.date
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month"]   = df["timestamp"].dt.to_period("M")
    df["year"]    = df["timestamp"].dt.year
    df["week"]    = df["timestamp"].dt.to_period("W")

    return df


def assign_sessions(df, gap_minutes=60):
    """
    Mark each message with a session ID and flag conversation initiators.
    A new session starts after silence exceeding gap_minutes.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    session_id   = 0
    session_ids  = []
    is_initiator = []

    for i, row in df.iterrows():
        if i == 0:
            session_ids.append(session_id)
            is_initiator.append(True)
        else:
            gap = (row["timestamp"] - df.loc[i - 1, "timestamp"]).total_seconds() / 60
            if gap > gap_minutes:
                session_id += 1
                is_initiator.append(True)
            else:
                is_initiator.append(False)
            session_ids.append(session_id)

    df["session_id"]   = session_ids
    df["is_initiator"] = is_initiator
    return df


# ════════════════════════════════════════════════════════════
#  SECTION 2: ACTIVITY & RHYTHM
#  How active was the chat, when did each person message,
#  and how did the balance of conversation shift over time?
# ════════════════════════════════════════════════════════════

def plot_activity(df, you, them, outdir):

    # A1 — Monthly message volume per person
    monthly = df.groupby(["month", "sender"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()
    for s in [you, them]:
        if s not in monthly.columns:
            monthly[s] = 0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly[you],  label=you,  color=YOU_COLOR,  marker="o", markersize=4)
    ax.plot(monthly.index, monthly[them], label=them, color=THEM_COLOR, marker="o", markersize=4)
    ax.fill_between(monthly.index, monthly[you],  alpha=0.15, color=YOU_COLOR)
    ax.fill_between(monthly.index, monthly[them], alpha=0.15, color=THEM_COLOR)
    ax.set_title("Monthly message volume", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Messages")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A1_monthly_volume.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A2 — Year-on-year total messages per person
    yearly = df.groupby(["year", "sender"]).size().unstack(fill_value=0)
    for s in [you, them]:
        if s not in yearly.columns:
            yearly[s] = 0

    x     = np.arange(len(yearly.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, yearly[you],  width, label=you,  color=YOU_COLOR,  alpha=0.85)
    b2 = ax.bar(x + width/2, yearly[them], width, label=them, color=THEM_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(yearly.index.astype(str))
    ax.set_xlabel("Year"); ax.set_ylabel("Total messages")
    ax.set_title("Year-on-year total messages", fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 2, str(int(h)),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A2_yearly_messages.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A3 — Monthly talk ratio: each person's % share of messages
    total_monthly = monthly[you] + monthly[them]
    you_pct  = (monthly[you]  / total_monthly * 100).fillna(50)
    them_pct = (monthly[them] / total_monthly * 100).fillna(50)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(monthly.index, [you_pct, them_pct],
                 labels=[you, them], colors=[YOU_COLOR, THEM_COLOR], alpha=0.75)
    ax.axhline(50, color="white", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Monthly talk ratio (% of messages)", fontweight="bold")
    ax.set_ylabel("% share"); ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A3_talk_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A4 — Day × hour heatmaps, side-by-side per person
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, sender, title in zip(
        axes,
        [you, them],
        [f"{you} — activity heatmap", f"{them} — activity heatmap"]
    ):
        sub    = df[df["sender"] == sender]
        hourly = sub.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        hourly = hourly.reindex([d for d in day_order if d in hourly.index])
        im = ax.imshow(hourly.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.set_yticks(range(len(hourly.index))); ax.set_yticklabels(hourly.index, fontsize=8)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Hour of day")
        plt.colorbar(im, ax=ax, label="Messages")
    fig.suptitle("When does each person message?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/A4_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A5 — Day-of-week bar comparison
    dow_counts = df.groupby(["weekday", "sender"]).size().unstack(fill_value=0)
    dow_counts = dow_counts.reindex([d for d in day_order if d in dow_counts.index])
    for s in [you, them]:
        if s not in dow_counts.columns:
            dow_counts[s] = 0

    x = np.arange(len(dow_counts)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, dow_counts[you],  width, label=you,  color=YOU_COLOR,  alpha=0.85)
    ax.bar(x + width/2, dow_counts[them], width, label=them, color=THEM_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(dow_counts.index, rotation=20, ha="right")
    ax.set_ylabel("Messages"); ax.set_title("Messages by day of week", fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A5_day_of_week.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅  Activity & rhythm charts saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 3: RESPONSE TIME ANALYSIS
#  How long does each person take to reply? Has this changed
#  over the course of the relationship?
# ════════════════════════════════════════════════════════════

def plot_response_times(df, you, them, outdir):

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    rt_records = []
    for i in range(1, len(df_sorted)):
        prev     = df_sorted.loc[i - 1]
        curr     = df_sorted.loc[i]
        gap_mins = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 60
        if curr["sender"] != prev["sender"] and 0 < gap_mins <= SESSION_GAP_MINUTES:
            rt_records.append({
                "responder":        curr["sender"],
                "response_minutes": gap_mins,
                "month":            curr["timestamp"].to_period("M"),
            })
    rt = pd.DataFrame(rt_records)

    if rt.empty:
        print("  ⚠️  No response time data found.")
        return

    # R1 — Response time distribution per person (histogram with median marker)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Response time distribution", fontsize=13, fontweight="bold")
    for ax, sender, color in zip(axes, [you, them], [YOU_COLOR, THEM_COLOR]):
        data = rt[rt["responder"] == sender]["response_minutes"].dropna()
        if data.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center"); continue
        ax.hist(data.clip(upper=SESSION_GAP_MINUTES), bins=30, color=color, alpha=0.8, edgecolor="white")
        median_val = data.median()
        ax.axvline(median_val, color="#333", linestyle="--", linewidth=1.5,
                   label=f"Median: {median_val:.1f} min")
        ax.set_title(f"{sender}", fontweight="bold")
        ax.set_xlabel("Minutes to reply"); ax.set_ylabel("Frequency")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/R1_response_time_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # R2 — Monthly trend of median response time
    monthly_rt = rt.groupby(["month", "responder"])["response_minutes"].median().unstack()
    monthly_rt.index = monthly_rt.index.to_timestamp()
    for s in [you, them]:
        if s not in monthly_rt.columns:
            monthly_rt[s] = np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_rt.index, monthly_rt[you],  label=you,  color=YOU_COLOR,  marker="o", markersize=4)
    ax.plot(monthly_rt.index, monthly_rt[them], label=them, color=THEM_COLOR, marker="o", markersize=4)
    ax.set_title("Median response time per month (minutes)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Median minutes to reply")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/R2_response_time_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅  Response time charts saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 4: CONVERSATION DYNAMICS
#  Who starts conversations, how long do they last, and
#  what are the biggest gaps in the chat's history?
# ════════════════════════════════════════════════════════════

def plot_conversation_dynamics(df, you, them, outdir):

    # D1 — Pie chart: who initiates conversations more often
    initiators  = df[df["is_initiator"]]["sender"].value_counts()
    colors_pie  = [YOU_COLOR if s == you else THEM_COLOR for s in initiators.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        initiators.values, labels=initiators.index, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold")
    ax.set_title(f"Who starts conversations?\n(gap > {SESSION_GAP_MINUTES} min)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/D1_conversation_initiator.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D2 — Average session length per month
    monthly_session_avg = df.groupby(["month", "session_id"]).size().reset_index(name="length")
    monthly_session_avg = monthly_session_avg.groupby("month")["length"].mean()
    monthly_session_avg.index = monthly_session_avg.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly_session_avg.index, monthly_session_avg.values, width=20, color=YOU_COLOR, alpha=0.8)
    ax.set_title("Average conversation length per month (messages per session)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg messages per session")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D2_session_length.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D3 — Burst length: how many messages in a row before other replies
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    turns = []
    current_sender = df_sorted.loc[0, "sender"]; count = 1
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, "sender"] == current_sender:
            count += 1
        else:
            turns.append({"sender": current_sender, "burst_length": count})
            current_sender = df_sorted.loc[i, "sender"]; count = 1
    turns.append({"sender": current_sender, "burst_length": count})
    turns_df = pd.DataFrame(turns)

    fig, ax = plt.subplots(figsize=(8, 5))
    for sender, color in [(you, YOU_COLOR), (them, THEM_COLOR)]:
        data = turns_df[turns_df["sender"] == sender]["burst_length"]
        ax.hist(data.clip(upper=15), bins=15, alpha=0.7, label=sender, color=color, edgecolor="white")
    ax.set_xlabel("Messages sent per turn"); ax.set_ylabel("Frequency")
    ax.set_title("Message burst length per turn", fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D3_burst_length.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D4 — Top 10 longest silences (only gaps > 1 day)
    gaps = []
    for i in range(1, len(df_sorted)):
        gap_hours = (df_sorted.loc[i,"timestamp"] - df_sorted.loc[i-1,"timestamp"]).total_seconds() / 3600
        if gap_hours > 24:
            gaps.append({
                "start": df_sorted.loc[i-1,"timestamp"].date(),
                "end":   df_sorted.loc[i,"timestamp"].date(),
                "gap_days": gap_hours / 24,
            })
    gaps_df = pd.DataFrame(gaps).sort_values("gap_days", ascending=False).head(10)

    if not gaps_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [f"{r.start} → {r.end}" for _, r in gaps_df.iterrows()]
        ax.barh(labels[::-1], gaps_df["gap_days"].values[::-1], color=THEM_COLOR, alpha=0.85)
        ax.set_xlabel("Silence duration (days)")
        ax.set_title("Top 10 longest silences in the chat", fontweight="bold")
        for i, v in enumerate(gaps_df["gap_days"].values[::-1]):
            ax.text(v + 0.2, i, f"{v:.1f} days", va="center", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{outdir}/D4_longest_silences.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("✅  Conversation dynamics charts saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 5: MESSAGING STYLE
#  How does each person write? Message length, question rate,
#  and vocabulary overlap as markers of communication style.
# ════════════════════════════════════════════════════════════

def plot_messaging_style(df, you, them, outdir):

    # S1 — Message length over time
    monthly_len = df.groupby(["month", "sender"])["msg_len"].mean().unstack()
    monthly_len.index = monthly_len.index.to_timestamp()
    for s in [you, them]:
        if s not in monthly_len.columns:
            monthly_len[s] = np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_len.index, monthly_len[you],  label=you,  color=YOU_COLOR,  marker="o", markersize=4)
    ax.plot(monthly_len.index, monthly_len[them], label=them, color=THEM_COLOR, marker="o", markersize=4)
    ax.set_title("Average message length over time (characters)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg message length")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S1_message_length_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # S2 — Overall question rate per person
    df["has_question"] = df["text"].str.contains(r"\?", regex=True)
    q_rate = (df.groupby("sender")["has_question"].mean() * 100).reindex([you, them]).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(q_rate.index, q_rate.values, color=[YOU_COLOR, THEM_COLOR], alpha=0.85, width=0.4)
    ax.set_ylabel("% of messages containing a question")
    ax.set_title("Question rate per person", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S2_question_rate.png", dpi=150, bbox_inches="tight")
    plt.close()

    # S3 — Vocabulary overlap (shared vs. unique words)
    you_tokens  = set(get_tokens(" ".join(df[df["sender"] == you ]["text"])))
    them_tokens = set(get_tokens(" ".join(df[df["sender"] == them]["text"])))
    shared      = you_tokens & them_tokens

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["Shared words", f"{you} only", f"{them} only"]
    values     = [len(shared), len(you_tokens - them_tokens), len(them_tokens - you_tokens)]
    bars = ax.bar(categories, values, color=[THEM_COLOR, YOU_COLOR, "#888"], alpha=0.85, width=0.4)
    ax.set_ylabel("Unique word count"); ax.set_title("Vocabulary overlap", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S3_vocabulary_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # S4 — Question rate trend over time
    monthly_q = df.groupby(["month", "sender"])["has_question"].mean().unstack() * 100
    monthly_q.index = monthly_q.index.to_timestamp()
    for s in [you, them]:
        if s not in monthly_q.columns:
            monthly_q[s] = np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_q.index, monthly_q[you],  label=you,  color=YOU_COLOR,  marker="o", markersize=4)
    ax.plot(monthly_q.index, monthly_q[them], label=them, color=THEM_COLOR, marker="o", markersize=4)
    ax.set_title("Question rate over time (% messages with ?)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("% messages with a question")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S4_question_rate_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅  Messaging style charts saved.")


# ════════════════════════════════════════════════════════════
#  SECTION 6: WORD PATTERNS BY YEAR
#  Top words, bigrams, and word clouds broken down by year
#  per person, to show how language evolved over time.
# ════════════════════════════════════════════════════════════

def plot_word_patterns_by_year(df, you, them, outdir):

    years = sorted(df["year"].unique())

    for year in years:
        df_year = df[df["year"] == year]

        # W1 — Top 15 words and bigrams per person for this year
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Word patterns — {year}", fontsize=15, fontweight="bold")

        for col_idx, (sender, color) in enumerate([(you, YOU_COLOR), (them, THEM_COLOR)]):
            tokens = []
            for text in df_year[df_year["sender"] == sender]["text"]:
                tokens.extend(get_tokens(text))

            ax_w = axes[0][col_idx]
            if tokens:
                top_words = Counter(tokens).most_common(15)
                w, f = zip(*top_words)
                ax_w.barh(list(w)[::-1], list(f)[::-1], color=color, alpha=0.85)
            else:
                ax_w.text(0.5, 0.5, "No data", transform=ax_w.transAxes, ha="center")
            ax_w.set_title(f"{sender} — top 15 words", fontweight="bold")
            ax_w.set_xlabel("Frequency")
            ax_w.spines["top"].set_visible(False); ax_w.spines["right"].set_visible(False)

            ax_b = axes[1][col_idx]
            if len(tokens) >= 2:
                top_bi = Counter(ngrams(tokens, 2)).most_common(15)
                bl = [" ".join(b) for b, _ in top_bi]
                bv = [c for _, c in top_bi]
                ax_b.barh(bl[::-1], bv[::-1], color=color, alpha=0.85)
            else:
                ax_b.text(0.5, 0.5, "No data", transform=ax_b.transAxes, ha="center")
            ax_b.set_title(f"{sender} — top 15 bigrams", fontweight="bold")
            ax_b.set_xlabel("Frequency")
            ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{outdir}/W1_words_bigrams_{year}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # W2 — Word clouds per person for this year
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Word clouds — {year}", fontsize=14, fontweight="bold")

        for ax, sender, color in [(axes[0], you, YOU_COLOR), (axes[1], them, THEM_COLOR)]:
            tokens = get_tokens(" ".join(df_year[df_year["sender"] == sender]["text"].tolist()))
            if tokens:
                wc = WordCloud(
                    width=700, height=400, background_color="white",
                    color_func=lambda *a, **k: color,
                    max_words=80, prefer_horizontal=0.8,
                ).generate(" ".join(tokens))
                ax.imshow(wc, interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "No messages this year",
                        transform=ax.transAxes, ha="center", va="center", fontsize=12, color="grey")
            ax.axis("off"); ax.set_title(sender, fontweight="bold", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{outdir}/W2_wordclouds_{year}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("✅  Word patterns by year saved.")


# ════════════════════════════════════════════════════════════
#  MAIN — runs all sections in order
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse all HTML files
    print("\n📂  Parsing HTML files...")
    df = parse_all_files(INPUT_PATTERN)

    # 2. Keep only the two specified senders — all others are excluded automatically
    df = df[df["sender"].isin(KEEP_SENDERS)].reset_index(drop=True)
    print(f"    Kept senders: {sorted(df['sender'].unique().tolist())}")

    # 3. Confirm exactly two senders remain
    senders = df["sender"].unique().tolist()
    if len(senders) != 2:
        print(f"  ⚠️  Expected 2 senders, found {len(senders)}: {senders}")
        print("     Check that names in KEEP_SENDERS match exactly (case-sensitive).")
        raise SystemExit

    you  = YOUR_NAME
    them = [s for s in senders if s != you][0]
    print(f"    You  : {you}")
    print(f"    Them : {them}")
    print(f"    Range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"    Total: {len(df)} messages\n")

    # 4. Assign session IDs and initiator flags
    df = assign_sessions(df, gap_minutes=SESSION_GAP_MINUTES)

    # 5. Save cleaned CSV
    df.to_csv(f"{OUTPUT_DIR}/parsed_messages.csv", index=False)

    # 6. Run all analysis sections
    print("📊  Section 1: Activity & rhythm...")
    plot_activity(df, you, them, OUTPUT_DIR)

    print("⏱️   Section 2: Response times...")
    plot_response_times(df, you, them, OUTPUT_DIR)

    print("💬  Section 3: Conversation dynamics...")
    plot_conversation_dynamics(df, you, them, OUTPUT_DIR)

    print("🔤  Section 4: Messaging style...")
    plot_messaging_style(df, you, them, OUTPUT_DIR)

    print("📅  Section 5: Word patterns by year...")
    plot_word_patterns_by_year(df, you, them, OUTPUT_DIR)

    print(f"\n🎉  Done! All charts saved to '{OUTPUT_DIR}/'")

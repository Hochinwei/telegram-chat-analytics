<span align="center">

# 📱 Telegram Chat Analysis

</span>

A Python NLP pipeline for analyzing exported Telegram chats — works with both **group chats** and **private (two-person) chats**.

This was built as a personal showcase of applied Natural Language Processing (NLP) and data visualization techniques using Python, pandas, NLTK, and matplotlib.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How to Export Your Telegram Chat](#how-to-export-your-telegram-chat)
- [Installation](#installation)
- [Scripts](#scripts)
    - [1. Group Chat Analysis](#1-group-chat-analysis)
    - [2. Private Chat Analysis](#2-private-chat-analysis)
- [Key Parameters to Configure](#key-parameters-to-configure)
- [Sample Outputs](#sample-outputs)
- [NLP Techniques Used](#nlp-techniques-used)
- [Project Structure](#project-structure)

---

## Overview

This project takes raw HTML exports that can be extracted from Telegram Desktop and turns them into visual insights. There are two scripts, each creating two different kinds of analysis:

| Script | Best for |
|---|---|
| `group_chat_analysis.py` | Group chats with 3+ members |
| `private_chat_analysis.py` | Two-person private chats |

All scripts handle multiple telegram export files automatically (i.e., `messages.html`, `messages2.html`, ... `messages10.html`).

---

## How to Export Your Telegram Chat

1. Open **Telegram Desktop** (not mobile)
2. Open the chat you want to export
3. Click the **three-dot menu** (⋮) at the top right
4. Select **Export Chat History**
5. Under format, select **HTML**
6. Uncheck media if you only want text analysis
7. Click Export — this produces one or more `messages*.html` files

> **Note:** Telegram splits large chats into multiple files (`messages.html`, `messages2.html`, etc.). All scripts automatically detect and combine them. All you have to do is upload these files into the `data\raw` folder.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/telegram-chat-analysis.git
cd telegram-chat-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
beautifulsoup4
pandas
matplotlib
seaborn
nltk
wordcloud
networkx
```

---

## Scripts

### 1. Group Chat Analysis

**File:** `scripts/group_chat_analysis.py`

Analyzes group chats with multiple members. Produces charts on:

- **Activity stats**: message counts per member, most active days, hourly heatmaps
- **Word patterns**: top words, bigrams, word clouds per member, vocabulary richness
- **Word patterns by year**: word clouds and top words/bigrams split by calendar year
- **Interaction metrics**: reply rates, media sharing, who replies to whom (network graph)

#### ⚙️ Parameters to configure

Open `scripts/group_chat_analysis.py` and edit the CONFIG block at the top:

```python
python -m scripts.group_chat_analysis \
  --input-pattern "data/raw/messages*.html" \
  --output-dir "data/outputs/group" \
  --exclude-sender "SG SHARES" \
  --exclude-sender "🇸🇬SGFOLLOWSALL🇸🇬"
```

Key arguments:

`--input-pattern` (required): Glob for your Telegram HTML export files, e.g. data/raw/messages*.html.

`--output-dir` (required): Directory where all charts and CSVs will be written.

`--exclude-sender` (optional, repeatable): Names to exclude (bots/channels/spam); you can pass this flag multiple times.

You can see all options with:

```python
python -m scripts.group_chat_analysis --help
```

> **Tip:** After the first run, inspect `parsed_messages.csv` in the output directory to check the exact sender names, then rerun with additional `--exclude-sender` arguments if needed.

---

### 2. Private Chat Analysis

**File:** `scripts/private_chat_analysis.py`

Analyzes two-person private chats. Focuses on conversation dynamics and relationship patterns:

- **Activity & rhythm**: monthly volume, year-on-year totals, talk ratio over time, day/hour heatmaps
- **Response times**: distribution of reply speeds, trend over time
- **Conversation dynamics**: who initiates conversations, session lengths, burst patterns, longest silences
- **Messaging style**: message length over time, question rates, vocabulary overlap
- **Word patterns by year**: top words, bigrams, and word clouds split by year per person

#### ⚙️ Example argument

```python
python -m scripts.private_chat_analysis \
  --input-pattern "data/raw/messages*.html" \
  --output-dir "data/outputs/private" \
  --your-name "Your Name Here" \
  --other-name "Their Name Here" \
  --session-gap-minutes 60
```

#### Key arguments:

`--input-pattern` (required): Glob for your Telegram HTML export files, e.g. data/raw/messages*.html.

`--output-dir` (required): Directory where all charts and CSVs will be written.

`--your-name` (required): Your own display name in the Telegram export.

`--other-name` (required): The other person’s display name in the chat.

`--session-gap-minutes` (optional): Minutes of silence that define a new “conversation session” (default matches the script’s internal default).

> **Important:** `--your-name` and `--other-name` must match exactly what appears in the export (including capitalisation and spacing); if unsure, run once and check `parsed_messages.csv` in the output directory.

---

## Sample Outputs

### Group Chat

| Chart | Description |
|---|---|
| `A1_activity_stats.png` | Message count, avg length, night owl index per member |
| `A2_messages_over_time.png` | Monthly message volume per member |
| `A3_hourly_heatmap.png` | Day × hour activity heatmap |
| `B1_top_words.png` | Top 15 most-used words (stopwords removed) |
| `B2_bigrams.png` | Top 15 two-word phrases |
| `B3_wordclouds.png` | Word cloud per member |
| `B4_vocabulary.png` | Unique vocabulary size per member |
| `C1_reply_rate.png` | % of messages that are direct replies |
| `C2_media_sharing.png` | Media messages per member |
| `C3_reply_network.png` | Network graph of who replies to whom |
| `C4_top_days.png` | Top 10 most active days |
| `W1_words_bigrams_{year}.png` | Top words & bigrams per member per year |
| `W2_wordclouds_{year}.png` | Word clouds per member per year |

### Private Chat

| Chart | Description |
|---|---|
| `A1_monthly_volume.png` | Monthly messages per person |
| `A2_yearly_messages.png` | Year-on-year totals side by side |
| `A3_talk_ratio.png` | Monthly share of messages per person (%) |
| `A4_heatmaps.png` | Day × hour heatmaps, one per person |
| `A5_day_of_week.png` | Messages by day of week |
| `R1_response_time_dist.png` | Reply speed distribution with median |
| `R2_response_time_trend.png` | Median response time per month |
| `D1_conversation_initiator.png` | Who starts conversations more often |
| `D2_session_length.png` | Avg messages per session per month |
| `D3_burst_length.png` | Messages sent per turn before other replies |
| `D4_longest_silences.png` | Top 10 longest gaps in the chat |
| `S1_message_length_trend.png` | Average message length over time |
| `S2_question_rate.png` | Who asks more questions |
| `S3_vocabulary_overlap.png` | Shared vs. unique vocabulary |
| `S4_question_rate_trend.png` | Question rate trend over time |
| `W1_words_bigrams_{year}.png` | Top words & bigrams per person per year |
| `W2_wordclouds_{year}.png` | Word clouds per person per year |

---

## NLP Techniques Used

- **Tokenization** - sentence and word tokenization via NLTK
- **Stopword removal** - English stopwords + custom terms
- **Lowercasing and normalization** - consistent text cleaning before any analysis
- **N-gram extraction** - bigram frequency analysis using NLTK's `ngrams`
- **Word frequency analysis** - `Counter`-based term frequency ranking
- **Vocabulary richness** - unique token count as a proxy for lexical diversity
- **Lexical mirroring** - vocabulary overlap between two speakers as a measure of linguistic alignment
- **Question detection** - regex-based pragmatic feature extraction
- **Temporal pattern analysis** - time-of-day, day-of-week, and seasonal message trends
- **Word clouds** - visual frequency summaries using `wordcloud`
- **Network analysis** - reply graph construction using `networkx`
---

*Built by [Ho Chin Wei](https://github.com/HoChinWei) · PhD student in Cognitive Psychology · University of Hull*
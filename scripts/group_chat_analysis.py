# ============================================================
# TELEGRAM GROUP CHAT ANALYSIS
# Parses Telegram HTML exports and produces visual analysis
# of word patterns, activity stats, and interaction metrics
# for group chats with 3 or more members.
#
# INSTALL DEPENDENCIES FIRST:
#   pip install beautifulsoup4 pandas matplotlib nltk wordcloud networkx
#
# HOW TO USE:
#   python group_chat_analysis.py \
#     --input  "../data/raw/messages*.html" \
#     --output "../data/outputs/group" \
#     --timezone 8 \
#     --exclude "SG SHARES" "BotName"
# ============================================================

import os
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud
import networkx as nx

# Shared utilities — parsed HTML, text cleaning, colour palette
from utils import get_tokens, parse_all_files, sender_colors

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CLI INTERFACE
# ════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a Telegram group chat HTML export."
    )
    parser.add_argument(
        "--input", default="../data/raw/messages*.html",
        help="Glob pattern for Telegram HTML export files "
             "(default: ../data/raw/messages*.html)"
    )
    parser.add_argument(
        "--output", default="../data/outputs/group",
        help="Directory to save output charts (default: ../data/outputs/group)"
    )
    parser.add_argument(
        "--timezone", type=int, default=0,
        help="UTC offset in hours (e.g. 8 for SGT, 1 for BST; default: 0)"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Additional sender names to exclude, on top of automatic bot-filtering "
             "(e.g. --exclude 'SG SHARES' 'ChannelName')"
    )
    return parser.parse_args()

# ════════════════════════════════════════════════════════════
# SECTION 1: ACTIVITY STATS
# Who messages most, when are peak hours, and how have
# message volumes changed over time?
# ════════════════════════════════════════════════════════════

def plot_activity_stats(df, colors, outdir):

    # A1 — Bar chart: total messages and avg message length per sender
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(colors)), 5))

    sender_counts = df["sender"].value_counts()
    avg_len = df.groupby("sender")["msg_len"].mean().reindex(sender_counts.index)

    axes[0].bar(sender_counts.index, sender_counts.values,
                color=[colors[s] for s in sender_counts.index])
    axes[0].set_title("Total messages per member", fontweight="bold")
    axes[0].set_ylabel("Messages")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].bar(avg_len.index, avg_len.values,
                color=[colors[s] for s in avg_len.index])
    axes[1].set_title("Average message length per member", fontweight="bold")
    axes[1].set_ylabel("Characters")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/A1_activity_stats.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A2 — Line chart: messages per month per sender
    monthly = df.groupby(["month", "sender"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    for sender in monthly.columns:
        ax.plot(monthly.index, monthly[sender], label=sender,
                color=colors.get(sender, "#888"), marker="o", markersize=3)
    ax.set_title("Monthly message volume per member", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Messages")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A2_messages_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A3 — Heatmap: average messages by day × hour
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hourly = df.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    hourly = hourly.reindex([d for d in day_order if d in hourly.index])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(hourly.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(hourly.index))); ax.set_yticklabels(hourly.index)
    ax.set_title("Group activity by day and hour", fontweight="bold")
    ax.set_xlabel("Hour of day")
    plt.colorbar(im, ax=ax, label="Messages")
    plt.tight_layout()
    plt.savefig(f"{outdir}/A3_hourly_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A4 — Top 10 most active days
    top_days = df.groupby("date").size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([str(d) for d in top_days.index[::-1]], top_days.values[::-1], color="#01696f")
    ax.set_xlabel("Messages"); ax.set_title("Top 10 most active days", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/C4_top_days.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Activity charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 2: WORD PATTERNS
# Most-used words, bigrams, word clouds, and vocabulary
# richness for the whole group and for each member.
# ════════════════════════════════════════════════════════════

def plot_word_patterns(df, colors, outdir, top_n=15):

    all_tokens = []
    for text in df["text"]:
        all_tokens.extend(get_tokens(text))

    # B1 — Top N single words (whole group)
    top_words = Counter(all_tokens).most_common(top_n)
    words, freqs = zip(*top_words)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(words)[::-1], list(freqs)[::-1], color="#01696f")
    ax.set_title(f"Top {top_n} most-used words (group)", fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/B1_top_words.png", dpi=150, bbox_inches="tight")
    plt.close()

    # B2 — Top N bigrams (whole group)
    top_bigrams = Counter(ngrams(all_tokens, 2)).most_common(top_n)
    labels = [" ".join(b) for b, _ in top_bigrams]
    vals   = [c for _, c in top_bigrams]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], vals[::-1], color="#da7101")
    ax.set_title(f"Top {top_n} bigrams (group)", fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/B2_bigrams.png", dpi=150, bbox_inches="tight")
    plt.close()

    # B3 — Word cloud per member
    # FIX: use default-argument capture (sender=sender) so each cloud gets
    # its own colour rather than all sharing the last loop value.
    senders = sorted(df["sender"].unique())
    cols = 3
    rows = math.ceil(len(senders) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = np.array(axes).flatten()

    for i, sender in enumerate(senders):
        sender_text = " ".join(df[df["sender"] == sender]["text"].tolist())
        tokens = get_tokens(sender_text)
        ax = axes[i]
        if tokens:
            wc = WordCloud(
                width=600, height=350,
                background_color="white",
                color_func=lambda *a, sender=sender, **k: colors.get(sender, "#333"),
                max_words=80, prefer_horizontal=0.8,
            ).generate(" ".join(tokens))
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(0.5, 0.5, "No text messages",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
        ax.axis("off")
        ax.set_title(sender, fontweight="bold")

    for j in range(len(senders), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Word clouds per member", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/B3_wordclouds.png", dpi=150, bbox_inches="tight")
    plt.close()

    # B4 — Vocabulary richness (unique word count per member)
    vocab_sizes = {
        s: len(set(get_tokens(" ".join(df[df["sender"] == s]["text"]))))
        for s in senders
    }
    fig, ax = plt.subplots(figsize=(max(10, len(senders)), 5))
    ax.bar(vocab_sizes.keys(), vocab_sizes.values(),
           color=[colors.get(s, "#888") for s in vocab_sizes])
    ax.set_title("Vocabulary richness per member (unique words)", fontweight="bold")
    ax.set_ylabel("Unique words")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/B4_vocabulary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Word pattern charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 3: INTERACTION METRICS
# Who replies to whom, who shares media, and how often does
# each member engage in direct replies?
# ════════════════════════════════════════════════════════════

def plot_interaction_metrics(df, colors, outdir):

    senders = sorted(df["sender"].unique())

    # C1 — Reply rate per member
    df["is_reply"] = (
        df["text"].str.startswith(">") |
        df["text"].str.contains("replied to", case=False, na=False)
    )
    reply_rate = (
        df.groupby("sender")["is_reply"].mean() * 100
    ).reindex(senders, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(10, len(senders)), 5))
    ax.bar(reply_rate.index, reply_rate.values,
           color=[colors.get(s, "#888") for s in reply_rate.index])
    ax.set_title("Reply rate per member (% of messages as direct replies)", fontweight="bold")
    ax.set_ylabel("% messages that are replies")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/C1_reply_rate.png", dpi=150, bbox_inches="tight")
    plt.close()

    # C2 — Media sharing per member
    media_counts = (
        df[df["has_media"]].groupby("sender").size()
    ).reindex(senders, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(10, len(senders)), 5))
    ax.bar(media_counts.index, media_counts.values,
           color=[colors.get(s, "#888") for s in media_counts.index])
    ax.set_title("Media messages per member", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/C2_media_sharing.png", dpi=150, bbox_inches="tight")
    plt.close()

    # C3 — Reply network graph (who replies to whom)
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    reply_pairs = Counter()
    for i in range(1, len(df_sorted)):
        curr = df_sorted.loc[i]
        prev = df_sorted.loc[i - 1]
        gap  = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 60
        if curr["sender"] != prev["sender"] and gap < 10:
            reply_pairs[(prev["sender"], curr["sender"])] += 1

    G = nx.DiGraph()
    for (src, tgt), weight in reply_pairs.items():
        G.add_edge(src, tgt, weight=weight)

    if G.number_of_nodes() > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        pos          = nx.spring_layout(G, seed=42, k=2)
        node_sizes   = [300 + G.degree(n) * 200 for n in G.nodes()]
        node_colors  = [colors.get(n, "#888") for n in G.nodes()]
        edge_widths  = [G[u][v]["weight"] / max(reply_pairs.values()) * 5 for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                               edge_color="#555", arrows=True, arrowsize=15, ax=ax)
        ax.set_title("Reply network (who replies to whom)", fontweight="bold", fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{outdir}/C3_reply_network.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("✅ Interaction metric charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 4: WORD PATTERNS BY YEAR
# Top words, bigrams, and word clouds broken down by year
# for each sender, to track how language evolved over time.
# ════════════════════════════════════════════════════════════

def plot_word_patterns_by_year(df, colors, outdir):

    years   = sorted(df["year"].unique())
    senders = sorted(df["sender"].unique())

    for year in years:
        df_year = df[df["year"] == year]

        # W1 — Top 15 words and bigrams per sender for this year
        cols = 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle(f"Word patterns — {year}", fontsize=15, fontweight="bold")

        for col_idx, sender in enumerate(senders[:2]):
            tokens = []
            for text in df_year[df_year["sender"] == sender]["text"]:
                tokens.extend(get_tokens(text))

            color = colors.get(sender, "#888")

            ax_words = axes[0][col_idx]
            if tokens:
                top_words = Counter(tokens).most_common(15)
                w, f = zip(*top_words)
                ax_words.barh(list(w)[::-1], list(f)[::-1], color=color, alpha=0.85)
            else:
                ax_words.text(0.5, 0.5, "No data", transform=ax_words.transAxes, ha="center")
            ax_words.set_title(f"{sender} — top 15 words", fontweight="bold")
            ax_words.set_xlabel("Frequency")
            ax_words.spines["top"].set_visible(False); ax_words.spines["right"].set_visible(False)

            ax_bigrams = axes[1][col_idx]
            if len(tokens) >= 2:
                top_bi = Counter(ngrams(tokens, 2)).most_common(15)
                bl = [" ".join(b) for b, _ in top_bi]
                bv = [c for _, c in top_bi]
                ax_bigrams.barh(bl[::-1], bv[::-1], color=color, alpha=0.85)
            else:
                ax_bigrams.text(0.5, 0.5, "No data", transform=ax_bigrams.transAxes, ha="center")
            ax_bigrams.set_title(f"{sender} — top 15 bigrams", fontweight="bold")
            ax_bigrams.set_xlabel("Frequency")
            ax_bigrams.spines["top"].set_visible(False); ax_bigrams.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{outdir}/W1_words_bigrams_{year}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # W2 — Word clouds per sender for this year
        # FIX: use default-argument capture (sender=sender) so each cloud gets
        # its own colour rather than all sharing the last loop value.
        cols_wc  = 3
        rows_wc  = math.ceil(len(senders) / cols_wc)
        fig, axes = plt.subplots(rows_wc, cols_wc, figsize=(18, 5 * rows_wc))
        fig.suptitle(f"Word clouds — {year}", fontsize=14, fontweight="bold")
        axes = np.array(axes).flatten()

        for i, sender in enumerate(senders):
            sender_text = " ".join(df_year[df_year["sender"] == sender]["text"].tolist())
            tokens = get_tokens(sender_text)
            ax = axes[i]
            if tokens:
                wc = WordCloud(
                    width=600, height=350, background_color="white",
                    color_func=lambda *a, sender=sender, **k: colors.get(sender, "#333"),
                    max_words=80, prefer_horizontal=0.8,
                ).generate(" ".join(tokens))
                ax.imshow(wc, interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "No messages this year",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="grey")
            ax.axis("off")
            ax.set_title(sender, fontweight="bold", fontsize=10)

        for j in range(len(senders), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{outdir}/W2_wordclouds_{year}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("✅ Word pattern by year charts saved.")

# ════════════════════════════════════════════════════════════
# MAIN — runs all sections in order
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    INPUT_PATTERN        = args.input
    OUTPUT_DIR           = args.output
    TIMEZONE_OFFSET      = args.timezone
    EXCLUDE_SENDERS      = set(args.exclude)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse all HTML files
    print("\n📂 Parsing HTML files...")
    df = parse_all_files(INPUT_PATTERN, timezone_offset=TIMEZONE_OFFSET)

    # 2. Filter out bots and any user-specified senders
    bot_mask     = (
        df["sender"].str.contains("bot", case=False, na=False) |
        df["sender"].str.contains("via @", case=False, na=False)
    )
    exclude_mask = df["sender"].isin(EXCLUDE_SENDERS)
    removed      = df[bot_mask | exclude_mask]["sender"].unique().tolist()
    df = df[~(bot_mask | exclude_mask)].reset_index(drop=True)
    if removed:
        print(f"  Filtered out: {removed}")

    # 3. Print summary
    senders = sorted(df["sender"].unique().tolist())
    print(f"  Members   : {senders}")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Total msgs: {len(df)}\n")

    # 4. Assign consistent colours
    colors = sender_colors(senders)

    # 5. Save cleaned CSV for further inspection
    df.to_csv(f"{OUTPUT_DIR}/parsed_messages.csv", index=False)

    # 6. Run all analysis sections
    print("📊 Section 1: Activity stats...")
    plot_activity_stats(df, colors, OUTPUT_DIR)

    print("🔤 Section 2: Word patterns...")
    plot_word_patterns(df, colors, OUTPUT_DIR)

    print("🔗 Section 3: Interaction metrics...")
    plot_interaction_metrics(df, colors, OUTPUT_DIR)

    print("📅 Section 4: Word patterns by year...")
    plot_word_patterns_by_year(df, colors, OUTPUT_DIR)

    print(f"\n🎉 Done! All charts saved to '{OUTPUT_DIR}/'")

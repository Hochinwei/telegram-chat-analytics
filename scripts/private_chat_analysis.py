# ============================================================
# TELEGRAM PRIVATE CHAT ANALYSIS
# Parses Telegram HTML exports and produces visual analysis
# of conversation dynamics, response times, and word patterns
# for two-person private chats.
#
# INSTALL DEPENDENCIES FIRST:
#   pip install beautifulsoup4 pandas matplotlib seaborn nltk wordcloud
#
# HOW TO USE:
#   python private_chat_analysis.py \
#     --your_name  "Your Name" \
#     --other_name "Their Name" \
#     --input      "../data/raw/messages*.html" \
#     --output     "../data/outputs" \
#     --timezone   8 \
#     --max-response-gap 1440
# ============================================================

import argparse
import os
import sys
import warnings

# Make local import work during tests
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from nltk.util import ngrams
from wordcloud import WordCloud

# Shared utilities — HTML parsing, text cleaning, colour palette
from utils import compute_response_times, get_tokens, parse_all_files, sender_colors

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CLI INTERFACE
# ════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a two-person Telegram private chat export."
    )
    parser.add_argument(
        "--input", default="data/raw/messages*.html",
        help="Glob pattern for Telegram HTML export files "
             "(default: ../data/raw/messages*.html)"
    )
    parser.add_argument(
        "--output", default="data/outputs",
        help="Directory to save output charts (default: ../data/outputs/private)"
    )

    ## Add both names as shown in the telegram chat logs
    parser.add_argument(
        "--your_name", required=True,
        help="Your name EXACTLY as it appears in the Telegram export"
    )
    parser.add_argument(
        "--other_name", required=True,
        help="The other person's name EXACTLY as it appears in the Telegram export"
    )
    parser.add_argument(
        "--timezone", type=int, default=0,
        help="UTC offset in hours (e.g. 8 for SGT, 1 for BST; default: 0)"
    )
    parser.add_argument(
        "--session-gap", type=int, default=60,
        help="Minutes of silence that mark the start of a new conversation (default: 60)"
    )
    parser.add_argument(
        "--max-response-gap", type=int, default=1440,
        help="Longer gaps are treated as a new conversation, not a reply (default: 1440)"
    )
    return parser.parse_args()

# ════════════════════════════════════════════════════════════
# SECTION 1: ACTIVITY & RHYTHM
# ════════════════════════════════════════════════════════════

def plot_activity(df, colors, outdir, your_name, other_name):
    senders   = [your_name, other_name]
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # A1 — Monthly message volume per person
    monthly = df.groupby(["month", "sender"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(columns=senders, fill_value=0)
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    for s in senders:
        ax.plot(monthly.index, monthly[s], label=s,
                color=colors.get(s, "#888"), marker="o", markersize=3)
    ax.set_title("Monthly message volume per person", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Messages")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A1_monthly_volume.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A2 — Year-on-year totals side by side
    yearly = (df.groupby(["year", "sender"]).size()
              .unstack(fill_value=0)
              .reindex(columns=senders, fill_value=0))
    x     = np.arange(len(yearly.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(yearly) * 1.5), 5))
    ax.bar(x - width / 2, yearly[senders[0]], width,
           label=senders[0], color=colors.get(senders[0], "#888"))
    ax.bar(x + width / 2, yearly[senders[1]], width,
           label=senders[1], color=colors.get(senders[1], "#888"))
    ax.set_xticks(x); ax.set_xticklabels(yearly.index)
    ax.set_title("Year-on-year message totals", fontweight="bold")
    ax.set_ylabel("Messages"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A2_yearly_messages.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A3 — Monthly talk ratio (stacked %)
    mc = (df.groupby(["month", "sender"]).size()
          .unstack(fill_value=0)
          .reindex(columns=senders, fill_value=0))
    mc.index = mc.index.to_timestamp()
    total = mc.sum(axis=1).replace(0, np.nan)
    ratio = mc.div(total, axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(ratio.index, ratio[senders[0]], ratio[senders[1]],
                 labels=senders,
                 colors=[colors.get(senders[0], "#888"), colors.get(senders[1], "#ccc")],
                 alpha=0.8)
    ax.axhline(50, color="white", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 100); ax.set_ylabel("Share of messages (%)")
    ax.set_title("Monthly talk ratio", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right"); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{outdir}/A3_talk_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A4 — Day × hour heatmaps, one per person
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    for ax, s in zip(axes, senders):
        sub = df[df["sender"] == s]
        hm  = sub.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        hm  = hm.reindex([d for d in day_order if d in hm.index])
        im  = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
        ax.set_title(f"{s} — activity heatmap", fontweight="bold")
        ax.set_xlabel("Hour of day")
        plt.colorbar(im, ax=ax, label="Messages")
    plt.tight_layout()
    plt.savefig(f"{outdir}/A4_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()

    # A5 — Messages by day of week
    dow = (df.groupby(["weekday", "sender"]).size()
           .unstack(fill_value=0)
           .reindex(columns=senders, fill_value=0))
    dow = dow.reindex([d for d in day_order if d in dow.index])
    x   = np.arange(len(dow))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, dow[senders[0]], 0.4, label=senders[0], color=colors.get(senders[0], "#888"))
    ax.bar(x + 0.2, dow[senders[1]], 0.4, label=senders[1], color=colors.get(senders[1], "#888"))
    ax.set_xticks(x); ax.set_xticklabels(dow.index, rotation=30, ha="right")
    ax.set_title("Messages by day of week", fontweight="bold")
    ax.set_ylabel("Messages"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/A5_day_of_week.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Activity charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 2: RESPONSE TIMES
# ════════════════════════════════════════════════════════════

def plot_response_times(df, colors, outdir, your_name, other_name, max_gap_minutes=1440):
    rdf, excluded = compute_response_times(df, max_gap_minutes=max_gap_minutes, return_excluded=True)

    if rdf.empty:
        print("⚠️  Not enough alternating messages to compute response times.")
        return

    print(f"  {len(rdf)} responses analysed, {excluded} excluded for exceeding the {max_gap_minutes}-minute cap.")

    senders = [your_name, other_name]

    # R1 — Distribution (clipped at 95th percentile for readability)
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in senders:
        sub = rdf[rdf["responder"] == s]["response_min"]
        if sub.empty:
            continue
        ax.hist(sub.clip(upper=sub.quantile(0.95)), bins=40,
                alpha=0.6, label=s, color=colors.get(s, "#888"))
        med = sub.median()
        ax.axvline(med, color=colors.get(s, "#888"), linestyle="--", linewidth=1.5,
                   label=f"{s} median: {med:.0f} min")
    ax.set_xlabel("Response time (minutes)")
    ax.set_title("Response time distribution (clipped at 95th percentile)", fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/R1_response_time_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # R2 — Median response time per month
    mrt = (rdf.groupby(["month", "responder"])["response_min"]
           .median().unstack(fill_value=np.nan))
    mrt.index = mrt.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    for s in senders:
        if s in mrt.columns:
            ax.plot(mrt.index, mrt[s], label=s, color=colors.get(s, "#888"),
                    marker="o", markersize=3)
    ax.set_title("Median response time per month", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Median response (minutes)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/R2_response_time_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Response time charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 3: CONVERSATION DYNAMICS
# ════════════════════════════════════════════════════════════

def plot_conversation_dynamics(df, colors, outdir, your_name, other_name, session_gap_minutes):
    df_s    = df.sort_values("timestamp").reset_index(drop=True)
    senders = [your_name, other_name]

    gaps = df_s["timestamp"].diff().dt.total_seconds().div(60).fillna(999)
    df_s["session_id"] = (gaps > session_gap_minutes).cumsum()

    # D1 — Who initiates conversations?
    initiators  = df_s.groupby("session_id").first()["sender"]
    init_counts = initiators.value_counts().reindex(senders, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(init_counts.index, init_counts.values,
           color=[colors.get(s, "#888") for s in init_counts.index])
    ax.set_title("Who initiates conversations?", fontweight="bold")
    ax.set_ylabel("Sessions started")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D1_conversation_initiator.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D2 — Average session length per month
    sl = (df_s.groupby(["session_id", "month"]).size()
          .reset_index(name="msgs")
          .groupby("month")["msgs"].mean())
    sl.index = sl.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sl.index, sl.values, color="#01696f", marker="o", markersize=3)
    ax.set_title("Average messages per conversation session (monthly)", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Messages per session")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D2_session_length.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D3 — Burst length (consecutive messages before the other replies)
    bursts = []
    cur_s  = df_s.loc[0, "sender"]; count = 1
    for i in range(1, len(df_s)):
        s = df_s.loc[i, "sender"]
        if s == cur_s:
            count += 1
        else:
            bursts.append({"sender": cur_s, "burst": count})
            cur_s = s; count = 1
    bursts.append({"sender": cur_s, "burst": count})
    burst_mean = (pd.DataFrame(bursts).groupby("sender")["burst"]
                  .mean().reindex(senders, fill_value=0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(burst_mean.index, burst_mean.values,
           color=[colors.get(s, "#888") for s in burst_mean.index])
    ax.set_title("Average messages per turn before the other replies", fontweight="bold")
    ax.set_ylabel("Average burst length")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D3_burst_length.png", dpi=150, bbox_inches="tight")
    plt.close()

    # D4 — Top 10 longest silences
    df_s["gap_hours"] = df_s["timestamp"].diff().dt.total_seconds().div(3600)
    top_gaps = df_s.nlargest(10, "gap_hours")[["timestamp", "gap_hours"]].copy()
    top_gaps["label"] = top_gaps["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_gaps["label"].iloc[::-1], top_gaps["gap_hours"].iloc[::-1], color="#a13544")
    ax.set_xlabel("Gap (hours)")
    ax.set_title("Top 10 longest silences in the chat", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/D4_longest_silences.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Conversation dynamics charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 4: MESSAGING STYLE
# ════════════════════════════════════════════════════════════

def plot_messaging_style(df, colors, outdir, your_name, other_name):
    senders = [your_name, other_name]

    # S1 — Average message length trend
    ml = (df.groupby(["month", "sender"])["msg_len"]
          .mean().unstack(fill_value=np.nan)
          .reindex(columns=senders))
    ml.index = ml.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    for s in senders:
        if s in ml.columns:
            ax.plot(ml.index, ml[s], label=s,
                    color=colors.get(s, "#888"), marker="o", markersize=3)
    ax.set_title("Average message length over time", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Characters")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S1_message_length_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # S2 — Overall question rate per person
    df["is_question"] = df["text"].str.contains(r"\?", na=False)
    q_rate = (df.groupby("sender")["is_question"].mean() * 100).reindex(senders, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(q_rate.index, q_rate.values,
           color=[colors.get(s, "#888") for s in q_rate.index])
    ax.set_title("Question rate per person (% of messages containing '?')", fontweight="bold")
    ax.set_ylabel("% questions")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S2_question_rate.png", dpi=150, bbox_inches="tight")
    plt.close()

    # S3 — Vocabulary overlap (pie chart)
    tok = {}
    for s in senders:
        tokens = []
        for text in df[df["sender"] == s]["text"]:
            tokens.extend(get_tokens(text))
        tok[s] = set(tokens)
    if len(tok) == 2:
        v0, v1  = tok[senders[0]], tok[senders[1]]
        shared  = v0 & v1
        sizes   = [len(v0 - v1), len(shared), len(v1 - v0)]
        labels  = [f"{senders[0]} only", "Shared", f"{senders[1]} only"]
        wcolors = [colors.get(senders[0], "#888"), "#aaaaaa", colors.get(senders[1], "#888")]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(sizes, labels=labels, colors=wcolors, autopct="%1.1f%%",
               startangle=140, explode=(0.05, 0.1, 0.05))
        ax.set_title("Vocabulary overlap between speakers", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{outdir}/S3_vocabulary_overlap.png", dpi=150, bbox_inches="tight")
        plt.close()

    # S4 — Question rate trend over time
    mq = (df.groupby(["month", "sender"])["is_question"]
          .mean().unstack(fill_value=np.nan)
          .reindex(columns=senders) * 100)
    mq.index = mq.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    for s in senders:
        if s in mq.columns:
            ax.plot(mq.index, mq[s], label=s,
                    color=colors.get(s, "#888"), marker="o", markersize=3)
    ax.set_title("Question rate trend over time", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("% questions")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=45, ha="right"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/S4_question_rate_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Messaging style charts saved.")

# ════════════════════════════════════════════════════════════
# SECTION 5: WORD PATTERNS BY YEAR
# ════════════════════════════════════════════════════════════

def plot_word_patterns_by_year(df, colors, outdir, your_name, other_name):
    years   = sorted(df["year"].unique())
    senders = [your_name, other_name]

    for year in years:
        df_year = df[df["year"] == year]

        # W1 — Top 15 words and bigrams per person for this year
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Word patterns — {year}", fontsize=15, fontweight="bold")

        for col_idx, sender in enumerate(senders):
            tokens = []
            for text in df_year[df_year["sender"] == sender]["text"]:
                tokens.extend(get_tokens(text))
            color = colors.get(sender, "#888")

            ax_w = axes[0][col_idx]
            if tokens:
                tw = Counter(tokens).most_common(15)
                w, f = zip(*tw)
                ax_w.barh(list(w)[::-1], list(f)[::-1], color=color, alpha=0.85)
            else:
                ax_w.text(0.5, 0.5, "No data", transform=ax_w.transAxes, ha="center")
            ax_w.set_title(f"{sender} — top 15 words", fontweight="bold")
            ax_w.set_xlabel("Frequency")
            ax_w.spines["top"].set_visible(False); ax_w.spines["right"].set_visible(False)

            ax_b = axes[1][col_idx]
            if len(tokens) >= 2:
                tb = Counter(ngrams(tokens, 2)).most_common(15)
                bl = [" ".join(b) for b, _ in tb]
                bv = [c for _, c in tb]
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
        # FIX: default-argument capture (sender=sender) ensures each cloud
        # uses its own colour and not the last sender's value from the loop.
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Word clouds — {year}", fontsize=14, fontweight="bold")

        for i, sender in enumerate(senders):
            tokens = get_tokens(
                " ".join(df_year[df_year["sender"] == sender]["text"].tolist())
            )
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

        plt.tight_layout()
        plt.savefig(f"{outdir}/W2_wordclouds_{year}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("✅ Word pattern by year charts saved.")

# ════════════════════════════════════════════════════════════
# MAIN — runs all sections in order
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    INPUT_PATTERN = args.input
    OUTPUT_DIR    = args.output
    YOUR_NAME     = args.your_name
    OTHER_NAME    = args.other_name
    TIMEZONE      = args.timezone
    SESSION_GAP   = args.session_gap

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📂 Parsing HTML files...")
    df = parse_all_files(INPUT_PATTERN, timezone_offset=TIMEZONE)

    # Keep only the two named participants
    df = df[df["sender"].isin({YOUR_NAME, OTHER_NAME})].reset_index(drop=True)
    if df.empty:
        print("❌ No messages found. Check that --your-name and --other-name match the export exactly.")
        sys.exit(1)

    senders = sorted(df["sender"].unique().tolist())
    print(f"  Participants: {senders}")
    print(f"  Date range : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Total msgs : {len(df)}\n")

    colors = sender_colors(senders)
    df.to_csv(f"{OUTPUT_DIR}/parsed_messages.csv", index=False)

    print("📊 Section 1: Activity & rhythm...")
    plot_activity(df, colors, OUTPUT_DIR, YOUR_NAME, OTHER_NAME)

    print("⏱  Section 2: Response times...")
    plot_response_times(df, colors, OUTPUT_DIR, YOUR_NAME, OTHER_NAME, args.max_response_gap)

    print("💬 Section 3: Conversation dynamics...")
    plot_conversation_dynamics(df, colors, OUTPUT_DIR, YOUR_NAME, OTHER_NAME, SESSION_GAP)

    print("✍️  Section 4: Messaging style...")
    plot_messaging_style(df, colors, OUTPUT_DIR, YOUR_NAME, OTHER_NAME)

    print("📅 Section 5: Word patterns by year...")
    plot_word_patterns_by_year(df, colors, OUTPUT_DIR, YOUR_NAME, OTHER_NAME)

    print(f"\n🎉 Done! All charts saved to '{OUTPUT_DIR}/'")

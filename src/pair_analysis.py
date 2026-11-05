from pathlib import Path
from hashlib import sha256
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from IPython.display import display

BLUE, RED, GREY = "#1E67B5", "#D02F2F", "#7F7F7F"
DAY = pd.Timedelta(days=1)

LINK_SHA256 = {
    "body": "9bbf57511dc0da7369c2e896d865b68f170136def4f223f4c9e9e10818f4bfb5",
    "title": "b6ea49dcbc83f948b177b341cace8c37fb8f12fc1b0e18bbfd1ba4630b699893",
}


def ensure_link_data(data_dir=Path("data/raw"), base_url="https://snap.stanford.edu/data",
                     checksums=LINK_SHA256):
    """Download missing SNAP TSV files into data/raw and verify their SHA-256 hashes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for location in ("body", "title"):
        name = f"soc-redditHyperlinks-{location}.tsv"
        path = data_dir / name
        if path.is_file():
            continue

        url = f"{base_url.rstrip('/')}/{name}"
        temporary = data_dir / f".{name}.part"
        print(f"Downloading {name} from {url}")
        try:
            digest = sha256()
            with urlopen(url) as response, temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != checksums[location]:
                raise ValueError(f"Checksum mismatch for {name}; check the URL and checksum")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return data_dir


def load_links(data_dir=Path("data/raw")):
    """Read title and body TSV files with explicit column types; return one link table."""
    data_dir = ensure_link_data(data_dir)
    columns = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT"]
    files = {"body": data_dir / "soc-redditHyperlinks-body.tsv",
             "title": data_dir / "soc-redditHyperlinks-title.tsv"}
    parts = []
    for location, path in files.items():
        part = pd.read_csv(path, sep="	", usecols=columns,
                           dtype={"SOURCE_SUBREDDIT": "string", "TARGET_SUBREDDIT": "string",
                                  "POST_ID": "string", "TIMESTAMP": "string", "LINK_SENTIMENT": "int8"})
        part["LOCATION"] = location
        parts.append(part)
    links = pd.concat(parts, ignore_index=True)
    print(f"Loaded {len(links):,} links: {len(parts[0]):,} body and {len(parts[1]):,} title")
    return links

def prepare_links(links):
    """Parse timestamps, remove self-links, sort links, and add unordered pair keys."""
    # Parse dates before ordering links within each pair.
    links["time"] = pd.to_datetime(links.pop("TIMESTAMP"), errors="coerce")
    assert links["time"].notna().all()
    links = links.loc[links["SOURCE_SUBREDDIT"] != links["TARGET_SUBREDDIT"]].copy()
    links = links.sort_values(["time", "POST_ID", "LOCATION"]).reset_index(drop=True)

    # One unordered key identifies a relationship; source and target preserve its direction.
    links["pair_a"] = np.minimum(links["SOURCE_SUBREDDIT"], links["TARGET_SUBREDDIT"]).astype("string")
    links["pair_b"] = np.maximum(links["SOURCE_SUBREDDIT"], links["TARGET_SUBREDDIT"]).astype("string")
    links["LOCATION"] = links["LOCATION"].astype("category")
    assert not any(dtype == "object" for dtype in links.dtypes)
    links.info()
    print(f"{len(links):,} usable links in {links.groupby(['pair_a', 'pair_b']).ngroups:,} pairs")
    print(f"Observed period: {links['time'].min()} to {links['time'].max()}")
    return links

def show_monthly_baseline(links):
    """Plot monthly counts for both sentiments and print comparable monthly growth rates."""
    # Count links by month and sentiment; omit the partial first month.
    monthly = (links.assign(month=links["time"].dt.to_period("M").dt.to_timestamp())
               .groupby(["month", "LINK_SENTIMENT"]).size().unstack(fill_value=0))
    monthly = monthly.loc[monthly.index >= "2014-01-01"].rename(
        columns={1: "Non-negative", -1: "Negative"}).reset_index()
    monthly["month_number"] = np.arange(len(monthly))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=monthly, x="month_number", y="Non-negative", ax=ax, color="#1E67B5",
                ci=95, scatter_kws={"s": 38, "alpha": 0.8}, line_kws={"linewidth": 2})
    sns.regplot(data=monthly, x="month_number", y="Negative", ax=ax, color="#D02F2F",
                ci=95, scatter_kws={"s": 38, "alpha": 0.8}, line_kws={"linewidth": 2})
    tick_positions = np.arange(0, len(monthly), 6)
    ax.set_xticks(tick_positions, monthly.loc[tick_positions, "month"].dt.strftime("%Y-%m"), rotation=45)
    ax.set(title="Monthly link counts and linear trends", xlabel="Month",
           ylabel="Links per month", ylim=(0, None))
    ax.legend(handles=ax.lines[:2], labels=["Non-negative", "Negative"], title="Sentiment")
    fig.tight_layout()
    plt.show()

    # Log-linear fits express the typical monthly change on a comparable percentage scale.
    monthly["Negative share"] = monthly["Negative"] / (monthly["Negative"] + monthly["Non-negative"])
    for label in ["Non-negative", "Negative", "Negative share"]:
        log_slope = np.polyfit(monthly["month_number"], np.log(monthly[label]), 1)[0]
        monthly_growth = np.expm1(log_slope)
        print(f"{label}: {monthly_growth:+.2%} per month")

    totals = links["LINK_SENTIMENT"].value_counts().rename(index={1: "Non-negative", -1: "Negative"})
    display(pd.DataFrame({"links": totals, "share": totals / totals.sum()}).style.format({"share": "{:.2%}"}))

def select_second_links(links):
    """Keep each pair’s second observed link when its first was non-negative and follow-up is complete."""
    # Identify the first and second recorded links of each unordered pair.
    pair = links.groupby(["pair_a", "pair_b"], sort=False)
    links["position_in_pair"] = pair.cumcount()
    links["first_sentiment"] = pair["LINK_SENTIMENT"].transform("first")
    links["first_source"] = pair["SOURCE_SUBREDDIT"].transform("first")

    follow_up = pd.Timedelta(30, unit="D")
    # Keep the second link if the first was non-negative and 30 days remain.
    events = links.loc[
        links["position_in_pair"].eq(1)
        & links["first_sentiment"].eq(1)
        & links["time"].le(links["time"].max() - follow_up)
    ].copy()
    events["negative_event"] = events["LINK_SENTIMENT"].eq(-1)
    # With one prior link, a different first source means that link ran in reverse.
    events["prior_reverse"] = events["SOURCE_SUBREDDIT"].ne(events["first_source"]).astype(bool)
    events["group"] = events["negative_event"].map({False: "Non-negative", True: "Negative"})
    event_mix = events.groupby("group").agg(
        pairs=("POST_ID", "size"), opposite_direction_share=("prior_reverse", "mean"))
    event_mix["same_direction_share"] = 1 - event_mix["opposite_direction_share"]
    display(event_mix[["pairs", "same_direction_share", "opposite_direction_share"]]
            .style.format({"same_direction_share": "{:.2%}",
                           "opposite_direction_share": "{:.2%}"}))
    return links, events, follow_up

def measure_pair_responses(links, events, follow_up):
    """Find the next pair link and the first reverse-direction link within 30 days."""
    pair = links.groupby(["pair_a", "pair_b"], sort=False)
    # The next link between the pair may run in either direction.
    links["next_pair_time"] = pair["time"].shift(-1)
    events["next_pair_time"] = links.loc[events.index, "next_pair_time"].to_numpy()
    events["another_link_30d"] = (events["next_pair_time"] - events["time"]).le(follow_up)

    # Swap source and target on the right to find the next link in the reverse direction.
    reverse_links = links[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "time", "LINK_SENTIMENT"]].rename(
        columns={"SOURCE_SUBREDDIT": "TARGET_SUBREDDIT",
                 "TARGET_SUBREDDIT": "SOURCE_SUBREDDIT",
                 "time": "reverse_time", "LINK_SENTIMENT": "reverse_sentiment"})
    event_lookup = events[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "time"]].copy()
    event_lookup["event_id"] = event_lookup.index
    matched = pd.merge_asof(
        event_lookup.sort_values("time"),
        reverse_links.sort_values("reverse_time"),
        left_on="time", right_on="reverse_time",
        by=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"],
        direction="forward", allow_exact_matches=False,
    )
    matched = matched.set_index("event_id")
    events["reverse_time"] = matched["reverse_time"]
    events["reverse_sentiment"] = matched["reverse_sentiment"]
    events["reverse_30d"] = (events["reverse_time"] - events["time"]).le(follow_up)
    events["negative_reverse_30d"] = events["reverse_30d"] & events["reverse_sentiment"].eq(-1)
    return links, events

def show_response_by_direction(events, links):
    """Compare response rates after same-direction and opposite-direction first links."""
    # Compare outcomes by the direction of the first observed link.
    events["first_link_direction"] = events["prior_reverse"].map(
        {False: "Same direction", True: "Opposite direction"})
    by_direction = events.groupby(["first_link_direction", "group"], observed=True).agg(
        pairs=("POST_ID", "size"), reverse_30d=("reverse_30d", "sum"),
        negative_reverse_30d=("negative_reverse_30d", "sum"))
    by_direction["reverse_rate"] = by_direction["reverse_30d"] / by_direction["pairs"]
    by_direction["negative_given_reverse"] = (
        by_direction["negative_reverse_30d"] / by_direction["reverse_30d"])
    display(by_direction[["reverse_rate", "negative_given_reverse", "pairs", "reverse_30d"]]
            .style.format({"reverse_rate": "{:.2%}", "negative_given_reverse": "{:.2%}"}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    plot_data = by_direction.reset_index()
    for ax, column, title, ylabel in [
        (axes[0], "reverse_rate", "Reverse link by first-link direction", "All pairs (%)"),
        (axes[1], "negative_given_reverse", "Negative among reverse links", "Responding pairs (%)")
    ]:
        sns.barplot(data=plot_data, x="first_link_direction", y=column, hue="group",
                    hue_order=["Non-negative", "Negative"], palette=["#1E67B5", "#D02F2F"], ax=ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set(title=title, xlabel="First link relative to second", ylabel=ylabel)
        ax.legend_.remove()
    baseline = links["LINK_SENTIMENT"].eq(-1).mean()
    axes[1].axhline(baseline, color="gray", linestyle="--", linewidth=1,
                    label=f"All links: {baseline:.2%} negative")
    axes[1].legend(title="Second link / benchmark", loc="upper left",
                   bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    plt.show()

def show_pooled_responses(events, links):
    """Summarize 30-day link and reverse-link rates across both first-link directions."""
    # Compare rates because the two event groups have different sizes.
    response = events.groupby("group").agg(
        pairs=("POST_ID", "size"), another_link_30d=("another_link_30d", "sum"),
        reverse_30d=("reverse_30d", "sum"), negative_reverse_30d=("negative_reverse_30d", "sum"),
    ).reindex(["Non-negative", "Negative"])
    response["another_link_rate"] = response["another_link_30d"] / response["pairs"]
    response["reverse_rate"] = response["reverse_30d"] / response["pairs"]
    response["negative_given_reverse"] = response["negative_reverse_30d"] / response["reverse_30d"]
    display(response[["another_link_rate", "reverse_rate", "negative_given_reverse"]].style.format(
        {"another_link_rate": "{:.2%}", "reverse_rate": "{:.2%}",
         "negative_given_reverse": "{:.2%}"}))

    # Wilson intervals convey the greater uncertainty among negative-event responders.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    comparisons = [("reverse_30d", "pairs", "Reverse link within 30 days", "All selected pairs (%)"),
                   ("negative_reverse_30d", "reverse_30d", "Negative first reverse link", "Pairs with a reverse link (%)")]
    for ax, (numerator, denominator, title, ylabel) in zip(axes, comparisons):
        successes, totals = response[numerator].to_numpy(), response[denominator].to_numpy()
        rates = successes / totals
        lower, upper = proportion_confint(successes, totals, method="wilson")
        sns.barplot(x=response.index, y=rates * 100, hue=response.index, legend=False,
                    palette={"Non-negative": "#1E67B5", "Negative": "#D02F2F"}, ax=ax)
        ax.errorbar(range(2), rates * 100, yerr=[(rates - lower) * 100, (upper - rates) * 100],
                    fmt="none", color="black", capsize=4)
        ax.set(title=title, xlabel="Second link", ylabel=ylabel)
        if numerator == "negative_reverse_30d":
            baseline = links["LINK_SENTIMENT"].eq(-1).mean() * 100
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                       label=f"All links: {baseline:.2f}% negative")
            ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show()

def show_window_sensitivity(events):
    """Recompute reverse-link outcomes for every follow-up window from 1 to 30 days."""
    # Use the same eligible pairs for every window; all have at least 30 days of coverage.
    windows = []
    for days in range(1, 31):
        reverse = (events["reverse_time"] - events["time"]).le(pd.Timedelta(days, unit="D"))
        for group, subset in events.groupby("group"):
            came_back = reverse.loc[subset.index]
            negative_back = came_back & subset["reverse_sentiment"].eq(-1)
            windows.append({"days": days, "group": group, "pairs": len(subset),
                            "reverse": came_back.sum(), "negative_reverse": negative_back.sum(),
                            "reverse_rate": came_back.mean(),
                            "negative_given_reverse": negative_back.sum() / came_back.sum() if came_back.sum() else np.nan})
    window_table = pd.DataFrame(windows)
    display(window_table.loc[window_table["days"].isin([7, 30])]
            .set_index(["days", "group"])[["reverse_rate", "negative_given_reverse", "pairs", "reverse"]].style.format(
                {"reverse_rate": "{:.2%}", "negative_given_reverse": "{:.2%}"}))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, column, title, ylabel in [
        (axes[0], "reverse_rate", "Reverse link by day", "All pairs (%)"),
        (axes[1], "negative_given_reverse", "Negative among reverse links", "Responding pairs (%)")
    ]:
        sns.lineplot(data=window_table, x="days", y=column, hue="group", ax=ax,
                     palette={"Non-negative": "#1E67B5", "Negative": "#D02F2F"})
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set(title=title, xlabel="Days after selected link", ylabel=ylabel, xlim=(1, 30))
        ax.legend_.remove()
    axes[1].legend(title="Selected link", loc="lower right")
    fig.tight_layout()
    plt.show()

def fit_response_models(events):
    """Fit adjusted logistic models and display odds ratios with clustered confidence intervals."""
    # Allow the sentiment association with a reverse link to differ by prior direction.
    events["negative_link"] = events["negative_event"].astype(int)
    events["prior_reverse_int"] = events["prior_reverse"].astype(int)
    events["year"] = events["time"].dt.year.astype(str)
    events["reverse_outcome"] = events["reverse_30d"].astype(int)
    events["negative_reverse_outcome"] = events["negative_reverse_30d"].astype(int)

    reverse_model = smf.logit(
        "reverse_outcome ~ negative_link * prior_reverse_int + C(year) + C(LOCATION)",
        data=events).fit(disp=False, cov_type="cluster",
                         cov_kwds={"groups": events["TARGET_SUBREDDIT"]})
    responders = events.loc[events["reverse_30d"]]
    sentiment_model = smf.logit(
        "negative_reverse_outcome ~ negative_link + prior_reverse_int + C(year) + C(LOCATION)",
        data=responders).fit(disp=False, cov_type="cluster",
                             cov_kwds={"groups": responders["TARGET_SUBREDDIT"]})

    # The negative-link coefficient is for same-direction pairs; add the interaction for opposite-direction pairs.
    base = reverse_model.params["negative_link"]
    interaction = reverse_model.params["negative_link:prior_reverse_int"]
    cov = reverse_model.cov_params()
    same_se = np.sqrt(cov.loc["negative_link", "negative_link"])
    opposite_se = np.sqrt(cov.loc["negative_link", "negative_link"]
                          + cov.loc["negative_link:prior_reverse_int", "negative_link:prior_reverse_int"]
                          + 2 * cov.loc["negative_link", "negative_link:prior_reverse_int"])
    sentiment_beta = sentiment_model.params["negative_link"]
    sentiment_se = sentiment_model.bse["negative_link"]
    model_rows = [("Reverse link: same direction", base, same_se),
                  ("Reverse link: opposite direction", base + interaction, opposite_se),
                  ("Negative if reversed", sentiment_beta, sentiment_se)]
    model_table = pd.DataFrame([
        {"outcome": label, "odds_ratio": np.exp(beta),
         "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se)}
        for label, beta, se in model_rows]).set_index("outcome")
    display(model_table.style.format({"odds_ratio": "{:.2f}",
                                      "ci_low": "{:.2f}", "ci_high": "{:.2f}"}))
    return events

def show_pair_timeline(events, links, follow_up):
    """Plot one observed pair to illustrate the selected link and first reverse link."""
    returned = events.loc[events["reverse_30d"]].copy()
    example = returned.loc[returned["negative_event"] & returned["negative_reverse_30d"]].sort_values(
        ["time", "pair_a", "pair_b"]).iloc[0]
    example_links = links.loc[
        links["pair_a"].eq(example["pair_a"]) & links["pair_b"].eq(example["pair_b"])
        & links["time"].between(example["time"] - follow_up, example["time"] + follow_up)
    ].copy()
    example_links["days_from_event"] = (
        example_links["time"] - example["time"]).dt.total_seconds() / 86400
    example_links["direction"] = np.where(
        example_links["SOURCE_SUBREDDIT"].eq(example["SOURCE_SUBREDDIT"]), 1, 0)

    # Show one observed sequence, not a representative pair.
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for sentiment, color, label in [(1, "#1E67B5", "Non-negative"),
                                    (-1, "#D02F2F", "Negative")]:
        part = example_links.loc[example_links["LINK_SENTIMENT"].eq(sentiment)]
        sns.scatterplot(x=part["days_from_event"], y=part["direction"], ax=ax,
                   s=65, color=color, label=label, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(-30, 30), ylim=(-0.5, 1.5), xlabel="Days from selected negative link",
           title=f"Example: {example['SOURCE_SUBREDDIT']} and {example['TARGET_SUBREDDIT']}")
    ax.set_yticks([0, 1], [f"{example['TARGET_SUBREDDIT']} → {example['SOURCE_SUBREDDIT']}",
                            f"{example['SOURCE_SUBREDDIT']} → {example['TARGET_SUBREDDIT']}"])
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    plt.show()

def show_mutual_negativity(links):
    """Classify active reciprocal pairs by whether negativity occurs in neither, one, or both directions."""
    # Summarize the entire observed history in each direction.
    directions = links.groupby(["pair_a", "pair_b", "SOURCE_SUBREDDIT"]).agg(
        links=("LINK_SENTIMENT", "size"),
        negative_links=("LINK_SENTIMENT", lambda x: x.eq(-1).sum()),
    ).reset_index()
    directions["negative_share"] = directions["negative_links"] / directions["links"]
    active = directions.loc[directions["links"].ge(5)].copy()
    active["negative_direction"] = active["negative_share"].ge(0.20)
    pairs_both = active.groupby(["pair_a", "pair_b"]).filter(lambda x: len(x) == 2)
    pair_types = pairs_both.groupby(["pair_a", "pair_b"])["negative_direction"].sum().map(
        {0: "Neither direction", 1: "One direction", 2: "Both directions"})
    counts = pair_types.value_counts().reindex(
        ["Neither direction", "One direction", "Both directions"], fill_value=0)
    print(f"Active reciprocal pairs: {len(pair_types):,}")
    display(pd.DataFrame({"pairs": counts, "share": counts / counts.sum()}).style.format({"share": "{:.2%}"}))

    # Compare link shares on the same unit: links in active reciprocal pairs versus all links.
    active_negative_share = pairs_both["negative_links"].sum() / pairs_both["links"].sum()
    overall_negative_share = links["LINK_SENTIMENT"].eq(-1).mean()
    print(f"Negative links in active reciprocal pairs: {active_negative_share:.2%}")
    print(f"Negative links in the full dataset: {overall_negative_share:.2%}")


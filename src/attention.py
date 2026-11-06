from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from IPython.display import display

BLUE, RED, GREY = "#1E67B5", "#D02F2F", "#7F7F7F"
DAY = pd.Timedelta(days=1)

MIN_LINKS = 4
WINDOWS = {"elig": (-180, -90), "pre90": (-90, 0), "pre30": (-30, 0), "post30": (1, 31)}

def select_first_attacks(links):
    """Find each target’s first observed negative incoming link and require full follow-up."""
    # Sort then keep the first row per group = the first event of each entity.
    negatives = links[links.LINK_SENTIMENT == -1]
    first_neg_time = negatives.groupby("TARGET_SUBREDDIT").time.min()

    events = negatives.drop_duplicates("TARGET_SUBREDDIT", keep="first")   # links is sorted: keeps the earliest
    events = events.rename(columns={"SOURCE_SUBREDDIT": "A", "TARGET_SUBREDDIT": "B", "time": "t0"})
    events = events[["A", "B", "t0"]]
    n_attacked_ever = len(events)

    events = events[events.t0 >= "2015-01-01"]
    n_from_2015 = len(events)

    events = events[events.t0 + 31 * DAY <= links.time.max()].reset_index(drop=True)   # full 30 days of follow-up
    events["event_id"] = events.index
    print(f"{n_attacked_ever:,} attacked subreddits -> {n_from_2015:,} first attacks from 2015 -> {len(events):,} with 30 days of follow-up")
    return events, first_neg_time, n_attacked_ever, n_from_2015

def incoming_links(events, links, lo=-180, hi=365):
    "Make a table of the incomming link to the attacked subredits in the defined window and from third parties"
    m = events.merge(links, left_on="B", right_on="TARGET_SUBREDDIT", how="left")
    m["rel_day"] = (m.time - m.t0) // DAY
    m["C"] = m.SOURCE_SUBREDDIT
    m["is_neg"] = m.LINK_SENTIMENT == -1

    in_window = (m.rel_day >= lo) & (m.rel_day <= hi)
    from_third = m.C != m.A
    m = m[in_window & from_third]

    return m[["event_id", "C", "rel_day", "is_neg"]].reset_index(drop=True)


def add_counts(table, long, windows=WINDOWS):
    """For each window, add the number of incoming links (<name>_n) and of negative ones (<name>_neg)."""
    table = table.copy()

    for name, (lo, hi) in windows.items():
        w = long[(long.rel_day >= lo) & (long.rel_day < hi)]
        groupped = w.groupby("event_id")

        n_tot = groupped["C"].size()
        n_neg = groupped["is_neg"].sum()

        counts = pd.DataFrame({f"{name}_n": n_tot, f"{name}_neg": n_neg})

        table = table.join(counts, on="event_id")
    
    count_cols = [c for c in table if c.endswith(("_n", "_neg"))]
    table[count_cols] = table[count_cols].fillna(0).astype(int)    # no link at all means 0, not NaN
    return table


def show_attack_funnel(events, n_attacked_ever, n_from_2015):
    """Show how the attack-event sample narrows and when events occur."""
    # Visual 1: from all attacked subreddits down to the eligible ones.
    funnel = pd.Series({
        "B attacked at least once": n_attacked_ever,
        "first attack in 2015 or later": n_from_2015,
        "with 30 days of follow-up": len(events),
        f"eligible (≥ {MIN_LINKS} links in [-180,-90) days)": events.eligible.sum(),
    })
    ax = funnel[::-1].plot.barh(color=BLUE, figsize=(9, 3))     # [::-1] puts the first step on top
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=3)
    ax.set(title="How many attacks can we analyse?", xlabel="Number of subreddits B")
    plt.tight_layout(); plt.show()

    # Visual 2: first attacks per month, to spot shocks.
    monthly = events.t0.dt.to_period("M").value_counts().sort_index()
    ax = monthly.plot(kind="bar", color=BLUE, figsize=(11, 3))
    ax.set(title="First attacks per month", xlabel="", ylabel="Attacks")
    plt.tight_layout(); plt.show()

def match_controls(events, links, first_neg_time):
    """Match each eligible target to an unattacked target with similar earlier attention."""
    max_d = 0.5                           # maximum distance between B and its control (log units)
    link_times = links["time"].to_numpy()   # sorted array: lets us slice by date with a binary search

    def find_control(event):
        """Return (best control subreddit, distance) for one event, or None."""
        start = np.searchsorted(link_times, (event.t0 - 180 * DAY).to_datetime64())
        stop = np.searchsorted(link_times, event.t0.to_datetime64())
        w = links.iloc[start:stop]
        w = w[w.SOURCE_SUBREDDIT != event.A]                                # same rule as for B: ignore links from A

        # Incoming links of every subreddit in the two windows (elig = more than 90 days before t0, pre90 = the rest).
        period = np.where((event.t0 - w.time) / DAY > 90, "elig", "pre90")
        counts = (w.assign(period=period).groupby(["TARGET_SUBREDDIT", "period"]).size()
                    .unstack(fill_value=0).reindex(columns=["elig", "pre90"], fill_value=0))

        cand = counts[counts.elig >= MIN_LINKS].drop(index=[event.B, event.A], errors="ignore")   # active enough, not B or A
        first_neg = first_neg_time.reindex(cand.index)
        cand = cand[(first_neg.isna() | (first_neg >= event.t0 + DAY)).to_numpy()]            # not attacked yet at t0
        if cand.empty:
            return None

        dist = ((np.log1p(cand.elig) - np.log1p(event.elig_n)).abs()
                + (np.log1p(cand.pre90) - np.log1p(event.pre90_n)).abs())
        return dist.idxmin(), dist.min()

    rows = []
    for ev in events[events.eligible].itertuples():
        res = find_control(ev)
        if res is not None:
            rows.append({"event_id": ev.event_id, "control": res[0], "dist": res[1]})
    controls = pd.DataFrame(rows)
    controls = controls[controls.dist <= max_d]
    print(f"{len(controls)} / {events.eligible.sum()} events matched | {controls.control.nunique()} distinct controls")
    return controls

def weekly_table(long, ids, weeks):
    """Rows = events, columns = weeks around the attack, values = number of incoming links."""
    w = long[long.event_id.isin(ids)].copy()
    w["week"] = np.floor(w.rel_day / 7).astype(int)                  # week -1 = days [-7, 0), week 0 = days [0, 7)
    tab = w.groupby(["event_id", "week"]).size().unstack(fill_value=0)
    return tab.reindex(index=ids, columns=list(weeks), fill_value=0)  # weeks with no link become 0

def bootstrap_mean(tab, n_boot=1000, seed=0):
    """Mean per week with a 95% interval, resampling EVENTS (rows) with replacement."""
    rng = np.random.default_rng(seed)
    X = tab.to_numpy()
    means = np.array([X[rng.integers(0, len(X), len(X))].mean(axis=0) for _ in range(n_boot)])
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    return X.mean(axis=0), low, high

def plot_weekly(ids, weeks, title, long, ctrl_long, attack_line=False):
    """Compare weekly incoming links for attacked targets and matched controls."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, lg, color in [("B (attacked)", long, RED), ("B' (control)", ctrl_long, BLUE)]:
        mean, low, high = bootstrap_mean(weekly_table(lg, ids, weeks))
        ax.plot(list(weeks), mean, color=color, marker="o", label=label)
        ax.fill_between(list(weeks), low, high, color=color, alpha=0.2)
    if attack_line:
        ax.axvline(-0.5, ls="--", color="k")        # the attack happens at the start of week 0
    ax.set(title=title, xlabel="Weeks since the attack", ylabel="Incoming links per week (mean)")
    ax.legend(); plt.tight_layout(); plt.show()

def boot_ratios(ev, ct, n_boot=1000, seed=41):
    """Bootstrap before-to-after link ratios and their ratio between targets and controls."""

    a = [ev.post30_n, ev.pre30_n, ct.post30_n, ct.pre30_n]
    a = [x.to_numpy(float) for x in a]
    n = len(a[0])

    pick = np.random.default_rng(seed).integers(0, n, (n_boot, n))

    def stats(post_b, pre_b, post_c, pre_c):
        rb, rc = post_b / pre_b, post_c / pre_c
        return np.array([rb, rc, rb / rc])

    point = stats(*[x.sum() for x in a])
    boot = stats(*[x[pick].sum(axis=1) for x in a])
    low, high = np.percentile(boot, [2.5, 97.5], axis=1)


    return pd.DataFrame({"estimate": point, "low": low, "high": high},
                        index=["B: after / before", "B': after / before", "B / B' (ratio of ratios)"])


def share_ci(table, num, den, n_boot=1000, seed=0):
    """Ratio sum(num) / sum(den) with a 95% interval, resampling EVENTS with replacement.
    `table` needs an `event_id` column; any number of rows per event is fine."""
    per_event = table.groupby("event_id")[[num, den]].sum()
    a, b = per_event[num].to_numpy(float), per_event[den].to_numpy(float)
    pick = np.random.default_rng(seed).integers(0, len(per_event), (n_boot, len(per_event)))
    boot = a[pick].sum(axis=1) / b[pick].sum(axis=1)
    return a.sum() / b.sum(), *np.percentile(boot, [2.5, 97.5])

def bar_with_ci(ax, positions, d, color, label, width=0.35):
    """Draw one bar and its interval from a share summary."""
    ax.bar(positions, d.share.iloc[0], width, color=color, label=label)
    ax.errorbar(positions, d.share.iloc[0], yerr=[[d.share.iloc[0] - d.low.iloc[0]], [d.high.iloc[0] - d.share.iloc[0]]], fmt="none", color="black", capsize=4)


def boot_share_diff(ev, ct, num, den, n_boot=1000, seed=0):
    """Difference between the negative share of B and of its control (B minus B'), with a 95% interval and
    a p-value. Events are resampled together (same indices for B and B'), since each B is paired with its B'."""
    a_num, a_den = ev[num].to_numpy(float), ev[den].to_numpy(float)
    c_num, c_den = ct[num].to_numpy(float), ct[den].to_numpy(float)
    diff = lambda i: a_num[i].sum() / a_den[i].sum() - c_num[i].sum() / c_den[i].sum()
    rng = np.random.default_rng(seed)
    n = len(ev)
    draws = np.array([diff(rng.integers(0, n, n)) for _ in range(n_boot)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return pd.Series({"difference": diff(np.arange(n)), "low": low, "high": high, "p_value": (np.abs(draws - diff(np.arange(n))) >= abs(diff(np.arange(n)))).mean()})


def make_link_lookups(links):
    """Index links by source and target for repeated event-window searches."""
    # One table of links per target and one per source: a fast way to look up all the links of a given subreddit.
    links_by_target = {
        target: group 
        for target, group in links.groupby("TARGET_SUBREDDIT")
    }
    links_by_source = {
        source: group 
        for source, group in links.groupby("SOURCE_SUBREDDIT")
    }
    return links_by_target, links_by_source

def find_triangles_fast(A, B, t0, links_by_target, links_by_source, window=30, lookback=90):
    """Find post-attack third-party links and label prior ties to the attacker."""
    # Event rows can carry NumPy datetime64 values; use a pandas timestamp for day arithmetic.
    t0 = pd.Timestamp(t0)
    start = t0 - pd.Timedelta(days=lookback)

    # 1. Find C → B links after the attack
    target_links = links_by_target.get(B)

    if target_links is None:
        return None

    after = target_links[
        (target_links["time"] > t0) &
        (target_links["time"] <= t0 + pd.Timedelta(days=window)) &
        (target_links["SOURCE_SUBREDDIT"] != A) &
        (target_links["SOURCE_SUBREDDIT"] != B)
    ].copy()

    if after.empty:
        return None

    # 2. Find positive connections A ↔ C before the attack
    outgoing = links_by_source.get(A)
    incoming = links_by_target.get(A)

    positive_neighbors = set()

    if outgoing is not None:
        positive_neighbors.update(
            outgoing.loc[
                (outgoing["time"] >= start) &
                (outgoing["time"] < t0) &
                (outgoing["LINK_SENTIMENT"] == 1),
                "TARGET_SUBREDDIT"
            ]
        )

    if incoming is not None:
        positive_neighbors.update(
            incoming.loc[
                (incoming["time"] >= start) &
                (incoming["time"] < t0) &
                (incoming["LINK_SENTIMENT"] == 1),
                "SOURCE_SUBREDDIT"
            ]
        )

    # 3. Find previous connections C → B
    previous_CB = target_links[
        (target_links["time"] >= start) &
        (target_links["time"] < t0)
    ]

    previous_sources = set(
        previous_CB["SOURCE_SUBREDDIT"]
    )

    # 4. Add information to the dataframe
    after["A"] = A
    after["B"] = B
    after["C"] = after["SOURCE_SUBREDDIT"]
    after["t0"] = t0

    after["positive_AC"] = (
        after["C"].isin(positive_neighbors)
    )

    after["previous_CB"] = (
        after["C"].isin(previous_sources)
    )

    after["negative_CB"] = (
        after["LINK_SENTIMENT"] == -1
    )

    return after

def run_triangles(table, links_by_target, links_by_source):
    """Apply the triangle search to each event and concatenate its third-party links."""
    # table: one row per event, with columns A, B and t0 (events or controls)
    all_results = []

    for event in table.itertuples(index=False):

        result = find_triangles_fast(
            A=event.A,
            B=event.B,
            t0=event.t0,
            links_by_target=links_by_target,
            links_by_source=links_by_source,
            window=30,
            lookback=90
        )

        if result is not None:
            all_results.append(result)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()




def plot_ratio_bars(ratios):
    """Plot after/before incoming link ratios for attacked B and control B' with 95% CI."""
    plt.figure(figsize=(6, 4))
    for i, (name, color) in enumerate([("B: after / before", RED), ("B': after / before", BLUE)]):
        r = ratios.loc[name]
        plt.bar(i, r.estimate, color=color)
        plt.errorbar(i, r.estimate, yerr=[[r.estimate - r.low], [r.high - r.estimate]],
                     fmt="none", color="black", capsize=5)
    ax = plt.gca()
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_xticks([0, 1], ["B (attacked)", "B' (control)"])
    ax.set(title="Links in the 30 days after vs before", ylabel="After / before")
    plt.tight_layout()
    plt.show()


def show_negative_share_comparison(ev_m, ct_m):
    """Compare and plot the share of negative links in the 30 days after t0 between B and B'."""
    neg = pd.DataFrame([share_ci(t.reset_index(), "post30_neg", "post30_n") for t in (ev_m, ct_m)],
                       columns=["share", "low", "high"], index=["B (attacked)", "B' (control)"])
    display(neg.round(3))
    display(boot_share_diff(ev_m, ct_m, "post30_neg", "post30_n").round(3).to_frame("B minus B'").T)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, color) in enumerate([("B (attacked)", RED), ("B' (control)", BLUE)]):
        bar_with_ci(ax, [i], neg.loc[[name]], color, name, width=0.6)
    ax.set_xticks([0, 1], ["B (attacked)", "B' (control)"])
    ax.set(title="Share of negative links in the 30 days after", ylabel="Share of negative links")
    plt.tight_layout()
    plt.show()

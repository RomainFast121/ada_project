# When Reddit communities link negatively to each other

## Abstract

A link from one subreddit to another can draw attention, continue an existing conversation, or express disagreement. We want to understand how negative links fit into the relationships between communities: where negativity first appears in the recorded network, what happens next between the two communities, and whether other communities become involved. We use the [Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html), which records directed links between subreddits from late 2013 to April 2017, with timestamps and sentiment labels. Our initial work follows pairs in chronological order and compares later links by direction and sentiment. We also compare the attention received by a subreddit after its first observed negative incoming link with that of similar subreddits. The dataset starts years after Reddit itself, and its sentiment labels are estimates. We will therefore describe observed patterns, not claim to have found the first-ever interaction or proved that one link caused another.

## Research questions

Our story moves from **origin**, to **evolution**, to **wider attention**.

1. **Origin:** Does negativity appear on a pair's first recorded link or after non-negative links? When a direction turns negative, was there already a negative link in the opposite direction?
2. **Evolution:** After a pair's second observed link, does the other community link back within 30 days? Does the sentiment of that second link relate to whether the first link back is negative, and does the pair's earlier direction matter?
3. **Wider attention:** After a subreddit receives its first observed negative link, does it receive more links from other communities than a similar subreddit does? Are those links more negative, especially when the third community had previously linked to the source?

These questions use different event samples. We will keep their denominators and observation windows visible rather than treating their percentages as directly interchangeable.

## Data and feasibility

The two SNAP files contain 858,488 parsed hyperlink rows in total (about 688 MB on disk). Each row has a source and target subreddit, post ID, timestamp, sentiment label, and 86 text-derived features. Links in titles and bodies are stored separately. Our current analyses need the first five fields and retain title/body origin; the text features remain available if a later question needs them. We parse timestamps, check missing values and duplicate rows, remove self-links, and sort links before constructing unordered pairs while preserving direction. In the downloaded files, the label is either `-1` (negative) or `+1` (neutral or positive); 9.58% of links are negative. Details, checksums, and the source paper are in [dataset_information.md](dataset_information.md).

The hyperlink network contains the links, timing, and sentiment needed for our core questions. For the exploratory similarity comparison, we also use SNAP subreddit embeddings, which cover only part of the network. Community subscriber counts or topic labels could add context later, but they are not required for these analyses.

## Methods and initial analysis

For origin, we order each pair's observed links and locate its first negative link. “First” always means first *in this dataset*. We compare first-link negativity with negativity that appears after recorded non-negative links, and check the earlier sentiment in the opposite direction. This analysis is included in the main results notebook alongside the other two chapters.

For evolution, we select one second link per pair when the first was non-negative and 30 days of later data remain. We separate pairs whose first two links ran in the same direction from pairs that had already linked both ways. We measure the share with a reverse link, then the negative share among first reverse links. Logistic models include an interaction with prior direction and adjust for year and title/body location; confidence intervals use target-clustered standard errors. Varying the response window checks whether the pattern depends on the 30-day choice.

For wider attention, we identify each target's first observed negative incoming link from 2015 onward. We match active targets to subreddits with similar earlier incoming-link counts, using distance between `log1p` counts. We compare before/after attention with a ratio of ratios and resample matched events for uncertainty intervals. We also count temporal triangles: a prior non-negative link between the source and a third community, followed by a link from that community to the target. This last comparison is exploratory; a prior link does not imply friendship or coordination.

The [main initial-results notebook](result_P2.ipynb) presents the origin, evolution, and wider-attention analyses with figures and interpretations. Reusable processing and plotting functions are in [`src/`](src/). The notebook downloads missing hyperlink TSVs and subreddit embeddings from SNAP, verifying new downloads against SHA-256 checksums. The large source files are excluded from Git.

## Plan through P3

| When | Team milestone |
| --- | --- |
| **6 November (P2)** | Agree on shared definitions and submit the proposal, initial results notebook, and documented pipeline. |
| **By 20 November** | Reconcile the origin analysis with the shared pipeline; review event selection, timing, and sensitivity to activity and observation windows. |
| **By 4 December** | Settle the main findings and limitations; select figures and examples that support one coherent story. |
| **By 11 December** | Complete the extended notebook and a first public-facing data story; exchange edits across research lines. |
| **18 December (P3)** | Final checks, add the published story URL to this README, and submit the repository. |

We will review the definitions, assumptions, figures, and writing together so the final notebook and data story read as one project.

## Use of generative AI

OpenAI ChatGPT/Codex provided substantial assistance during this project. We used it to explore whether our ideas were feasible with the dataset, help develop and debug analysis code, and draft and revise parts of the interpretation. It also helped combine notebook sections developed separately by the team and edit their structure and wording for a consistent presentation. We selected the questions and methods to pursue and remain responsible for checking the data, code, figures, and conclusions and for explaining all submitted work.

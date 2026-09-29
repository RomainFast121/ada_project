# Reddit Hyperlink Network — dataset information

## Official sources

- Dataset page: <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>
- Body hyperlinks: <https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv>
- Title hyperlinks: <https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv>
- Optional subreddit embeddings: <https://snap.stanford.edu/data/web-RedditEmbeddings.html>
- Associated paper: <https://cs.stanford.edu/~srijan/pubs/conflict-paper-www18.pdf>

The dataset represents directed hyperlinks from a post in one subreddit to another subreddit. Each link has a timestamp, a sentiment label, and text-derived properties. Separate files record hyperlinks found in post bodies and post titles, making the network directed, signed, temporal, and attributed.

The two downloaded network files are stored locally under `data/raw/`. They are ignored by Git because together they occupy 687,512,603 bytes, or approximately 687.5 MB in decimal units. Their SHA-256 checksums are tracked in [`data/raw/SHA256SUMS`](data/raw/SHA256SUMS).

## Downloaded files

| File | Size (bytes) | Parsed rows | Negative (`-1`) | Neutral or positive (`+1`) |
|---|---:|---:|---:|---:|
| `soc-redditHyperlinks-body.tsv` | 318,931,394 | 286,561 | 21,070 | 265,491 |
| `soc-redditHyperlinks-title.tsv` | 368,581,209 | 571,927 | 61,140 | 510,787 |
| **Combined** | **687,512,603** | **858,488** | **82,210** | **776,278** |

## Schema

Both files are tab-separated and have the same six columns:

| Column | Meaning |
|---|---|
| `SOURCE_SUBREDDIT` | Community containing the source post |
| `TARGET_SUBREDDIT` | Community to which the post links |
| `POST_ID` | Reddit identifier of the source post |
| `TIMESTAMP` | Timestamp attached to the hyperlink record |
| `LINK_SENTIMENT` | `-1` for a negative link; `+1` for a neutral or positive link |
| `PROPERTIES` | Comma-separated vector of 86 text features for the source post |

## Local validation

Validation performed on 29 September 2026 found that:

- both files match the byte sizes returned by the official server;
- every parsed row has exactly six tab-separated fields;
- every `PROPERTIES` field contains exactly 86 values;
- labels are exclusively `-1` and `+1`;
- observed timestamps run from `2013-12-31 16:20:20` to `2017-04-30 16:58:21`;
- there are 55,863 distinct source subreddits and 34,572 distinct target subreddits; their union contains 67,180 distinct names;
- the body file contains 259,092 distinct post IDs and the title file contains 571,922;
- a post can create multiple subreddit-to-subreddit edges, so `POST_ID` is not a unique row key;
- no post ID occurs in both downloaded files.

## Published-versus-observed differences

The SNAP page reports 55,863 nodes and 858,490 edges. Direct parsing of the downloaded files gives 55,863 distinct **source** subreddits, 67,180 distinct subreddit names across source and target fields, and 858,488 rows. The files are internally consistent and exactly match the sizes served by SNAP, so later work should preserve and transparently report these observed values.

The website describes the period as January 2014–April 2017, whereas the earliest observed records are dated 31 December 2013. The raw timestamps should be retained, with any later date filtering documented explicitly.

## Interpretation notes

- The `+1` class combines neutral and positive links; the label is not a three-class sentiment variable.
- Link labels were created using crowdsourced annotations and a classifier described in the associated paper. They are model-derived annotations rather than unquestionable ground truth.
- `PROPERTIES` contains derived textual features, not the original post text.
- The optional embedding dataset has 51,278 vectors of dimension 300 and therefore does not cover every community in the network files.
- The dataset is historical and observational. Associations should not be described as causal effects without a suitable research design.

## Citation

Kumar, S., Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2018). *Community Interaction and Conflict on the Web*. Proceedings of The Web Conference 2018, 933–943.

```bibtex
@inproceedings{kumar2018community,
  title={Community interaction and conflict on the web},
  author={Kumar, Srijan and Hamilton, William L and Leskovec, Jure and Jurafsky, Dan},
  booktitle={Proceedings of the 2018 World Wide Web Conference on World Wide Web},
  pages={933--943},
  year={2018},
  organization={International World Wide Web Conferences Steering Committee}
}
```

## Re-download and verify

Run from the repository root:

```bash
mkdir -p data/raw
curl -L --fail -o data/raw/soc-redditHyperlinks-body.tsv \
  https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
curl -L --fail -o data/raw/soc-redditHyperlinks-title.tsv \
  https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv
shasum -a 256 -c data/raw/SHA256SUMS
```

## Proposal 1: Map of Interactions and Predominant Sentiments
 
**Research questions:** How do Reddit communities interact? Which sentiment dominates their connections? Are there isolated positive or negative clusters?
 
**Outline:** Extract source/target pairs, draw a large network graph with edges coloured by sentiment, then analyse whether communities group together according to positive or negative links.
 
**Advantages**
- Directly feasible: source, target and sentiment are already in the data.
- Visually clear and easy to present.
- Rich analysis possible (community detection, sentiment homophily, structural balance, temporal evolution).
- Descriptive results are always interpretable and reusable for the other proposals.


**Disadvantages**
- Mostly descriptive, little predictive or causal content.
- The full graph (~67k nodes) is unreadable and must be filtered, which introduces selection bias.
- Sentiment labels are automatic and noisy; positive links dominate, so negative clusters may be small.
- Partly covered by the original paper, so a specific angle is needed.
---
 
## Proposal 2: Detection of Hateful Mobilisations and Their Communities
 
**Research questions:** Which groups initiate the most attacks? How can they be identified? What do they have in common?
 
**Outline:** Isolate negative links, use text properties to confirm hostility and the type of language used, then group the data to rank the most toxic source communities and profile them.
 
**Advantages**
- Strong societal relevance and a concrete output (ranking and profile of toxic sources).
- Combines network analysis, statistical tests and clustering.
- Negative links provide a ready-made subset to study.
- Natural extension: bursts of attacks on a single target.


**Disadvantages**
- No raw text: "hate" and "words used" can only be studied through LIWC/VADER-type features.
- Negative does not mean hateful (criticism, rivalry, sarcasm), and there is no hate-speech label to validate.
- Large subreddits emit more negative links in absolute terms, so rates must be normalised.
- Negative links are a minority, which limits statistical power for small communities.
- Ranking "toxic" communities requires careful, neutral wording.
---
 
## Proposal 3: Predicting the Target Community from Text Properties
 
**Research question:** Can we guess which subreddit is targeted from the writing style, the words used and the length of the source post?
 
**Outline:** Group target subreddits into broader topics, train a classifier on the numerical text properties, then test it on new data to see if it predicts the correct target category.
 
**Advantages**
- Clear and measurable objective (accuracy, macro-F1).
- Showcases machine-learning skills: feature engineering, cross-validation, model comparison, SHAP.
- Difficulty can be scaled progressively from a simple baseline.
- A weak result is still an insight.

**Disadvantages**
- Style features are generic and probably carry a weak signal about the target.
- Without raw text, no TF-IDF or language-model approach.
- The source subreddit is a strong confounder (communities link to the same targets repeatedly).
- Results depend on the arbitrary grouping of targets into topics; classes are imbalanced.
- Leakage risk (near-duplicate posts); a temporal split is required.

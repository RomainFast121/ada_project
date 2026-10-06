## Tracing the path to radicalization: How do users become attackers ?
The Community Interaction and Conflict paper raises multiple questions, one of them being: How does one become an attacker? Perhaps the hyperlink network holds answers. Indeed, by reconstructing the activity timeline of a user and identifying common patterns among attackers, one could map attacker hubs and subreddits acting as entry points for them. For instance, the r/History subreddit may not hold many attackers but could be a common entry point for many of them. Following this idea, one could trace the path between this group and another holding more attackers (r/History → r/9-11 → r/Conspiracy).
1. Define Attackers (using paper's method) and Attacking scale
2. Reconstruct user activity timeline and follow the attacking behavior evolution 
3. Identify entry points and hubs
4. Analyze and compare patterns between attackers and others

---

## Is controversial attention a gift ? 
It aims to explore whether controversial or negative attention can drive subreddit growth.
First, we would study the effect of hyperlinks (received or sent); do they correlate with growth metrics, such as the number of followers or engagement levels ? This analysis would help establish whether hyperlink activity, regardless of sentiment, plays a role in subreddit expansion. 
We would distinguish between negative and positive attention, which can easily be done thanks to the embedding containing a negative/neutral dimension : 1, -1. By distinguishing the sentiment associated with hyperlinks, we can analyze whether negative attention has a different impact on growth compared to neutral attention.
Eventually, it could also lead us to identify some growth catalysts: Subreddits with influence thanks to a particularly active community or other factors.

---

## The Anatomy of Information Propagation: Bridges, Speed, and Conflict
Understanding the parameters influencing information propagation speed, identifying bridge subreddits connecting distant communities, and determining whether they foster constructive discussion or conflict.
For this study, one could leverage the 'POST_PROPERTY' embedding to look for common features (readability, emotions) in posts that propagate rapidly, distinguishing them from those that do not. 
Also, one could identify hubs that increase the propagation speed and width and hubs that allow information to pass between two distant subreddits. 
First, one would reconstruct propagation paths for different informations and measure the speed of propagation (e.g. the overall distance traveled by the info in the embedding / the time it took). Then one would look for influencing parameters : centrality, connectivity and linguistics. Eventually, one could find subreddits with high betweenness centrality or edge diversity (connecting multiple clusters) and label them as 'bridges' before classifying them (conflict, constructive, neutral). This information could be represented with a network map.



## Are the friends of my enemies also my enemies?

The idea here is to go beyond a two-way hyperlink and try to analyze the connections between more than two communities. The main idea is to look at the sentiment between two communities, and their sentiment for a third one than could be used to create a bridge between them. The idea would be to test the hypothesis, are the friends of my enemies also my enemies. To this end, we need the hyperlink network dataset but also the embeddings of the subreddits dataset to have a more precise analysis and see if there are any similarities between communities. The data will thus be restricted to those types of cycles, where 2 communities can be linked through another one. Network and directed graphs can be used to represent and analyze this type of data. We could also go further by analyzing how structured are those three parties’ relation depending on the subject (sport, politics, gaming etc.). We know than some recommendation systems are based on what our friends tend to like, if the friend of my enemies are not my enemies, this could lead to another type of recommendation system.

## Do connections between communities overtime make them alive?

The idea here is to analyze how links between communities evolve over time. We would look at whether new hyperlinks appear, whether existing links become more or less frequent, and how the sentiment associated with them changes. Using the timestamp, source and target communities, and sentiment information from the hyperlink network dataset, we could study the evolution of one community through its links with others. Time series methods, potentially including GARCH, could be used to analyze whether sentiment becomes more volatile over time and whether these changes are related to the creation or evolution of links between communities. Sentiment being a discrete value, either -1 or +1, we will need to treat the data, for example by looking at the average over different periods (weekly or monthly depending) and their volatility. We could leverage the time series to predict possible future trends and identify if subjects are going to gain in popularity, could help get better recommendations.

## Do conflicts arise more when the hyperlink is in the body or in the title?

The idea here is to analyze whether conflicts are more frequent when a hyperlink appears in the title or in the body of a post. Using the hyperlink dataset and the post labels, we could compare the proportion of negative interactions in both cases.

This approach could help us flag faster when conflict arise, if there is a higher risk or not associated to where the hyperlink is and try and mitigate them if needed.

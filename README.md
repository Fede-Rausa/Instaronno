# Instaronno

A database for Instagram social network with some data of the users in Saronno city.
The streamlit application lets play with graph theory.

It is also a demonstration for a simple 
node centrality measure / score I have invented, that I call IVND score

## ivnd algorithm

ivnd (iterated voting normalized degree) scoring algorithm:

- initialize a list of weights, one for each node, all equal to 1

- given T iterations, at the iteration t and for the node i performs the following steps:

- voting step: score_i_(t+1) = score_i_(t) + sum_j(score_j_(t)) where j is the index of i's neighbours

- normalization step: score_i(t+1) = score_i(t+1) / sum_k(score_k(t+1)) where the k are all the nodes


It's a node level score, inspired by the degree score.
The degree is the most simple centrality index of a node in the graph:
is simply the number of edges a node has.
But we can see the degree as a voting system:
all the nodes can assign or not a vote to all the others, adding it to others degree level,
and their votes are all equal to 1.

Now, the ivnd score is an iterated version of the degree, 
but where the degree of each node at the end of the previous iteration becomes
the weight of its vote at the next iteration.
It is also normalized, because at the end of each iteration the weights of the nodes
are all divided by their sum. This is because obviously at each iteration adding the old scores
of a node to all the old scores of its neighbours would result in an exponential
growth of the numbers.
At least for this data, pretty always the ivnd scores ranking (not the scores, but the ranks of them)
seems to converges after just 3 iterations.
This happens also when the scores are not normalized.
However, if they are normalized, for more iterations 
also the scores seems to converge, due to the normalization step.
If the normalization step is not taken, they all diverge to infinity.
It must be noticed that if the weights are all initialized to 1,
at the first iteration the ranking result would be just the same of the degree.







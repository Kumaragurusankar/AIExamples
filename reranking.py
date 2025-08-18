def rerank_ results (query: str, results: list, deal_mnc: str, boost: float = 0.2):
Rerank FAISS results based on Deal Mnc match.
Args:
query: User input query
results: List of tuples (doc, score) from FAISS deal_mnc: The Deal Mc extracted from query .(e.g. "AGGRTEST")
• boost: How much to boost if Deal Mc matches
Returns:
Reranked list of (doc, new_score)
reranked = []
for doc, score in results:
new_ score = score
if deal_mnc. lower () in doc. Lower () :
new score += boost
# give priority
reranked. append ( (doc, new _score))
# Sort descending by new_score
reranked = sorted (reranked, key=lambda ×: x(1], reverse=True) return reranked

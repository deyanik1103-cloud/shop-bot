import glob, os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks(folder="knowledge"):
    """Split every markdown file into chunks, one per '## ' section."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for part in re.split(r"\n(?=## )", text):
            if part.strip() and not part.lstrip().startswith("# "):  # skip title-only chunks
                chunks.append({"source": os.path.basename(path), "text": part.strip()})
    return chunks


class Retriever:
    """TF-IDF retriever. To upgrade to embeddings, keep this same interface
    (search(query, k) -> list of chunks) and swap the internals."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([c["text"] for c in chunks])

    def search(self, query, k=3, min_score=0.05):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if scores[i] >= min_score]

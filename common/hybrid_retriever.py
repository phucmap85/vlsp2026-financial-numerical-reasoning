import os, re, sys, math, json, torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def tokenize(text: str) -> List[str]:
    words = re.findall(r'[\w/%+-]+', text.lower().replace('\xa0', ' '))
    return words + [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]


class BM25Retriever:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.size = k1, b, len(corpus)
        self.lens = [len(doc) for doc in corpus]
        self.avg_len = sum(self.lens) / max(self.size, 1)
        self.tfs = [Counter(doc) for doc in corpus]
        dfs = Counter(t for doc in corpus for t in set(doc))
        self.idf = {t: math.log((self.size - df + 0.5) / (df + 0.5) + 1.0) for t, df in dfs.items()}

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.size
        
        for t in query:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf_map in enumerate(self.tfs):
                if t in tf_map:
                    tf = tf_map[t]
                    denom = tf + self.k1 * (1 - self.b + self.b * (self.lens[i] / self.avg_len))
                    scores[i] += idf * (tf * (self.k1 + 1)) / denom
        
        return scores


class DenseRetriever:
    def __init__(
        self,
        texts: List[str], 
        model_name: str = "BAAI/bge-m3",
        cache_path: str = "embeddings_cache.pt",
        batch_size: int = 32
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if os.path.exists(cache_path):
            print(f"Loaded existing embedding ckpt: {cache_path}")
            self.embeddings = torch.load(cache_path, weights_only=True)
        else:
            print(f"No ckpt found at '{cache_path}'. Generating embeddings...")
            self.embeddings = self._encode_batch(texts, batch_size=batch_size)
            torch.save(self.embeddings, cache_path)
            print(f"Saved embedding ckpt: {cache_path}")

    def _encode_batch(self, texts: List[str], batch_size: int = 32):
        all_embs = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
                all_embs.append(emb.cpu())
            if (i // batch_size) % 10 == 0 or i + batch_size >= total:
                print(f"  -> Encoded {min(i+batch_size, total)}/{total} samples...")
        
        return torch.cat(all_embs, dim=0)

    def query_scores(self, query: str) -> List[float]:
        inputs = self.tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model(**inputs)
            q_emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1).cpu()
            scores = (q_emb @ self.embeddings.T).squeeze(0)
            return scores.tolist()


class HybridRetriever:
    def __init__(
        self,
        train_path: str,
        model_name: str = "BAAI/bge-m3",
        cache_path: str = "embeddings_cache.pt",
        batch_size: int = 32,
    ):
        if not os.path.isabs(train_path):
            train_path = os.path.join(os.path.dirname(__file__), train_path)
        with open(train_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.questions = [d['qa']['question'] for d in self.data]
        self.programs = [d['qa']['program'] for d in self.data]
        self.bm25 = BM25Retriever([tokenize(q) for q in self.questions])
        self.dense = DenseRetriever(
            self.questions,
            model_name=model_name,
            cache_path=cache_path,
            batch_size=batch_size,
        )

    def search(self, query: str, top_k: int = 3, rrf_k: int = 60, exclude_index: Optional[int] = None) -> List[Dict[str, Any]]:
        cands = [i for i in range(len(self.data)) if i != exclude_index]

        bm25_s = self.bm25.get_scores(tokenize(query))
        dense_s = self.dense.query_scores(query)

        bm25_ranked = sorted(cands, key=lambda i: bm25_s[i], reverse=True)[:30]
        dense_ranked = sorted(cands, key=lambda i: dense_s[i], reverse=True)[:30]

        rrf = defaultdict(float)
        for r, i in enumerate(bm25_ranked):
            rrf[i] += 1.0 / (rrf_k + r + 1)
        for r, i in enumerate(dense_ranked):
            rrf[i] += 1.0 / (rrf_k + r + 1)

        top_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:top_k]
        
        return [{
            "index": i,
            "question": self.questions[i],
            "answer": self.programs[i],
            "rrf_score": rrf[i],
            "bm25_score": bm25_s[i],
            "dense_score": dense_s[i]
        } for i in top_ids]

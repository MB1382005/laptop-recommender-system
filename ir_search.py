import re
import math
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","of","with","for","in","on","at","to",
    "is","it","its","this","that","are","was","be","by","from",
    "laptop","notebooks","inch","gb","tb","display","processor","graphics",
    "version","upgraded","english","arabic"
}

# ─────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────
def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_doc(laptop: dict) -> str:
    parts = [
        laptop.get("title", ""),
        laptop.get("cpu", ""),
        laptop.get("ram", ""),
        laptop.get("storage", ""),
        laptop.get("screen", ""),
    ]
    return " ".join(str(p) for p in parts if p and str(p) != "nan")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _parse_ram(x):
    nums = re.findall(r"\d+", str(x))
    return int(nums[0]) if nums else None

def _safe_price(lp):
    try:
        return float(lp.get("price", 0))
    except:
        return 0.0


# ─────────────────────────────────────────────
# TF-IDF INDEX
# ─────────────────────────────────────────────
class TFIDFIndex:
    def __init__(self):
        self.docs       = []
        self.doc_tokens = []
        self.idf        = {}
        self.tf_matrix  = []
        self._built     = False

    def build(self, laptops: list):
        self.docs = laptops
        N = len(laptops)

        self.doc_tokens = [tokenize(build_doc(lp)) for lp in laptops]

        df_counts = defaultdict(int)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df_counts[t] += 1

        self.idf = {
            t: math.log((N + 1) / (df + 1)) + 1
            for t, df in df_counts.items()
        }

        self.tf_matrix = []
        for tokens in self.doc_tokens:
            tf = defaultdict(float)
            for t in tokens:
                tf[t] += 1
            n = len(tokens) or 1
            self.tf_matrix.append({t: v / n for t, v in tf.items()})

        self._built = True
        print(f"[IR] Index built: {N} laptops indexed")
        return self

    def search(
        self,
        query: str,
        top_n: int = 10,
        min_price: float = None,
        max_price: float = None,
        ram_filter: str = None,
        cpu_filter: str = None
    ):
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # ─── STEP 1: FILTER ───────────────────────────────────────────────
        valid_indices = []
        for i, lp in enumerate(self.docs):
            price = _safe_price(lp)

            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue

            if ram_filter and ram_filter != "all":
                lp_ram = _parse_ram(lp.get("ram"))
                f_ram  = _parse_ram(ram_filter)
                if lp_ram is None or f_ram is None or lp_ram != f_ram:
                    continue

            if cpu_filter and cpu_filter != "all":
                if cpu_filter.lower() not in str(lp.get("cpu", "")).lower():
                    continue

            valid_indices.append(i)

        if not valid_indices:
            return []

        # ─── STEP 2: SCORE ────────────────────────────────────────────────
        scored = []
        for i in valid_indices:
            score = 0.0
            for t in q_tokens:
                score += self.tf_matrix[i].get(t, 0) * self.idf.get(t, 0)
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        all_zero = all(s == 0 for _, s in scored)
        if all_zero:
            results = []
            for i, _ in scored[:top_n]:
                lp = dict(self.docs[i])
                lp["ir_score"] = 0.0
                results.append(lp)
            return results

        relevant = [(i, s) for i, s in scored if s > 0]
        results = []
        for i, score in relevant[:top_n]:
            lp = dict(self.docs[i])
            lp["ir_score"] = round(float(score), 4)
            results.append(lp)

        return results


_index = None

def get_index():
    global _index
    if _index is None:
        from model import get_all_laptops
        laptops = get_all_laptops()
        _index  = TFIDFIndex().build(laptops)
    return _index

def reset_index():
    global _index
    _index = None

def search(query, top_n=10, **filters):
    idx = get_index()
    return idx.search(query, top_n=top_n, **filters)
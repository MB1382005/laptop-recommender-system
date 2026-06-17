import re
import numpy as np
import pandas as pd
from collections import Counter

# ── internal helpers ──────────────────────────────────────────────────────────

def _parse_gb(val):
    if pd.isna(val): return np.nan
    val = str(val).strip().upper()
    if "TB" in val:
        try: return float(val.replace("TB","").strip()) * 1024
        except: return np.nan
    if "GB" in val:
        try: return float(val.replace("GB","").strip())
        except: return np.nan
    return np.nan

def _parse_screen(val):
    if pd.isna(val): return np.nan
    try: return float(str(val).replace("in","").strip())
    except: return np.nan

def _cpu_brand(cpu):
    if pd.isna(cpu): return "Other"
    cpu = str(cpu).lower()
    if "ryzen" in cpu or "amd" in cpu: return "AMD"
    if "apple" in cpu or " m1" in cpu or " m2" in cpu or " m3" in cpu or " m4" in cpu: return "Apple"
    if "intel" in cpu or "core" in cpu or "celeron" in cpu or "pentium" in cpu: return "Intel"
    return "Other"

def _laptop_type(title):
    t = str(title).lower()
    if "gaming" in t or "rog" in t or "tuf" in t or "predator" in t or "nitro" in t: return "Gaming"
    if "macbook" in t or "mac" in t: return "MacBook"
    if "business" in t or "thinkpad" in t or "elitebook" in t or "latitude" in t: return "Business"
    if "chromebook" in t: return "Chromebook"
    return "General"

# ── main function ─────────────────────────────────────────────────────────────

def get_eda_stats(csv_path="data/laptops.csv"):
    """
    Returns a dict with all EDA stats for the dashboard.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()


    required_with_defaults = {
        "title":      "Unknown Laptop",
        "price":      0,
        "cpu":        "Unknown",
        "ram":        "Unknown",
        "storage":    "Unknown",
        "screen":     "Unknown",
        "rating":     0,       
        "os_version": "Unknown", 
        "link":       "",        
        "image_main": "",        
    }

    for col, default in required_with_defaults.items():
        if col not in df.columns:
            alt_names = {
                "image_main": ["image", "img", "image_url", "Image"],
                "os_version": ["os", "OS", "operating_system", "Operating System"],
                "rating":     ["Rating", "rate", "stars", "Stars", "score"],
                "link":       ["Link", "url", "URL", "product_url"],
            }
            found = False
            for alt in alt_names.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    found = True
                    break
            if not found:
                df[col] = default

    work_cols = ["title", "price", "rating", "cpu", "ram",
                 "storage", "screen", "os_version", "link", "image_main"]
    df = df[work_cols].copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    total_before = len(df)
    df = df[df["price"].notna() & (df["price"] >= 100)].copy()
    total_filtered = total_before - len(df)

    if total_filtered > 0:
        print(f"[EDA] Filtered out {total_filtered} laptops with invalid/zero prices")

    print(f"[EDA] Processing {len(df)} valid laptops out of {total_before} total")

    # ── parse numerics ──
    df["ram_gb"]     = df["ram"].apply(_parse_gb)
    df["storage_gb"] = df["storage"].apply(_parse_gb)
    df["screen_in"]  = df["screen"].apply(_parse_screen)
    df["cpu_brand"]  = df["cpu"].apply(_cpu_brand)
    df["type"]       = df["title"].apply(_laptop_type)

    # ── overview ──────────────────────────────────────────────────────────────
    overview = {
        "total":          int(len(df)),
        "total_raw":      int(total_before),
        "filtered_out":   int(total_filtered),
        "avg_price":      round(float(df["price"].mean()), 2),
        "min_price":      round(float(df["price"].min()), 2),
        "max_price":      round(float(df["price"].max()), 2),
        "avg_ram_gb":     round(float(df["ram_gb"].dropna().mean()), 1) if df["ram_gb"].notna().any() else 0,
        "avg_storage_gb": round(float(df["storage_gb"].dropna().mean()), 1) if df["storage_gb"].notna().any() else 0,
        "unique_cpu":     int(df["cpu"].nunique()),
    }

    # ── price distribution (10 bins) ─────────────────────────────────────────
    prices = df["price"].dropna()
    counts, bin_edges = np.histogram(prices, bins=10)
    price_dist = {
        "labels": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(counts))],
        "values": counts.tolist(),
    }

    # ── RAM distribution ─────────────────────────────────────────────────────
    ram_counts = df["ram"].value_counts()
    ram_dist = {
        "labels": ram_counts.index.tolist(),
        "values": ram_counts.values.tolist(),
    }

    # ── Storage distribution ─────────────────────────────────────────────────
    storage_counts = df["storage"].value_counts()
    storage_dist = {
        "labels": storage_counts.index.tolist(),
        "values": storage_counts.values.tolist(),
    }

    # ── CPU brand breakdown ──────────────────────────────────────────────────
    brand_counts = df["cpu_brand"].value_counts()
    cpu_brands = {
        "labels": brand_counts.index.tolist(),
        "values": brand_counts.values.tolist(),
    }

    # ── Laptop type breakdown ────────────────────────────────────────────────
    type_counts = df["type"].value_counts()
    laptop_types = {
        "labels": type_counts.index.tolist(),
        "values": type_counts.values.tolist(),
    }

    # ── Screen size distribution ─────────────────────────────────────────────
    screen_counts = df["screen"].value_counts().head(8)
    screen_sizes = {
        "labels": screen_counts.index.tolist(),
        "values": screen_counts.values.tolist(),
    }

    # ── Avg price by RAM ─────────────────────────────────────────────────────
    pbr = df.groupby("ram")["price"].mean().sort_values(ascending=False)
    price_by_ram = {
        "labels": pbr.index.tolist(),
        "values": [round(float(v), 2) for v in pbr.values],
    }

    # ── Top 5 expensive / cheapest ───────────────────────────────────────────
    def _to_list(sub):
        out = []
        for _, row in sub.iterrows():
            out.append({
                "title":   str(row["title"])[:70],
                "price":   round(float(row["price"]), 2),
                "cpu":     str(row["cpu"]),
                "ram":     str(row["ram"]),
                "storage": str(row["storage"]),
            })
        return out

    top_expensive = _to_list(df.nlargest(5,  "price"))
    top_cheap     = _to_list(df.nsmallest(5, "price"))

    # ── Keyword frequency in titles ──────────────────────────────────────────
    stopwords = {
        "with","inch","laptop","and","or","of","the","a","in",
        "for","to","from","at","display","version","upgraded",
        "full","hd","english","arabic","home","pro","windows"
    }
    all_words = []
    for title in df["title"].dropna():
        tokens = re.sub(r"[^a-z0-9\s]", " ", title.lower()).split()
        all_words.extend([t for t in tokens if t not in stopwords and len(t) > 2])

    word_freq = Counter(all_words).most_common(20)
    keyword_freq = {
        "labels": [w for w, _ in word_freq],
        "values": [c for _, c in word_freq],
    }

    return {
        "overview":      overview,
        "price_dist":    price_dist,
        "ram_dist":      ram_dist,
        "storage_dist":  storage_dist,
        "cpu_brands":    cpu_brands,
        "laptop_types":  laptop_types,
        "screen_sizes":  screen_sizes,
        "price_by_ram":  price_by_ram,
        "top_expensive": top_expensive,
        "top_cheap":     top_cheap,
        "keyword_freq":  keyword_freq,
    }

# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    stats = get_eda_stats()
    ov = stats["overview"]
    print(f"Total valid: {ov['total']} / {ov['total_raw']} (filtered: {ov['filtered_out']})")
    print("CPU brands:  ", stats["cpu_brands"])
    print("Laptop types:", stats["laptop_types"])
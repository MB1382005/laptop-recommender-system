"""
query_parser.py
---------------
بيحول الـ query اللي كتبها المستخدم لشكل الـ search engine يفهمه.

مثال:
  "RTX4060"     → ["RTX", "4060", "RTX 4060"]
  "i7 16جيجا"   → ["i7", "16GB", "16 GB"]
  "لاب توب شغل" → ["office", "business", "work"]
"""

import re

# ─────────────────────────────────────────────────────────────
# GPU MODEL PATTERNS
# ─────────────────────────────────────────────────────────────
# بيوعى إن "RTX4060" = "RTX 4060"
GPU_PATTERNS = [
    # RTX / GTX بدون مسافة
    (r"(rtx|gtx)(\d{3,4})", r"\1 \2"),
    # RX بدون مسافة
    (r"\brx(\d{3,4})\b", r"RX \1"),
    # AMD Radeon بدون مسافة
    (r"radeon(\d{3,4})", r"Radeon \1"),
]

# ─────────────────────────────────────────────────────────────
# CPU PATTERNS
# ─────────────────────────────────────────────────────────────
CPU_PATTERNS = [
    # i7-12700H → "i7 12700H" و "Core i7"
    (r"\b(i[3579])[- ]?(\d{4,5})", r"\1 \2"),
    # ryzen7 → ryzen 7
    (r"\bryzen(\d)\b", r"ryzen \1"),
    # corei7 → core i7
    (r"\bcore(i[3579])\b", r"core \1"),
]

# ─────────────────────────────────────────────────────────────
# RAM PATTERNS
# ─────────────────────────────────────────────────────────────
RAM_PATTERNS = [
    # 16GB / 16جيجا / 16 gb → "16 GB"
    (r"(\d+)\s*(gb|جيجا|جيجابايت)\b", r"\1 GB"),
    # 1TB → "1 TB"
    (r"(\d+)\s*(tb|تيرا)\b", r"\1 TB"),
]

# ─────────────────────────────────────────────────────────────
# ARABIC → ENGLISH TRANSLATIONS
# ─────────────────────────────────────────────────────────────
AR_TO_EN = {
    # use cases
    "جيمنج":         "gaming",
    "قيمنق":         "gaming",
    "العاب":         "gaming",
    "ألعاب":         "gaming",
    "لعب":           "gaming",
    "برمجة":         "programming",
    "مبرمج":         "programming developer",
    "كودينج":        "coding",
    "تطوير":         "development",
    "مطور":          "developer",
    "مونتاج":        "video editing",
    "فيديو":         "video",
    "يوتيوب":        "youtube content creator",
    "كونتنت":        "content creator",
    "محتوى":         "content creator",
    "شغل":           "office work business",
    "عمل":           "office work",
    "اوفيس":         "office",
    "موظف":          "office business",
    "طالب":          "student",
    "دراسة":         "study student",
    "جامعة":         "university student",
    "كلية":          "college student",
    "رخيص":          "budget affordable cheap",
    "ميزانية":       "budget",
    "خفيف":          "lightweight portable slim",
    "محمول":         "portable",
    "رفيع":          "slim thin",
    "سفر":           "travel portable",
    "بطارية":        "battery",
    "اولترابوك":     "ultrabook",
    "موسيقى":        "music production",
    "اوديو":         "audio",
    "مكساج":         "mixing music",
    "تسجيل":         "recording",
    "بودكاست":       "podcast",
    "اوتوكاد":       "autocad architecture",
    "معماري":        "architecture",
    "معمارى":        "architecture",
    "تصميم":         "design",
    "ذكاء اصطناعي":  "machine learning ai",
    "تعلم آلي":      "machine learning",
    "داتا":          "data science",
    # specs
    "جيجا":          "GB",
    "جيجابايت":      "GB",
    "تيرا":          "TB",
    "رام":           "RAM",
    "معالج":         "processor CPU",
    "شاشة":          "screen display",
    "بطارية":        "battery",
    # brands
    "أسوس":          "asus",
    "لينوفو":        "lenovo",
    "هواوي":         "huawei",
    "ابل":           "apple",
    "آبل":           "apple",
    "سامسونج":       "samsung",
    "ديل":           "dell",
    "اتش بي":        "hp",
    "ايسر":          "acer",
}

# ─────────────────────────────────────────────────────────────
# SPEC EXTRACTORS (من الـ query مباشرة)
# ─────────────────────────────────────────────────────────────

def extract_specs_from_query(query: str) -> dict:
    """
    بيستخرج specs صريحة من الـ query.
    مثال: "i7 16GB RTX 4060" → {cpu: "i7", ram: 16, gpu: "RTX 4060"}
    """
    q   = query.lower()
    out = {}

    # RAM
    ram_m = re.search(r"(\d+)\s*(?:gb|جيجا)", q)
    if ram_m:
        out["ram_gb"] = int(ram_m.group(1))

    # Storage
    stg_m = re.search(r"(\d+)\s*(?:tb|تيرا)", q)
    if stg_m:
        out["storage_tb"] = int(stg_m.group(1))
    else:
        stg_m2 = re.search(r"(\d+)\s*gb\s*(?:ssd|nvme|storage)", q)
        if stg_m2:
            out["storage_gb"] = int(stg_m2.group(1))

    # GPU model
    gpu_m = re.search(r"(rtx|gtx)\s?(\d{3,4})", q)
    if gpu_m:
        out["gpu_model"] = f"{gpu_m.group(1).upper()} {gpu_m.group(2)}"

    rx_m = re.search(r"rx\s?(\d{4})", q)
    if rx_m:
        out["gpu_model"] = f"RX {rx_m.group(1)}"

    # CPU
    cpu_m = re.search(r"(i[3579])[- ]?(\d{4,5})?", q)
    if cpu_m:
        out["cpu_gen"] = cpu_m.group(1)
        if cpu_m.group(2):
            out["cpu_model"] = f"{cpu_m.group(1)}-{cpu_m.group(2)}"

    ryzen_m = re.search(r"ryzen\s?([579])", q)
    if ryzen_m:
        out["cpu_gen"] = f"ryzen {ryzen_m.group(1)}"

    # Screen size
    scr_m = re.search(r"(\d{2})\.?\d?\s*(?:inch|in|بوصة)", q)
    if scr_m:
        out["screen_in"] = float(scr_m.group(1))

    # Price range
    price_m = re.search(r"(?:under|أقل من|تحت|max)\s*(\d{3,5})", q)
    if price_m:
        out["max_price"] = int(price_m.group(1))

    price_m2 = re.search(r"(\d{3,5})\s*(?:to|-)\s*(\d{3,5})", q)
    if price_m2:
        out["min_price"] = int(price_m2.group(1))
        out["max_price"] = int(price_m2.group(2))

    return out


# ─────────────────────────────────────────────────────────────
# QUERY NORMALIZER
# ─────────────────────────────────────────────────────────────

def normalize_query(query: str) -> str:

    q = query.strip()

    ar_sorted = sorted(AR_TO_EN.keys(), key=len, reverse=True)
    for ar, en in [(k, AR_TO_EN[k]) for k in ar_sorted]:
        if ar in q:
            q = q.replace(ar, f" {en} ")

    q_lower = q.lower()

    for pattern, replacement in GPU_PATTERNS:
        q_lower = re.sub(pattern, replacement, q_lower, flags=re.IGNORECASE)

    for pattern, replacement in CPU_PATTERNS:
        q_lower = re.sub(pattern, replacement, q_lower, flags=re.IGNORECASE)

    for pattern, replacement in RAM_PATTERNS:
        q_lower = re.sub(pattern, replacement, q_lower, flags=re.IGNORECASE)

    q_lower = re.sub(r"\s+", " ", q_lower).strip()

    return q_lower


def expand_query(query: str) -> list[str]:
    """
    بيرجع قائمة بكل الـ variations للـ query.
    مثال: "RTX4060" → ["rtx 4060", "rtx", "4060", "geforce rtx 4060"]
    """
    normalized = normalize_query(query)
    tokens     = normalized.split()
    variants   = {normalized}

    for t in tokens:
        if len(t) > 1:
            variants.add(t)

    # GPU expansions
    gpu_m = re.search(r"(rtx|gtx)\s?(\d{3,4})", normalized, re.IGNORECASE)
    if gpu_m:
        brand = gpu_m.group(1).upper()
        model = gpu_m.group(2)
        variants.update([
            f"{brand} {model}",
            f"geforce {brand} {model}",
            f"nvidia {brand} {model}",
            brand.lower(),
            model,
        ])

    rx_m = re.search(r"rx\s?(\d{4})", normalized, re.IGNORECASE)
    if rx_m:
        model = rx_m.group(1)
        variants.update([
            f"rx {model}",
            f"radeon rx {model}",
            f"amd rx {model}",
            model,
        ])

    # CPU expansions
    cpu_m = re.search(r"(i[3579])[- ]?(\d{4,5})?", normalized)
    if cpu_m:
        gen = cpu_m.group(1)
        variants.update([
            f"core {gen}",
            f"intel {gen}",
            gen,
        ])
        if cpu_m.group(2):
            variants.add(f"{gen}-{cpu_m.group(2)}")

    return list(variants)


# ─────────────────────────────────────────────────────────────
# HARD FILTER (بيفلتر بـ specs صريحة من الـ query)
# ─────────────────────────────────────────────────────────────

def hard_filter(laptops: list, specs: dict) -> list:
    """
    لو المستخدم كتب spec صريحة (زي RTX 4060)،
    بيفلتر ويشيل اللاب توبات اللي مش عندها الـ spec دي.
    """
    filtered = []

    for lp in laptops:
        gpu     = str(lp.get("gpu",     "")).lower()
        title   = str(lp.get("title",   "")).lower()
        cpu     = str(lp.get("cpu",     "")).lower()
        ram_txt = str(lp.get("ram",     "")).lower()
        price   = float(lp.get("price", 0) or 0)

        searchable = f"{title} {gpu} {cpu}"

        # GPU model filter (الأهم — لو كتب RTX 4060 لازم يبقى موجود)
        if "gpu_model" in specs:
            model = specs["gpu_model"].lower()
            # بنفصل الـ brand عن الـ number
            parts = model.split()
            brand_match  = parts[0] in searchable if parts else False
            number_match = parts[1] in searchable if len(parts) > 1 else True
            if not (brand_match and number_match):
                continue

        # RAM filter
        if "ram_gb" in specs:
            ram_nums = re.findall(r"\d+", ram_txt)
            lp_ram   = int(ram_nums[0]) if ram_nums else 0
            if lp_ram < specs["ram_gb"]:
                continue

        # Price filter
        if "max_price" in specs and price > specs["max_price"]:
            continue
        if "min_price" in specs and price < specs["min_price"]:
            continue

        filtered.append(lp)

    return filtered
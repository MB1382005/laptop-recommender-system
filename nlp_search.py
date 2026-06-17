"""
nlp_search.py
-------------
Smart NLP-based laptop search with intent detection and scoring.

✅ التطويرات:
  - intents جديدة: video_editing, office, architecture, music_production, portable
  - دعم اللغة العربية في الـ keywords
  - scoring منطقي ومـ normalized
  - فهم أفضل للـ query (multi-intent + fallback)
  - debugging: كل laptop بيجيب score_breakdown
"""

import re
from operator import itemgetter

# ─────────────────────────────────────────────────────────────
# REGEX HELPERS
# ─────────────────────────────────────────────────────────────
RAM_REGEX     = re.compile(r"\d+")
STORAGE_REGEX = re.compile(r"(\d+)\s*(tb|gb)", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────
# INTENTS
# ─────────────────────────────────────────────────────────────
INTENTS = {

    "gaming": {
        "keywords": {
            # EN
            "gaming", "gamer", "games", "game", "fps", "rtx",
            "esports", "rog", "tuf", "legion", "nitro", "predator",
            # AR
            "جيمنج", "العاب", "ألعاب", "قيمنق", "لعب"
        },
        "min_ram": 16,
        "weights": {"ram": 6, "gpu": 20, "cpu": 8, "ssd": 3, "screen": 3},
        "gpu_keywords": {"rtx", "gtx", "rx 6", "rx 7", "arc"},
        "boost_words":  {"rog", "tuf", "legion", "predator", "nitro", "scar", "strix"},
        "preferred_cpu": {"i7", "i9", "ryzen 7", "ryzen 9"},
        "min_storage_gb": 512,
    },

    "machine_learning": {
        "keywords": {
            # EN
            "machine learning", "ml", "ai", "deep learning", "data science",
            "tensorflow", "pytorch", "neural", "training", "model",
            "data", "python", "jupyter",
            # AR
            "ذكاء اصطناعي", "تعلم آلي", "تعلم الآلة", "داتا ساينس"
        },
        "min_ram": 32,
        "weights": {"ram": 12, "gpu": 15, "cpu": 8, "ssd": 5, "screen": 1},
        "gpu_keywords": {"rtx 40", "rtx 30", "rtx 4", "a2000", "a3000", "quadro"},
        "boost_words":  {"workstation", "studio", "zbook", "precision"},
        "preferred_cpu": {"i9", "ryzen 9", "i7", "ryzen 7"},
        "min_storage_gb": 512,
    },

    "programming": {
        "keywords": {
            # EN
            "coding", "programming", "developer", "software", "dev",
            "code", "engineer", "backend", "frontend", "fullstack",
            "linux", "docker", "git",
            # AR
            "برمجة", "مبرمج", "كودينج", "تطوير", "مطور"
        },
        "min_ram": 16,
        "weights": {"ram": 7, "gpu": 1, "cpu": 10, "ssd": 8, "screen": 2},
        "gpu_keywords": set(),
        "boost_words":  {"thinkpad", "xps", "macbook", "elitebook", "latitude"},
        "preferred_cpu": {"i7", "i9", "ryzen 7", "ryzen 9", "ultra 7", "ultra 9"},
        "min_storage_gb": 256,
    },

    "video_editing": {
        "keywords": {
            # EN
            "video editing", "video", "editing", "premiere", "after effects",
            "davinci", "resolve", "4k", "render", "rendering", "youtube",
            "content creator", "creator", "vlog", "filmmaking",
            # AR
            "مونتاج", "تصوير", "فيديو", "يوتيوب", "كونتنت", "محتوى"
        },
        "min_ram": 16,
        "weights": {"ram": 10, "gpu": 12, "cpu": 10, "ssd": 8, "screen": 5},
        "gpu_keywords": {"rtx", "rx 6", "rx 7", "m1", "m2", "m3", "m4"},
        "boost_words":  {"macbook", "xps", "studio", "creator", "proart"},
        "preferred_cpu": {"i7", "i9", "ryzen 7", "ryzen 9", "m1", "m2", "m3", "m4"},
        "min_storage_gb": 512,
        "prefer_large_screen": True,
    },

    "office": {
        "keywords": {
            # EN
            "office", "business", "work", "excel", "word", "powerpoint",
            "teams", "zoom", "email", "corporate", "professional",
            "productivity", "remote work", "work from home",
            # AR
            "شغل", "عمل", "اوفيس", "اوفيس", "شركة", "بيزنس", "موظف"
        },
        "min_ram": 8,
        "weights": {"ram": 4, "gpu": 0, "cpu": 6, "ssd": 5, "screen": 2},
        "gpu_keywords": set(),
        "boost_words":  {"thinkpad", "elitebook", "latitude", "inspiron", "vostro"},
        "preferred_cpu": {"i5", "i7", "ryzen 5", "ryzen 7"},
        "min_storage_gb": 256,
    },

    "architecture": {
        "keywords": {
            # EN
            "architecture", "autocad", "revit", "3ds max", "sketchup",
            "rendering", "bim", "civil", "design", "cad", "3d modeling",
            # AR
            "معمارى", "معماري", "اوتوكاد", "ريفيت", "تصميم معماري", "هندسة"
        },
        "min_ram": 32,
        "weights": {"ram": 10, "gpu": 14, "cpu": 10, "ssd": 6, "screen": 5},
        "gpu_keywords": {"rtx", "quadro", "rx 6", "rx 7", "arc"},
        "boost_words":  {"workstation", "zbook", "precision", "studio", "proart"},
        "preferred_cpu": {"i7", "i9", "ryzen 7", "ryzen 9", "ultra 7"},
        "min_storage_gb": 512,
        "prefer_large_screen": True,
    },

    "music_production": {
        "keywords": {
            # EN
            "music", "audio", "daw", "fl studio", "ableton", "logic",
            "producer", "recording", "mixing", "mastering", "podcast",
            # AR
            "موسيقى", "اوديو", "مكساج", "تسجيل", "بودكاست"
        },
        "min_ram": 16,
        "weights": {"ram": 8, "gpu": 2, "cpu": 10, "ssd": 8, "screen": 2},
        "gpu_keywords": set(),
        "boost_words":  {"macbook", "xps", "spectre"},
        "preferred_cpu": {"i7", "i9", "m1", "m2", "m3", "m4", "ryzen 7"},
        "min_storage_gb": 512,
    },

    "portable": {
        "keywords": {
            # EN
            "portable", "lightweight", "light", "thin", "slim",
            "travel", "ultrabook", "battery", "long battery",
            # AR
            "خفيف", "محمول", "رفيع", "سفر", "بطارية", "اولترابوك"
        },
        "min_ram": 8,
        "weights": {"ram": 3, "gpu": 0, "cpu": 4, "ssd": 4, "screen": 3},
        "gpu_keywords": set(),
        "boost_words":  {"ultrabook", "slim", "air", "yoga", "spectre", "swift"},
        "preferred_cpu": {"i5", "i7", "ryzen 5", "ryzen 7", "m1", "m2", "m3"},
        "min_storage_gb": 256,
        "prefer_small_screen": True,
    },

    "student": {
        "keywords": {
            # EN
            "student", "study", "college", "university", "school",
            "budget", "cheap", "affordable",
            # AR
            "طالب", "دراسة", "جامعة", "كلية", "رخيص", "ميزانية"
        },
        "max_price": 3000,
        "min_ram": 8,
        "weights": {"ram": 3, "gpu": 1, "cpu": 4, "ssd": 3, "screen": 2},
        "gpu_keywords": set(),
        "boost_words":  set(),
        "preferred_cpu": {"i5", "ryzen 5"},
        "min_storage_gb": 256,
    },
}

# ─────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────

def detect_intents(query: str, top_k: int = 2) -> list[tuple[str, float]]:
    """
    بيرجع أحسن top_k intents مرتبين بالـ score.
    بدل intent واحد، بنقدر نـ blend أكتر من intent.
    """
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    scores = {}

    for intent_name, config in INTENTS.items():
        score = 0.0
        for kw in config["keywords"]:
            kw_lower = kw.lower()
            # exact phrase match → وزن أعلى
            if kw_lower in query_lower:
                score += len(kw_lower.split()) * 2  # عدد الكلمات × 2
            else:
                # partial word overlap
                kw_words = set(kw_lower.split())
                matched  = query_words & kw_words
                score   += len(matched) * 0.5

        if score > 0:
            scores[intent_name] = score

    if not scores:
        return []

    # رتب وخد أحسن top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# backward compat
def detect_intent(query: str) -> str | None:
    results = detect_intents(query, top_k=1)
    return results[0][0] if results else None

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────

def extract_ram_gb(text: str) -> int:
    m = RAM_REGEX.search(str(text))
    return int(m.group()) if m else 0

def extract_storage_gb(text: str) -> float:
    text = str(text).lower()
    m = STORAGE_REGEX.search(text)
    if not m:
        return 0
    val, unit = float(m.group(1)), m.group(2).lower()
    return val * 1024 if unit == "tb" else val

def extract_screen_in(text: str) -> float:
    m = re.search(r"(\d+\.?\d*)", str(text))
    return float(m.group(1)) if m else 0.0

def _cpu_tier(cpu: str) -> int:
    """بيرجع رقم من 0-5 يمثل قوة الـ CPU."""
    cpu = cpu.lower()
    if any(x in cpu for x in ["i9", "ryzen 9", "ultra 9", "m3 max", "m4 max"]): return 5
    if any(x in cpu for x in ["i7", "ryzen 7", "ultra 7", "m3 pro", "m4 pro"]): return 4
    if any(x in cpu for x in ["i5", "ryzen 5", "ultra 5", "m3", "m4", "m2", "m1"]): return 3
    if any(x in cpu for x in ["i3", "ryzen 3"]):                                      return 2
    if any(x in cpu for x in ["celeron", "pentium", "n4", "n5"]):                     return 1
    return 2  # unknown → average

# ─────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────

def _score_laptop(lp: dict, cfg: dict, intent_weight: float = 1.0) -> tuple[float, dict]:
    """
    بيحسب score للاب توب بناءً على intent config.
    بيرجع (total_score, breakdown) للـ debugging.
    """
    weights  = cfg.get("weights", {})
    breakdown = {}

    title   = str(lp.get("title",   "")).lower()
    cpu     = str(lp.get("cpu",     "")).lower()
    gpu     = str(lp.get("gpu",     "")).lower()
    ram_txt = str(lp.get("ram",     "")).lower()
    storage = str(lp.get("storage", "")).lower()
    screen  = str(lp.get("screen",  "")).lower()

    try:
        price = float(lp.get("price", 0))
    except (ValueError, TypeError):
        price = 0.0

    ram_gb     = extract_ram_gb(ram_txt)
    storage_gb = extract_storage_gb(storage)
    screen_in  = extract_screen_in(screen)
    cpu_tier   = _cpu_tier(cpu)

    # ── RAM ──────────────────────────────────────────────────
    min_ram = cfg.get("min_ram", 0)
    ram_score = 0
    if ram_gb >= min_ram and min_ram > 0:
        ram_score = weights.get("ram", 0)
        # bonus لو RAM فوق الـ minimum
        extra = (ram_gb - min_ram) // 8
        ram_score += min(extra, 3)  # max bonus = 3
    elif ram_gb > 0:
        # أقل من المطلوب → penalty بس مش صفر
        ratio     = ram_gb / max(min_ram, 1)
        ram_score = weights.get("ram", 0) * ratio * 0.5
    breakdown["ram"] = round(ram_score, 2)

    # ── GPU ──────────────────────────────────────────────────
    gpu_score = 0
    for g in cfg.get("gpu_keywords", ()):
        if g in gpu:
            gpu_score += weights.get("gpu", 0)
            break  # match واحد يكفي
        if g in title:
            gpu_score += weights.get("gpu", 0) * 0.6
            break
    breakdown["gpu"] = round(gpu_score, 2)

    # ── CPU ──────────────────────────────────────────────────
    cpu_score = (cpu_tier / 5) * weights.get("cpu", 0)
    # bonus لو CPU في الـ preferred list
    for pref in cfg.get("preferred_cpu", ()):
        if pref in cpu:
            cpu_score += 2
            break
    breakdown["cpu"] = round(cpu_score, 2)

    # ── SSD ──────────────────────────────────────────────────
    ssd_score = 0
    if "ssd" in storage or "nvme" in storage or "pcie" in storage:
        ssd_score += weights.get("ssd", 0)
    min_stg = cfg.get("min_storage_gb", 0)
    if storage_gb >= min_stg and min_stg > 0:
        ssd_score += 2
    breakdown["ssd"] = round(ssd_score, 2)

    # ── Screen ───────────────────────────────────────────────
    screen_score = 0
    base_screen  = weights.get("screen", 0)
    if cfg.get("prefer_large_screen") and screen_in >= 15.6:
        screen_score = base_screen
    elif cfg.get("prefer_small_screen") and 0 < screen_in <= 14:
        screen_score = base_screen
    elif base_screen > 0:
        screen_score = base_screen * 0.5  # neutral
    breakdown["screen"] = round(screen_score, 2)

    # ── Boost words ─────────────────────────────────────────
    boost_score = 0
    searchable  = f"{title} {cpu} {gpu}"
    for word in cfg.get("boost_words", ()):
        if word in searchable:
            boost_score += 3
    breakdown["boost"] = round(boost_score, 2)

    # ── Price ────────────────────────────────────────────────
    price_score = 0
    max_price   = cfg.get("max_price")
    if max_price:
        if price <= max_price:
            price_score = 5
        else:
            price_score = max(0, 5 - (price - max_price) / max_price * 5)
    breakdown["price"] = round(price_score, 2)

    # ── Total ────────────────────────────────────────────────
    raw_total = sum(breakdown.values())
    total     = raw_total * intent_weight
    breakdown["total"] = round(total, 2)

    return total, breakdown


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def smart_search(query: str, laptops: list, limit: int = 10) -> list:
    """
    بحث ذكي بيفهم نية المستخدم ويرتب اللاب توبات بناءً عليها.

    Features:
    - Multi-intent: لو الـ query فيه أكتر من intent بيـ blend الـ scores
    - Fallback: لو مفيش intent بيرجع IR-style results
    - Score breakdown: كل laptop بيجيب تفاصيل الـ score
    """
    # ── Detect intents ───────────────────────────────────────
    intents_found = detect_intents(query, top_k=2)

    # لو مفيش intent → fallback: رجّع اللاب توبات كما هي
    if not intents_found:
        return laptops[:limit]

    # Normalize intent weights (الأقوى يبقى 1.0 والتاني أقل)
    max_score = intents_found[0][1]
    intent_weights = [
        (name, score / max_score)
        for name, score in intents_found
    ]

    # ── Score each laptop ────────────────────────────────────
    scored = []
    for lp in laptops:
        total_score = 0.0
        combined_breakdown = {}

        for intent_name, i_weight in intent_weights:
            cfg   = INTENTS[intent_name]
            score, breakdown = _score_laptop(lp, cfg, i_weight)
            total_score += score
            # merge breakdowns
            for k, v in breakdown.items():
                combined_breakdown[k] = combined_breakdown.get(k, 0) + v

        item = {
            **lp,
            "ai_score":       round(total_score, 2),
            "intent":         intents_found[0][0],  # primary intent
            "score_breakdown": combined_breakdown,   # للـ debugging
        }
        scored.append(item)

    # ── Sort ─────────────────────────────────────────────────
    scored.sort(key=itemgetter("ai_score"), reverse=True)

    # ── Diversity: منع تكرار نفس الـ brand في أول 5 نتائج ───
    top      = []
    rest     = []
    seen_brands = set()

    for item in scored:
        brand = str(item.get("brand", "")).strip().lower()
        if not brand or brand in ("", "nan", "unknown"):
            # لو مفيش brand → حدد من الـ title
            title = str(item.get("title", "")).lower()
            for b in ["asus", "lenovo", "hp", "dell", "acer", "apple", "msi", "samsung", "huawei"]:
                if b in title:
                    brand = b
                    break

        if len(top) < limit and brand not in seen_brands:
            top.append(item)
            if brand:
                seen_brands.add(brand)
        else:
            rest.append(item)

    result = top + rest
    return result[:limit]
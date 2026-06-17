
import os, json
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from nlp_search import smart_search
from query_parser import normalize_query, expand_query, extract_specs_from_query, hard_filter

# ── bootstrap: copy data file so modules can find it ─────────────────────────
import shutil, pathlib
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
src = pathlib.Path("laptops.csv")
if not src.exists():
    src = pathlib.Path("/mnt/user-data/uploads/laptops.csv")
dst = DATA_DIR / "laptops.csv"
if not dst.exists() and src.exists():
    shutil.copy(src, dst)

from model    import get_all_laptops, get_recommendations, get_laptop, get_artifact
from ir_search import search as ir_search
from eda      import get_eda_stats

app = Flask(__name__)

FALLBACK_CARD = "data:image/svg+xml,%3Csvg%20xmlns%3D'http%3A//www.w3.org/2000/svg'%20width%3D'280'%20height%3D'140'%20viewBox%3D'0%200%20280%20140'%3E%3Crect%20width%3D'280'%20height%3D'140'%20fill%3D'%231a1d2e'/%3E%3Ctext%20x%3D'140'%20y%3D'60'%20text-anchor%3D'middle'%20fill%3D'%237c3aed'%20font-size%3D'40'%3E%F0%9F%92%BB%3C/text%3E%3Ctext%20x%3D'140'%20y%3D'95'%20text-anchor%3D'middle'%20fill%3D'%2364748b'%20font-size%3D'13'%20font-family%3D'sans-serif'%3ENo%20Image%3C/text%3E%3C/svg%3E"
FALLBACK_DETAIL = "data:image/svg+xml,%3Csvg%20xmlns%3D'http%3A//www.w3.org/2000/svg'%20width%3D'300'%20height%3D'220'%20viewBox%3D'0%200%20300%20220'%3E%3Crect%20width%3D'300'%20height%3D'220'%20fill%3D'%231a1d2e'/%3E%3Ctext%20x%3D'150'%20y%3D'100'%20text-anchor%3D'middle'%20fill%3D'%237c3aed'%20font-size%3D'56'%3E%F0%9F%92%BB%3C/text%3E%3Ctext%20x%3D'150'%20y%3D'145'%20text-anchor%3D'middle'%20fill%3D'%2364748b'%20font-size%3D'14'%20font-family%3D'sans-serif'%3ENo%20Image%3C/text%3E%3C/svg%3E"

# ── shared CSS/JS ──────────────────────────────────────────────────────────────
BASE_STYLE = """
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
a{color:inherit;text-decoration:none}
.nav{background:#1a1d2e;border-bottom:1px solid #2d3748;padding:0 2rem;display:flex;align-items:center;gap:2rem;height:60px;position:sticky;top:0;z-index:100}
.nav-logo{font-size:1.2rem;font-weight:700;color:#7c3aed;display:flex;align-items:center;gap:.5rem}
.nav a{font-size:.9rem;color:#94a3b8;transition:.2s}
.nav a:hover,.nav a.active{color:#e2e8f0}
.container{max-width:1200px;margin:0 auto;padding:2rem}
.hero{text-align:center;padding:4rem 2rem 2rem}
.hero h1{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#7c3aed,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.75rem}
.hero p{color:#64748b;font-size:1.05rem}
.search-box{background:#1a1d2e;border-radius:16px;padding:1.5rem;margin:2rem 0;border:1px solid #2d3748}
.search-box form{display:flex;flex-direction:column;gap:1rem}
.search-row{display:flex;gap:.75rem;flex-wrap:wrap}
.search-input{flex:1;min-width:200px;background:#0f1117;border:1px solid #2d3748;border-radius:10px;padding:.75rem 1rem;color:#e2e8f0;font-size:.95rem;outline:none;transition:.2s}
.search-input:focus{border-color:#7c3aed}
.search-input::placeholder{color:#475569}
.btn{padding:.75rem 1.5rem;border-radius:10px;border:none;cursor:pointer;font-size:.9rem;font-weight:600;transition:.2s}
.btn-primary{background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff}
.btn-primary:hover{opacity:.9;transform:translateY(-1px)}
.btn-secondary{background:#1e293b;color:#94a3b8;border:1px solid #2d3748}
.btn-secondary:hover{background:#2d3748;color:#e2e8f0}
select.search-input{cursor:pointer}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1.25rem;margin:1.5rem 0}
.card{background:#1a1d2e;border:1px solid #2d3748;border-radius:14px;padding:1.25rem;transition:.2s;cursor:pointer}
.card:hover{border-color:#7c3aed;transform:translateY(-2px);box-shadow:0 8px 24px rgba(124,58,237,.15)}
.card-img{width:100%;height:140px;object-fit:contain;background:#0f1117;border-radius:8px;margin-bottom:1rem}
.card-title{font-size:.85rem;font-weight:600;color:#e2e8f0;margin-bottom:.5rem;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card-specs{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:.75rem}
.tag{background:#0f1117;border:1px solid #2d3748;border-radius:6px;padding:.2rem .5rem;font-size:.75rem;color:#94a3b8}
.tag.highlight{border-color:#7c3aed;color:#a78bfa}
.card-price{font-size:1.1rem;font-weight:700;color:#7c3aed}
.card-score{font-size:.75rem;color:#475569;float:right;margin-top:.2rem}
.results-meta{color:#64748b;font-size:.9rem;margin-bottom:1rem}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:600;background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed44}
.detail-header{display:grid;grid-template-columns:300px 1fr;gap:2rem;margin-bottom:2rem;background:#1a1d2e;border:1px solid #2d3748;border-radius:16px;padding:1.5rem}
.detail-img{width:100%;height:220px;object-fit:contain;background:#0f1117;border-radius:10px}
.detail-specs{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1rem}
.spec-row{background:#0f1117;border-radius:8px;padding:.6rem .9rem;display:flex;justify-content:space-between;align-items:center}
.spec-label{color:#64748b;font-size:.8rem}
.spec-val{color:#e2e8f0;font-size:.85rem;font-weight:500}
.section-title{font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:1.25rem;display:flex;align-items:center;gap:.5rem}
.section-title span{background:linear-gradient(135deg,#7c3aed,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dash-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem}
.dash-grid.wide{grid-template-columns:repeat(2,1fr)}
@media(max-width:768px){.dash-grid{grid-template-columns:repeat(2,1fr)}.dash-grid.wide{grid-template-columns:1fr}.detail-header{grid-template-columns:1fr}.search-row{flex-direction:column}}
.stat-card{background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;padding:1.25rem;text-align:center}
.stat-num{font-size:1.8rem;font-weight:700;color:#7c3aed}
.stat-label{font-size:.8rem;color:#64748b;margin-top:.25rem}
.chart-card{background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;padding:1.25rem}
.chart-card h3{font-size:.9rem;font-weight:600;color:#94a3b8;margin-bottom:1rem}
.chart-wrap{position:relative;height:220px}
.table-card{background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;padding:1.25rem;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{color:#64748b;font-weight:500;padding:.6rem .75rem;text-align:left;border-bottom:1px solid #2d3748}
td{padding:.6rem .75rem;border-bottom:1px solid #1e293b;color:#e2e8f0}
tr:last-child td{border:none}
.empty{text-align:center;padding:3rem;color:#475569}
</style>
"""

NAV = """
<nav class="nav">
  <div class="nav-logo">💻 LaptopIQ</div>
  <a href="/" class="{h}">🏠 Home</a>
  <a href="/search" class="{s}">🔍 Search</a>
  <a href="/dashboard" class="{d}">📊 Dashboard</a>
</nav>
"""

def nav(active="home"):
    h = "active" if active == "home" else ""
    s = "active" if active == "search" else ""
    d = "active" if active == "dash" else ""
    return NAV.format(h=h, s=s, d=d)


def laptop_card(lp, extra=""):
    img = lp.get("image_main", "")
    if not img or str(img).strip() in ("", "nan", "None") or not str(img).startswith("http"):
        img = FALLBACK_CARD

    title_str = str(lp.get("title", ""))[:90]
    cpu_str   = str(lp.get("cpu",   ""))[:25]
    ram_str   = str(lp.get("ram",   ""))
    stg_str   = str(lp.get("storage", ""))
    scr_str   = str(lp.get("screen",  ""))

    try:
        price_val = float(lp.get("price", 0))
    except (ValueError, TypeError):
        price_val = 0.0

    return f"""
    <div class="card" onclick="location.href='/laptop/{lp.get('id', 0)}'">
      <img class="card-img"
           src="{img}"
           data-fb="{FALLBACK_CARD}"
           onerror="this.src=this.dataset.fb;this.onerror=null;"
           loading="lazy">
      <div class="card-title">{title_str}</div>
      <div class="card-specs">
        <span class="tag highlight">{cpu_str}</span>
        <span class="tag">{ram_str}</span>
        <span class="tag">{stg_str}</span>
        <span class="tag">{scr_str}</span>
      </div>
      <div class="card-price">SAR {price_val:,.2f}</div>
      {extra}
    </div>"""


# ── HOME ───────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    q       = request.args.get("q", "").strip()
    laptops = get_all_laptops()

    if q:
        # ── STEP 1: Parse & normalize the query ──────────────────────────────
        normalized_q = normalize_query(q)          # "RTX4060" → "rtx 4060"
        query_vars   = expand_query(q)              # ["rtx 4060", "rtx", "4060", ...]
        specs        = extract_specs_from_query(q)  # {gpu_model: "RTX 4060", ram_gb: 16}


        seen_ids  = set()
        ir_pool   = []
        for variant in query_vars:
            for lp in ir_search(variant, top_n=30):
                lid = lp.get("id", id(lp))
                if lid not in seen_ids:
                    seen_ids.add(lid)
                    ir_pool.append(lp)

        if specs:
            filtered_pool = hard_filter(ir_pool, specs)
            if not filtered_pool:
                filtered_pool = ir_pool
        else:
            filtered_pool = ir_pool

        # ── STEP 4: NLP re-ranking ────────────────────────────────────────────
        if filtered_pool:
            recommendations = smart_search(normalized_q, filtered_pool, limit=8)
        elif ir_pool:
            recommendations = ir_pool[:8]
        else:
            recommendations = smart_search(normalized_q, laptops, limit=8)

        title = f"🤖 AI Results for: {q}"
    else:
        try:
            recommendations = get_recommendations(0, top_n=8)
        except Exception:
            recommendations = laptops[:8]
        title = "🔥 Recommended For You"

    cards = "".join(
        laptop_card(lp, '<div class="card-score">🤖 AI Match</div>')
        for lp in recommendations
    )

    return render_template_string(f"""<!DOCTYPE html>
<html>
<head>
{BASE_STYLE}
<title>LaptopIQ</title>
</head>
<body>
{nav('home')}
<div class="container">
  <div class="hero">
    <h1>🤖 AI Laptop Recommendations</h1>
    <p>Smart laptop discovery powered by AI</p>
  </div>

  <div class="search-box">
    <form action="/" method="get">
      <div class="search-row">
        <input class="search-input" name="q"
               placeholder="Search gaming, RTX, Ryzen, editing..."
               value="{q}" autofocus>
        <button class="btn btn-primary" type="submit">🤖 AI Search</button>
        <a href="/dashboard" class="btn btn-secondary">📊 Dashboard</a>
      </div>
    </form>
  </div>

  <div class="section-title"><span>{title}</span></div>
  <div class="card-grid">{cards}</div>
</div>
</body>
</html>""")


# ── SEARCH ─────────────────────────────────────────────────────────────────────
@app.route("/search")
def search_page():
    q     = request.args.get("q", "").strip()
    min_p = request.args.get("min_price", "")
    max_p = request.args.get("max_price", "")
    ram_f = request.args.get("ram", "all")
    cpu_f = request.args.get("cpu", "all")

    all_laptops = get_all_laptops()
    ram_options = sorted(set(
        str(lp["ram"]) for lp in all_laptops if str(lp["ram"]) not in ("nan", "")
    ))
    ram_opts_html = '<option value="all">All RAM</option>' + "".join(
        f'<option value="{r}" {"selected" if r == ram_f else ""}>{r}</option>'
        for r in ram_options
    )

    cpu_keywords  = ["Intel", "AMD", "Apple", "Ryzen", "Core i", "Ultra"]
    cpu_opts_html = '<option value="all">All CPU</option>' + "".join(
        f'<option value="{c}" {"selected" if c == cpu_f else ""}>{c}</option>'
        for c in cpu_keywords
    )

    if q:
        kwargs = {}
        if min_p: kwargs["min_price"]  = float(min_p)
        if max_p: kwargs["max_price"]  = float(max_p)
        if ram_f != "all": kwargs["ram_filter"] = ram_f
        if cpu_f != "all": kwargs["cpu_filter"]  = cpu_f

        results = ir_search(q, top_n=20, **kwargs)
        meta    = f'<div class="results-meta">Found <strong>{len(results)}</strong> results for "<strong>{q}</strong>"</div>'
        cards_html = (
            "".join(laptop_card(lp, f'<div class="card-score">IR Score: {lp.get("ir_score", 0):.3f}</div>') for lp in results)
            if results else
            '<div class="empty"><p>No results found. Try different keywords.</p></div>'
        )
    else:
        featured   = all_laptops[:12]
        cards_html = "".join(laptop_card(lp) for lp in featured)
        meta       = '<div class="results-meta">Showing all laptops — enter a search term to filter</div>'

    return render_template_string(f"""<!DOCTYPE html>
<html>
<head>{BASE_STYLE}<title>Search — LaptopIQ</title></head>
<body>
{nav('search')}
<div class="container">
  <div class="section-title" style="margin-top:1.5rem"><span>🔍 Search Laptops</span></div>
  <div class="search-box">
    <form action="/search" method="get">
      <div class="search-row">
        <input class="search-input" name="q" value="{q}"
               placeholder="gaming, ryzen, i7, 16gb…" style="flex:2">
        <input class="search-input" name="min_price" value="{min_p}"
               placeholder="Min Price (SAR)" style="max-width:160px">
        <input class="search-input" name="max_price" value="{max_p}"
               placeholder="Max Price (SAR)" style="max-width:160px">
      </div>
      <div class="search-row">
        <select class="search-input" name="ram">{ram_opts_html}</select>
        <select class="search-input" name="cpu">{cpu_opts_html}</select>
        <button class="btn btn-primary" type="submit">🔍 Search</button>
        <a href="/search" class="btn btn-secondary">Clear</a>
      </div>
    </form>
  </div>
  {meta}
  <div class="card-grid">{cards_html}</div>
</div>
</body>
</html>""")


# ── LAPTOP DETAIL + RECOMMENDATIONS ───────────────────────────────────────────
@app.route("/laptop/<int:lid>")
def laptop_detail(lid):
    lp   = get_laptop(lid)
    recs = get_recommendations(lid, top_n=6)

    img = lp.get("image_main", "")
    if not img or str(img).strip() in ("", "nan", "None") or not str(img).startswith("http"):
        img = FALLBACK_DETAIL

    rec_cards = "".join(
        laptop_card(r, f'<div class="card-score">Similarity: {r["score"]:.0%}</div>')
        for r in recs
    )

    link_btn = (
        f'<a href="{lp["link"]}" target="_blank" class="btn btn-primary" style="margin-top:1rem">🛒 View on Store</a>'
        if lp.get("link") and str(lp["link"]) not in ("nan", "")
        else ""
    )

    title_short = str(lp.get("title", ""))[:40]

    rating_raw = lp.get("rating", 0)
    try:
        rating_int = int(float(rating_raw)) if str(rating_raw) not in ("nan", "", "None") else 0
    except (ValueError, TypeError):
        rating_int = 0
    rating_html = ("⭐" * rating_int) if rating_int > 0 else "N/A"

    return render_template_string(f"""<!DOCTYPE html>
<html>
<head>{BASE_STYLE}<title>{title_short} — LaptopIQ</title></head>
<body>
{nav()}
<div class="container">
  <div style="margin:1.5rem 0">
    <a href="/search" class="btn btn-secondary" style="padding:.5rem 1rem;font-size:.85rem">← Back</a>
  </div>
  <div class="detail-header">
    <div>
      <img class="detail-img"
           src="{img}"
           data-fb="{FALLBACK_DETAIL}"
           onerror="this.src=this.dataset.fb;this.onerror=null;">
      <div style="margin-top:.75rem;font-size:1.6rem;font-weight:700;color:#7c3aed">
        SAR {float(lp.get('price', 0)):,.2f}
      </div>
      {link_btn}
    </div>
    <div>
      <h1 style="font-size:1.1rem;font-weight:600;line-height:1.5;color:#e2e8f0;margin-bottom:1rem">
        {lp.get('title', '—')}
      </h1>
      <div class="detail-specs">
        <div class="spec-row"><span class="spec-label">CPU</span><span class="spec-val">{lp.get('cpu','—')}</span></div>
        <div class="spec-row"><span class="spec-label">RAM</span><span class="spec-val">{lp.get('ram','—')}</span></div>
        <div class="spec-row"><span class="spec-label">Storage</span><span class="spec-val">{lp.get('storage','—')}</span></div>
        <div class="spec-row"><span class="spec-label">Screen</span><span class="spec-val">{lp.get('screen','—')}</span></div>
        <div class="spec-row"><span class="spec-label">OS</span><span class="spec-val">{lp.get('os_version','—')}</span></div>
        <div class="spec-row"><span class="spec-label">Rating</span><span class="spec-val">{rating_html}</span></div>
      </div>
    </div>
  </div>

  <div class="section-title"><span>🤖 AI Recommendations</span> — Similar Laptops</div>
  <div class="card-grid">{rec_cards}</div>
</div>
</body>
</html>""")


# ── DASHBOARD ──────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    stats = get_eda_stats("data/laptops.csv")
    ov    = stats["overview"]

    js_code = f"""
    const C = (id, type, labels, data, colors) => {{
      const ctx = document.getElementById(id);
      if (!ctx) return;
      const palette = colors || ['#7c3aed','#06b6d4','#10b981','#f59e0b','#ef4444','#8b5cf6','#3b82f6','#ec4899'];
      new Chart(ctx, {{
        type: type,
        data: {{
          labels: labels,
          datasets: [{{
            data: data,
            backgroundColor: type === 'bar' ? palette[0] : palette,
            borderColor: type === 'bar' ? '#7c3aed' : 'transparent',
            borderWidth: 1,
            borderRadius: type === 'bar' ? 4 : 0,
            hoverOffset: 8
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: type !== 'bar', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
            tooltip: {{ callbacks: {{ label: c => type === 'bar' ? c.formattedValue : c.label + ': ' + c.formattedValue }} }}
          }},
          scales: type === 'bar' ? {{
            x: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }}, maxRotation: 30 }}, grid: {{ color: '#1e293b' }} }},
            y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
          }} : {{}}
        }}
      }});
    }};

    C('priceChart',   'bar',      {json.dumps(stats['price_dist']['labels'])},    {json.dumps(stats['price_dist']['values'])});
    C('typeChart',    'doughnut', {json.dumps(stats['laptop_types']['labels'])},  {json.dumps(stats['laptop_types']['values'])});
    C('ramChart',     'doughnut', {json.dumps(stats['ram_dist']['labels'])},      {json.dumps(stats['ram_dist']['values'])});
    C('cpuChart',     'pie',      {json.dumps(stats['cpu_brands']['labels'])},    {json.dumps(stats['cpu_brands']['values'])});
    C('storageChart', 'doughnut', {json.dumps(stats['storage_dist']['labels'])},  {json.dumps(stats['storage_dist']['values'])});
    C('pbrChart',     'bar',      {json.dumps(stats['price_by_ram']['labels'])},  {json.dumps(stats['price_by_ram']['values'])});
    C('kwChart',      'bar',      {json.dumps(stats['keyword_freq']['labels'])},  {json.dumps(stats['keyword_freq']['values'])});
    """

    top_exp  = "".join(f"<tr><td>{r['title']}</td><td>SAR {r['price']:,.2f}</td><td>{r['cpu']}</td></tr>" for r in stats['top_expensive'])
    top_chp  = "".join(f"<tr><td>{r['title']}</td><td>SAR {r['price']:,.2f}</td><td>{r['ram']}</td></tr>"  for r in stats['top_cheap'])

    html_content = f"""<!DOCTYPE html>
<html>
<head>{BASE_STYLE}<title>Dashboard — LaptopIQ</title></head>
<body>
{nav('dash')}
<div class="container">
  <div class="section-title" style="margin-top:1.5rem"><span>📊 Exploratory Data Analysis Dashboard</span></div>

  <div class="dash-grid">
    <div class="stat-card"><div class="stat-num">{ov['total']}</div><div class="stat-label">Total Laptops</div></div>
    <div class="stat-card"><div class="stat-num">SAR {ov['avg_price']:,.2f}</div><div class="stat-label">Avg Price</div></div>
    <div class="stat-card"><div class="stat-num">{ov['avg_ram_gb']:.1f} GB</div><div class="stat-label">Avg RAM</div></div>
    <div class="stat-card"><div class="stat-num">{ov['unique_cpu']}</div><div class="stat-label">Unique CPUs</div></div>
  </div>

  <div class="dash-grid wide">
    <div class="chart-card"><h3>💰 Price Distribution (SAR)</h3><div class="chart-wrap"><canvas id="priceChart"></canvas></div></div>
    <div class="chart-card"><h3>🏷️ Laptop Types</h3><div class="chart-wrap"><canvas id="typeChart"></canvas></div></div>
  </div>

  <div class="dash-grid wide">
    <div class="chart-card"><h3>💾 RAM Distribution</h3><div class="chart-wrap"><canvas id="ramChart"></canvas></div></div>
    <div class="chart-card"><h3>🖥️ CPU Brand Breakdown</h3><div class="chart-wrap"><canvas id="cpuChart"></canvas></div></div>
  </div>

  <div class="dash-grid wide">
    <div class="chart-card"><h3>💿 Storage Distribution</h3><div class="chart-wrap"><canvas id="storageChart"></canvas></div></div>
    <div class="chart-card"><h3>📐 Average Price by RAM</h3><div class="chart-wrap"><canvas id="pbrChart"></canvas></div></div>
  </div>

  <div class="chart-card" style="margin-bottom:2rem">
    <h3>🔤 Top Keywords in Titles</h3>
    <div class="chart-wrap" style="height:180px"><canvas id="kwChart"></canvas></div>
  </div>

  <div class="dash-grid wide" style="margin-bottom:2rem">
    <div class="table-card">
      <h3 style="color:#94a3b8;font-size:.9rem;margin-bottom:1rem">💎 Top 5 Most Expensive</h3>
      <table><tr><th>Laptop</th><th>Price</th><th>CPU</th></tr>{top_exp}</table>
    </div>
    <div class="table-card">
      <h3 style="color:#94a3b8;font-size:.9rem;margin-bottom:1rem">🏷️ Top 5 Most Affordable</h3>
      <table><tr><th>Laptop</th><th>Price</th><th>RAM</th></tr>{top_chp}</table>
    </div>
  </div>
</div>
<script>{js_code}</script>
</body>
</html>"""

    return render_template_string(html_content)


# ── JSON APIs ──────────────────────────────────────────────────────────────────
@app.route("/api/search")
def api_search():
    q       = request.args.get("q", "")
    results = ir_search(q, top_n=int(request.args.get("n", 10)))
    return jsonify(results)

@app.route("/api/recommend/<int:lid>")
def api_recommend(lid):
    return jsonify(get_recommendations(lid, top_n=int(request.args.get("n", 5))))

@app.route("/api/eda")
def api_eda():
    return jsonify(get_eda_stats("data/laptops.csv"))


# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  LaptopIQ — starting up …")
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
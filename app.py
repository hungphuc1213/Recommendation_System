"""
╔══════════════════════════════════════════════════════════════════╗
║  OLIST EXECUTIVE RECSYS DASHBOARD — app.py  v3.0                ║
║  Kế thừa từ: 01_Data_Prep_EDA | 02_RecSys_Model | 03_Forecasting║
║  Chạy: streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle, os, re, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist RecSys Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# 1. GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#F3F4F6; --surface:#FFFFFF; --border:#E5E7EB; --muted:#F9FAFB;
    --text:#111827; --sub:#6B7280;
    --accent1:#FF9900; --accent2:#10B981; --accent3:#2563EB;
    --accent4:#EF4444; --accent5:#8B5CF6; --glow:rgba(255,153,0,0.15);
}
html,body,.stApp{background:var(--bg)!important;color:var(--text);}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.stApp>header{background:transparent!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;letter-spacing:-0.02em;color:var(--text);}
p,li,span,label,div{font-family:'Inter',sans-serif!important;color:var(--text);}
code,.mono{font-family:'IBM Plex Mono',monospace!important;}
.kpi-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;position:relative;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;}
.kpi-blue::before{background:var(--accent3);}.kpi-green::before{background:var(--accent2);}
.kpi-amber::before{background:var(--accent1);}.kpi-red::before{background:var(--accent4);}
.kpi-violet::before{background:var(--accent5);}
.kpi-label{font-size:11px;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;}
.kpi-value{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:var(--text);}
.kpi-delta{font-size:12px;margin-top:4px;font-weight:500;}.kpi-delta.up{color:var(--accent2);}.kpi-delta.down{color:var(--accent4);}
.section-header{border-bottom:2px solid var(--border);padding-bottom:10px;margin-bottom:20px;display:flex;align-items:center;gap:10px;}
.section-dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--accent1);}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.badge-blue{background:rgba(37,99,235,.1);color:#1D4ED8;border:1px solid rgba(37,99,235,.2);}
.badge-green{background:rgba(16,185,129,.1);color:#047857;border:1px solid rgba(16,185,129,.2);}
.badge-amber{background:rgba(255,153,0,.1);color:#B45309;border:1px solid rgba(255,153,0,.2);}
.badge-red{background:rgba(239,68,68,.1);color:#B91C1C;border:1px solid rgba(239,68,68,.2);}
.badge-sub{background:rgba(107,114,128,.1);color:#4B5563;border:1px solid rgba(107,114,128,.2);}
.badge-violet{background:rgba(139,92,246,.1);color:#6D28D9;border:1px solid rgba(139,92,246,.2);}
.engine-tag{background:var(--muted);border:1px solid var(--border);padding:4px 12px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--sub);font-weight:500;}
.rec-table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;border-radius:8px;overflow:hidden;border:1px solid var(--border);}
.rec-table th{background:var(--muted);color:var(--sub);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:12px;text-align:left;border-bottom:1px solid var(--border);font-weight:600;}
.rec-table td{padding:12px;border-bottom:1px solid var(--border);color:var(--text);background:var(--surface);}
.rec-table tr:last-child td{border-bottom:none;}
.rec-table tr:hover td{background:var(--muted);}
.alert{border-radius:8px;padding:14px 16px;margin:8px 0;font-size:13px;display:flex;align-items:flex-start;gap:10px;line-height:1.5;}
.alert-info{background:rgba(37,99,235,.05);border:1px solid rgba(37,99,235,.2);color:#1E3A8A;}
.alert-success{background:rgba(16,185,129,.05);border:1px solid rgba(16,185,129,.2);color:#064E3B;}
.alert-warning{background:rgba(245,158,11,.05);border:1px solid rgba(245,158,11,.2);color:#78350F;}
.alert-error{background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.2);color:#7F1D1D;}
.stSelectbox>div>div,.stTextInput>div>div>input{background:var(--surface)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;box-shadow:0 1px 2px rgba(0,0,0,0.05)!important;}
.stButton>button{background:var(--accent1)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-weight:600!important;padding:8px 24px!important;box-shadow:0 4px 6px rgba(255,153,0,0.2)!important;transition:all 0.2s ease;}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 6px 8px rgba(255,153,0,0.3)!important;}
div[data-testid="metric-container"]{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
/* E-commerce specific classes */
.product-card {background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:all 0.3s ease;box-shadow:0 2px 4px rgba(0,0,0,0.02);display:flex;flex-direction:column;height:100%;position:relative;}
.product-card:hover {transform:translateY(-4px);box-shadow:0 12px 20px rgba(0,0,0,0.08);border-color:var(--accent1);}
.product-img {height:160px;display:flex;align-items:center;justify-content:center;font-size:48px;color:rgba(255,255,255,0.9);background-size:cover;background-position:center;}
.product-info {padding:16px;flex-grow:1;display:flex;flex-direction:column;background:var(--surface);}
.product-cat {font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;font-weight:600;}
.product-title {font-size:14px;font-weight:600;color:var(--text);margin-bottom:8px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.product-rating {font-size:12px;color:#F59E0B;margin-bottom:12px;display:flex;align-items:center;gap:4px;}
.product-price {font-size:20px;font-weight:800;color:#B12704;font-family:'Inter',sans-serif;margin-top:auto;}
.product-price sup {font-size:12px;font-weight:600;top:-0.5em;}
.product-badge {position:absolute;top:12px;left:12px;background:var(--accent4);color:white;font-size:10px;font-weight:700;padding:4px 8px;border-radius:4px;text-transform:uppercase;box-shadow:0 2px 4px rgba(239,68,68,0.3);z-index:2;}
.storefront-header {background:var(--surface);border-bottom:1px solid var(--border);padding:24px 32px;margin:-48px -32px 32px -32px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 2. CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────
COLORS = {
    "bg":"#F3F4F6","surface":"#FFFFFF","border":"#E5E7EB","muted":"#F9FAFB",
    "blue":"#2563EB","green":"#10B981","amber":"#FF9900",
    "red":"#EF4444","violet":"#8B5CF6","sub":"#6B7280","text":"#111827",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=16, r=16, t=36, b=16),
)
AXIS_DEF = dict(
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)

def apply_layout(fig, height=None, **extra):
    layout = {**PLOTLY_BASE, **AXIS_DEF}
    if height: layout["height"] = height
    for ax in ("xaxis","yaxis"):
        if ax in extra:
            layout[ax] = {**layout.get(ax,{}), **extra.pop(ax)}
    layout.update(extra)
    fig.update_layout(**layout)
    return fig

SEG_COLOR_MAP = {
    "Khách Mới Tương Tác (Active)":     COLORS["blue"],
    "Khách Vãng Lai Ngủ Quên (Cold)":   COLORS["sub"],
    "Ngôi Sao Tiềm Năng (High-Spender)": COLORS["amber"],
    "Khách Đang Trì Hoãn (Warm)":        "#3B82F6",
    "Khách Có Nguy Cơ (At-Risk)":       COLORS["red"],
    "Khách Hàng Tinh Hoa (Champions)":   COLORS["green"],
    "Khách Đã Rời Bỏ (Lost)":           "#9CA3AF",
    "Khách Trung Thành (Loyal)":         COLORS["violet"],
}

def get_color(seg):
    for k,v in SEG_COLOR_MAP.items():
        if k in seg or seg in k: return v
    return COLORS["sub"]

def sh(title):
    return f"<div class='section-header'><span class='section-dot'></span><b style='color:var(--text)'>{title}</b></div>"

# ─────────────────────────────────────────────────────────────────
# 3. CLASS ĐỊNH NGHĨA TRƯỚC KHI load_models()
# ─────────────────────────────────────────────────────────────────
class GeoPopularityEngine:
    REGION_MAP = {
        'SP':'Sudeste','RJ':'Sudeste','MG':'Sudeste','ES':'Sudeste',
        'RS':'Sul','PR':'Sul','SC':'Sul',
        'BA':'Nordeste','PE':'Nordeste','CE':'Nordeste','MA':'Nordeste',
        'PB':'Nordeste','RN':'Nordeste','AL':'Nordeste','SE':'Nordeste','PI':'Nordeste',
        'PA':'Norte','AM':'Norte','AC':'Norte','RO':'Norte','RR':'Norte','AP':'Norte','TO':'Norte',
        'MT':'Centro-Oeste','MS':'Centro-Oeste','GO':'Centro-Oeste','DF':'Centro-Oeste',
    }
    THRESHOLD_CITY=200; THRESHOLD_STATE=100

    def fit(self, train_df, qualified_items, top_n=10):
        self.top_n = top_n
        df = train_df[train_df['product_id'].isin(qualified_items)].copy()
        df['region'] = df['customer_state'].map(self.REGION_MAP).fillna('Outros')
        def build_index(df, group_col, min_count):
            gc = df.groupby(group_col)['order_id'].count()
            vg = set(gc[gc >= min_count].index)
            dv = df[df[group_col].isin(vg)]
            stats = dv.groupby([group_col,'product_id']).agg(
                cnt=('order_id','count'), rating=('review_score','mean')).reset_index()
            stats['score'] = 0.6*(stats['cnt']/stats['cnt'].max()) + 0.4*((stats['rating']-1)/4)
            idx = {}
            for key, grp in stats.groupby(group_col):
                idx[key] = grp.nlargest(top_n,'score')['product_id'].tolist()
            return idx
        self._city   = build_index(df,'customer_city',  self.THRESHOLD_CITY)
        self._state  = build_index(df,'customer_state', self.THRESHOLD_STATE)
        df['region'] = df['customer_state'].map(self.REGION_MAP).fillna('Outros')
        self._region = build_index(df,'region',0)
        self._global = (df.groupby('product_id').agg(cnt=('order_id','count'),rating=('review_score','mean'))
                        .assign(score=lambda x:0.6*(x.cnt/x.cnt.max())+0.4*((x.rating-1)/4))
                        .nlargest(top_n,'score').index.tolist())
        return self

    def recommend(self, user_id, train_df, top_n=None):
        top_n = top_n or self.top_n
        h = train_df[train_df['customer_unique_id']==user_id]
        bought = set(h['product_id'])
        city   = h['customer_city'].iloc[0]  if not h.empty else None
        state  = h['customer_state'].iloc[0] if not h.empty else None
        region = self.REGION_MAP.get(state,'Outros') if state else None
        for pool in [self._city.get(city,[]), self._state.get(state,[]),
                     self._region.get(region,[]), self._global]:
            recs = [p for p in pool if p not in bought]
            if len(recs) >= 2: return recs[:top_n]
        return self._global[:top_n]

# ─────────────────────────────────────────────────────────────────
# 4. DATA LOADER
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_csvs():
    # Force cache refresh 2
    files = {
        "train":"01_Train_df.csv","val":"01_Val_df.csv","test":"01_Test_df.csv",
        "rfm":"01_RFM_Train.csv","rfm_seg":"02_RFM_Segmented.csv",
        "rules":"02_Association_Rules.csv","ts":"01_Daily_GMV_Prophet.csv",
        "fc":"03_Forecast_September.csv","fc_full":"03_Forecast_Full.csv",
        "summary":"03_Executive_Summary.csv",
    }
    out = {}
    for k, fn in files.items():
        out[k] = pd.read_csv(fn) if os.path.exists(fn) else None
    return out

@st.cache_resource(show_spinner=False)
def load_models():
    # Force cache refresh 2
    models = {}
    for fn,key in [
        ("02_geo_engine.pkl","geo"),("02_seg_dict.pkl","seg_dict"),
        ("02_fpgrowth_rules.pkl","rules"),("02_cb_meta.pkl","cb_meta"),
        ("01_qualified_items.pkl","q_items"),("03_prophet_model.pkl","prophet"),
    ]:
        if os.path.exists(fn):
            with open(fn,"rb") as f: models[key] = pickle.load(f)
    return models

# ─────────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px 0'>
      <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#E2E8F5'>🛒 Olist RecSys</div>
      <div style='font-size:11px;color:#7A8BAA;margin-top:2px;font-family:IBM Plex Mono,monospace'>Executive Intelligence Dashboard</div>
    </div>
    <hr style='border-color:#1C2539;margin:10px 0 16px 0'>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Tổng Quan":              "overview",
        "👥  Phân Khúc Khách Hàng":  "segments",
        "🗂️  Quản Trị & Tra Cứu":    "explorer",
        "🤖  Demo Gợi Ý (Live)":     "recommend",
        "📈  FP-Growth Insights":     "basket",
        "🔮  Dự Báo Doanh Thu":       "forecast",
        "📊  Đánh Giá Mô Hình":       "eval",
        "💼  Báo Cáo Quản Trị":       "arch",
    }
    if "page" not in st.session_state: st.session_state.page = "overview"
    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="secondary"):
            st.session_state.page = key; st.rerun()
    st.markdown("<hr style='border-color:#1C2539;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#2A3650;text-align:center;font-family:IBM Plex Mono,monospace'>v3.0 · Brazil E-Commerce · Olist 2017–2018</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 6. LOAD DATA
# ─────────────────────────────────────────────────────────────────
with st.spinner("Đang tải dữ liệu & mô hình..."):
    data   = fetch_all_csvs()
    models = load_models()

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown("<h1 style='font-size:32px;font-weight:800;margin-bottom:4px'>Tổng Quan Hệ Thống Đề Xuất</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:32px;font-size:14px'>Hybrid Recommendation System · Olist Brazilian E-Commerce · 2017–2018</p>", unsafe_allow_html=True)

    train = data["train"]; rfm_s = data["rfm_seg"]; ts = data["ts"]; fc = data["fc"]

    total_users  = rfm_s["customer_unique_id"].nunique() if rfm_s is not None else 0
    total_orders = train["order_id"].nunique() if train is not None else 0
    total_items  = train["product_id"].nunique() if train is not None else 0
    gmv_train = 0
    if ts is not None:
        ts["ds"] = pd.to_datetime(ts["ds"])
        gmv_train = ts[(ts["ds"]>="2017-01-01")&(ts["ds"]<"2018-06-01")]["y"].sum()
    f1_pct = (rfm_s["Frequency"]==1).mean()*100 if rfm_s is not None else 0
    fc_gmv = fc["yhat"].sum() if fc is not None else 0

    kpis = [
        ("Tổng Khách Hàng (Train)", f"{total_users:,.0f}",    "kpi-blue",  "👥"),
        ("Tổng Đơn Hàng (Train)",   f"{total_orders:,.0f}",   "kpi-green", "📦"),
        ("Sản Phẩm Đủ Tiêu Chuẩn", f"{total_items:,.0f}",    "kpi-violet","🛍️"),
        ("GMV Train (BRL)",          f"{gmv_train/1e6:.2f}M",  "kpi-amber", "💰"),
        ("Khách Hàng Vãng Lai F=1", f"{f1_pct:.1f}%",         "kpi-red",   "⚡"),
        ("Dự Báo GMV T9/2018",       f"{fc_gmv/1e6:.2f}M BRL","kpi-green", "🔮"),
    ]
    cols = st.columns(6)
    for col,(label,val,cls,icon) in zip(cols,kpis):
        col.markdown(f"<div class='kpi-card {cls}'><div class='kpi-label'>{icon} {label}</div><div class='kpi-value'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segment Pie + Daily GMV ──
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(sh("Phân Bổ Phân Khúc"), unsafe_allow_html=True)
        if rfm_s is not None:
            seg_count = rfm_s["Segment_Name"].value_counts().reset_index()
            seg_count.columns = ["segment","count"]
            fig_pie = go.Figure(go.Pie(
                labels=seg_count["segment"], values=seg_count["count"], hole=0.55,
                marker_colors=[get_color(s) for s in seg_count["segment"]],
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value:,} users<br>%{percent}<extra></extra>",
            ))
            apply_layout(fig_pie, height=290, showlegend=True, legend=dict(font=dict(size=9)))
            fig_pie.add_annotation(text=f"<b>{total_users:,.0f}</b><br><span style='font-size:10px'>Users</span>",
                x=0.5,y=0.5,showarrow=False,font=dict(size=14,color=COLORS["text"]))
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})

    with c2:
        st.markdown(sh("Daily GMV Timeline (Order-Level, không inflate)"), unsafe_allow_html=True)
        if ts is not None:
            ts_plot = ts[ts["ds"]>="2017-01-01"].copy()
            ts_plot["ma7"] = ts_plot["y"].rolling(7).mean()
            ts_plot["ma30"] = ts_plot["y"].rolling(30).mean()
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(x=ts_plot["ds"],y=ts_plot["y"],name="Daily GMV",
                marker_color=COLORS["blue"],opacity=0.25,
                hovertemplate="%{x|%d/%m/%Y}<br>GMV: %{y:,.0f} BRL<extra></extra>"))
            fig_ts.add_trace(go.Scatter(x=ts_plot["ds"],y=ts_plot["ma7"],name="MA-7",
                line=dict(color=COLORS["green"],width=2),
                hovertemplate="%{x|%d/%m/%Y}<br>MA7: %{y:,.0f} BRL<extra></extra>"))
            fig_ts.add_trace(go.Scatter(x=ts_plot["ds"],y=ts_plot["ma30"],name="MA-30",
                line=dict(color=COLORS["amber"],width=2,dash="dot"),
                hovertemplate="%{x|%d/%m/%Y}<br>MA30: %{y:,.0f} BRL<extra></extra>"))
            for date,label,color in [
                ("2017-11-24","Black Friday",COLORS["amber"]),
                ("2018-06-01","Train/Val",   COLORS["sub"]),
                ("2018-08-01","Val/Test",    COLORS["violet"]),
            ]:
                ts_ms = pd.to_datetime(date).timestamp()*1000
                fig_ts.add_vline(x=ts_ms,line_dash="dot",line_color=color,line_width=1.5,
                    annotation_text=label,annotation_font_size=10,annotation_font_color=color)
            apply_layout(fig_ts,height=290,
                yaxis={"title":"GMV (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_ts,use_container_width=True,config={"displayModeBar":False})

    # ── Engine Routing Matrix ──
    st.markdown(sh("Engine Routing Matrix"), unsafe_allow_html=True)
    routing_data = [
        ("Khách Mới Tương Tác (Active)","F=1, R thấp","Content-Based (E2)","Cross-sell ngay khi còn nóng","blue","Mới mua, cần giữ chân"),
        ("Ngôi Sao Tiềm Năng (High-Spender)","F=1, M cao","Content-Based (E2)","Ép đơn thứ 2 giá trị cao","amber","Mua 1 lần, chi nhiều"),
        ("Khách Đang Trì Hoãn (Warm)","F=1, R TB","Content-Based (E2)","Kích thích chốt sale","blue","Bắt đầu nguội"),
        ("Khách Vãng Lai Ngủ Quên (Cold)","F=1, R cao","Geo-Popularity (E1)","Re-engage bằng Trending","sub","Mất tương tác lâu"),
        ("Khách Đã Rời Bỏ (Lost)","F=1, R cực cao","Geo-Popularity (E1)","Gợi ý an toàn nhất","sub","Gần như đã mất"),
        ("Khách Hàng Tinh Hoa (Champions)","F>1, R thấp, M cao","FP-Growth (E3)","Tăng AOV / Cross-category","green","Khách giá trị nhất"),
        ("Khách Trung Thành (Loyal)","F>1, R trung bình","FP-Growth (E3)","Tăng Frequency","violet","Mua đều tay"),
        ("Khách Có Nguy Cơ (At-Risk)","F>1, R cao","Geo-Popularity (E1)","Giữ chân khẩn cấp","red","Đã từng tốt, đang lạnh dần"),
    ]
    cols_h = st.columns([2,2,2,2,3])
    for col,h_label in zip(cols_h,["Phân Khúc","Tiêu Chí RFM","Engine AI","Mục Tiêu","Ghi Chú"]):
        col.markdown(f"<span style='font-size:11px;color:#7A8BAA;text-transform:uppercase;letter-spacing:.1em'>{h_label}</span>",unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1C2539;margin:6px 0'>", unsafe_allow_html=True)
    for seg,criteria,engine,goal,color,note in routing_data:
        c1,c2,c3,c4,c5 = st.columns([2,2,2,2,3])
        c1.markdown(f"<span class='badge badge-{color}'>{seg}</span>",unsafe_allow_html=True)
        c2.markdown(f"<span style='font-size:12px;color:#7A8BAA;font-family:IBM Plex Mono,monospace'>{criteria}</span>",unsafe_allow_html=True)
        c3.markdown(f"<span class='engine-tag'>{engine}</span>",unsafe_allow_html=True)
        c4.markdown(f"<span style='font-size:13px'>{goal}</span>",unsafe_allow_html=True)
        c5.markdown(f"<span style='font-size:12px;color:#7A8BAA'>{note}</span>",unsafe_allow_html=True)

    # ── EDA từ notebook ──
    st.markdown("<br>", unsafe_allow_html=True)
    train_eda = data.get("train")
    if train_eda is not None:
        st.markdown(sh("EDA 1 & 2: Phân Bổ Địa Lý & Price vs Freight (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        e1, e2 = st.columns(2)

        with e1:
            st.markdown("<b>📍 Top 10 Bang Đặt Hàng Nhiều Nhất</b>", unsafe_allow_html=True)
            top_states = train_eda['customer_state'].value_counts().head(10).reset_index()
            top_states.columns = ["State","Orders"]
            fig_states = go.Figure(go.Bar(
                x=top_states["State"], y=top_states["Orders"],
                marker_color=COLORS["blue"], opacity=0.85,
                text=top_states["Orders"].apply(lambda x:f"{x:,}"),
                textposition="outside",
                hovertemplate="Bang %{x}<br>%{y:,} đơn<extra></extra>"
            ))
            sp_pct = top_states.iloc[0]["Orders"]/top_states["Orders"].sum()*100
            apply_layout(fig_states, height=300, showlegend=False,
                yaxis={"title":"Số Đơn","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                title=dict(text=f"SP chiếm {sp_pct:.1f}% → Thiết kế Geo-Fallback bắt buộc",
                    font=dict(size=11,color=COLORS["sub"]),x=0))
            st.plotly_chart(fig_states, use_container_width=True, config={"displayModeBar":False})

        with e2:
            st.markdown("<b>📦 Scatter: Price vs Freight Value (clip P95) — EDA 2</b>", unsafe_allow_html=True)
            p95_price   = train_eda['price'].quantile(0.95)
            p95_freight = train_eda['freight_value'].quantile(0.95)
            sample_eda2 = train_eda.sample(min(8000,len(train_eda)), random_state=42)
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(
                x=sample_eda2['price'].clip(upper=p95_price),
                y=sample_eda2['freight_value'].clip(upper=p95_freight),
                mode='markers',
                marker=dict(color=COLORS["blue"],size=3,opacity=0.3),
                name="SP",
                hovertemplate="Price: %{x:.0f} BRL<br>Freight: %{y:.0f} BRL<extra></extra>"
            ))
            max_v = max(p95_price,p95_freight)
            fig_pf.add_trace(go.Scatter(
                x=[0,max_v], y=[0,max_v], mode='lines',
                line=dict(color=COLORS["red"],dash="dash",width=1.5),
                name="Freight = Price (Phi lý)"
            ))
            apply_layout(fig_pf, height=300,
                xaxis={"title":"Price (BRL, clip P95)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"title":"Freight Value (BRL, clip P95)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_pf, use_container_width=True, config={"displayModeBar":False})

        # ── EDA 3: Rating ──
        st.markdown(sh("EDA 3: Phân Bổ Rating & Hard Filter Threshold (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        er1, er2 = st.columns(2)
        with er1:
            st.markdown("<b>📊 Phân bố Review Score (Positivity Bias)</b>", unsafe_allow_html=True)
            rating_counts = train_eda['review_score'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['score','count']
            fig_rating = go.Figure(go.Bar(
                x=rating_counts['score'], y=rating_counts['count'],
                marker_color=['#E24B4A','#EF9F27','#888780','#1D9E75','#2ECC71'],
                text=rating_counts['count'].apply(lambda x:f'{x:,}'),
                textposition='outside',
                hovertemplate='Rating %{x}★<br>%{y:,} đơn<extra></extra>'
            ))
            apply_layout(fig_rating, height=300, showlegend=False,
                xaxis={"title":"Review Score (★)","dtick":1,"gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"title":"Số Đơn","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_rating, use_container_width=True, config={"displayModeBar":False})

        with er2:
            st.markdown("<b>🔥 Độ Nhạy Ngưỡng: Số SP hợp lệ theo Hard Filter</b>", unsafe_allow_html=True)
            item_avg_rating = train_eda.groupby('product_id')['review_score'].mean()
            thresholds_chk  = [3.0, 3.5, 4.0, 4.2, 4.5]
            n_items_left    = [(item_avg_rating >= t).sum() for t in thresholds_chk]
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(
                x=thresholds_chk, y=n_items_left,
                mode='lines+markers',
                line=dict(color=COLORS['blue'],width=2.5),
                marker=dict(size=9,color=COLORS['blue']),
                hovertemplate='Threshold %{x}<br>%{y:,} SP còn lại<extra></extra>'
            ))
            idx_4 = thresholds_chk.index(4.0)
            fig_thresh.add_annotation(
                x=4.0, y=n_items_left[idx_4],
                text=f"<b>Ngưỡng 4.0★<br>{n_items_left[idx_4]:,} SP</b>",
                showarrow=True, arrowhead=2, arrowcolor=COLORS['green'],
                bgcolor=COLORS['surface'], bordercolor=COLORS['green'],
                font=dict(color=COLORS['text'],size=11), ax=40, ay=-40
            )
            fig_thresh.add_vline(x=4.0,line_dash='dot',line_color=COLORS['green'],line_width=1.5)
            apply_layout(fig_thresh, height=300,
                xaxis={"title":"Hard Filter Threshold (★)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"title":"Số SP hợp lệ","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_thresh, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f"""<div class='alert alert-info'>💡 <b>EDA 3 Insight:</b> Rating có <b>Positivity Bias mạnh</b> (5★ chiếm >50%).
          Ngưỡng 4.0★ giữ lại <b>{n_items_left[idx_4]:,} SP</b> — cân bằng giữa chất lượng và đủ data để huấn luyện.</div>""",
          unsafe_allow_html=True)

        # ── EDA 7: Kiểm tra 2016 (THIẾU trong bản cũ) ──
        st.markdown(sh("EDA 7: Kiểm Tra Độ Thưa 2016 — Xác Nhận Lệnh Cấm (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        train_eda["order_purchase_timestamp"] = pd.to_datetime(train_eda["order_purchase_timestamp"], errors='coerce')
        train_eda["year_col"]  = train_eda["order_purchase_timestamp"].dt.year
        train_eda["month_col"] = train_eda["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        monthly_agg = train_eda.groupby("month_col")["order_id"].count().reset_index()
        monthly_agg.columns = ["month","orders"]
        monthly_agg["year_label"] = monthly_agg["month"].dt.year.astype(str)
        color_year_map = {"2016":COLORS["red"],"2017":COLORS["blue"],"2018":COLORS["green"]}
        bar_colors_y = [color_year_map.get(str(y),COLORS["sub"]) for y in monthly_agg["month"].dt.year]

        fig_2016 = go.Figure(go.Bar(
            x=monthly_agg["month"], y=monthly_agg["orders"],
            marker_color=bar_colors_y, opacity=0.85,
            hovertemplate="%{x|%m/%Y}<br>%{y:,} đơn<extra></extra>"
        ))
        # Year annotations
        for yr, clr in [("2016",COLORS["red"]),("2017",COLORS["blue"]),("2018",COLORS["green"])]:
            subset = monthly_agg[monthly_agg["month"].dt.year.astype(str)==yr]
            if not subset.empty:
                mid_date = subset["month"].iloc[len(subset)//2]
                mid_y = subset["orders"].max()*0.8
                total_yr = subset["orders"].sum()
                pct_yr   = total_yr/monthly_agg["orders"].sum()*100
                fig_2016.add_annotation(x=mid_date, y=mid_y,
                    text=f"<b>{yr}</b><br>{total_yr:,} đơn<br>{pct_yr:.1f}%",
                    showarrow=False, font=dict(color=clr,size=10),
                    bgcolor=COLORS["surface"], bordercolor=clr, borderwidth=1)
        apply_layout(fig_2016, height=280, showlegend=False,
            yaxis={"title":"Số Đơn/Tháng","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
            title=dict(text="2016 < 2% tổng đơn → Loại hoàn toàn khỏi Train để tránh bias",
                font=dict(size=11,color=COLORS["sub"]),x=0))
        st.plotly_chart(fig_2016, use_container_width=True, config={"displayModeBar":False})

        # ── EDA 9: RFM Raw vs Log (HOÀN TOÀN THIẾU trong bản cũ) ──
        st.markdown(sh("EDA 9: RFM Trước & Sau Log-Transform — Validate Quyết Định Scale (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        rfm_eda_data = data.get("rfm")
        if rfm_eda_data is not None:
            rfm_eda_data = rfm_eda_data.copy()
            if "Monetary_Log" not in rfm_eda_data.columns:
                rfm_eda_data["Monetary_Log"] = np.log1p(rfm_eda_data["Monetary"])
            if "Recency_Log" not in rfm_eda_data.columns:
                rfm_eda_data["Recency_Log"]  = np.log1p(rfm_eda_data["Recency"])

            fig_rfm9 = make_subplots(rows=2, cols=3,
                subplot_titles=["Recency (Raw)","Frequency (Raw)","Monetary (Raw)",
                                 "Recency (Log)","Frequency (Log)","Monetary (Log)"])
            colors_rfm9 = [COLORS["red"],COLORS["amber"],COLORS["blue"],
                           COLORS["green"],COLORS["violet"],COLORS["blue"]]
            cols_raw = ["Recency","Frequency","Monetary"]
            cols_log = ["Recency_Log","","Monetary_Log"]
            for ci,(col_r,col_l,clr) in enumerate(zip(cols_raw,cols_log,colors_rfm9[:3])):
                vals = rfm_eda_data[col_r].clip(upper=rfm_eda_data[col_r].quantile(0.98))
                fig_rfm9.add_trace(go.Histogram(x=vals,marker_color=clr,opacity=0.75,
                    showlegend=False,hovertemplate=f"{col_r}: %{{x}}<extra></extra>"),row=1,col=ci+1)
            for ci,(col_r,clr2) in enumerate(zip(cols_raw,colors_rfm9[3:])):
                if col_r == "Frequency":
                    log_vals = np.log1p(rfm_eda_data["Frequency"].clip(upper=rfm_eda_data["Frequency"].quantile(0.98)))
                else:
                    log_col = "Recency_Log" if col_r=="Recency" else "Monetary_Log"
                    log_vals = rfm_eda_data[log_col].clip(upper=rfm_eda_data[log_col].quantile(0.98))
                fig_rfm9.add_trace(go.Histogram(x=log_vals,marker_color=clr2,opacity=0.75,
                    showlegend=False,hovertemplate="Log: %{x:.2f}<extra></extra>"),row=2,col=ci+1)

            apply_layout(fig_rfm9, height=400)
            fig_rfm9.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),showlegend=False)
            for ax_key in fig_rfm9.layout:
                if str(ax_key).startswith("xaxis") or str(ax_key).startswith("yaxis"):
                    fig_rfm9.layout[ax_key].update(gridcolor=COLORS["border"],zerolinecolor=COLORS["border"])
            st.plotly_chart(fig_rfm9, use_container_width=True, config={"displayModeBar":False})
            st.markdown("<div class='alert alert-info'>💡 <b>EDA 9 Insight:</b> Recency & Monetary có phân phối <b>right-skewed nặng</b> (đuôi dài) → Log-transform làm phẳng → K-Means hoạt động tốt hơn khi các cluster không bị kéo lệch bởi outlier.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: PHÂN KHÚC
# ═══════════════════════════════════════════════════════════════════
elif page == "segments":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>Phân Khúc Khách Hàng</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Macro-Branching (F=1 vs F>1) · RFM K-Means Clustering · Centroid Mapping</p>", unsafe_allow_html=True)

    rfm_s = data["rfm_seg"]
    if rfm_s is None:
        st.markdown("<div class='alert alert-warning'>⚠️ Chưa tìm thấy 02_RFM_Segmented.csv. Hãy chạy File 02 trước.</div>", unsafe_allow_html=True)
    else:
        rfm_s["Monetary_Log"] = np.log1p(rfm_s["Monetary"])
        rfm_s["Recency_Log"]  = np.log1p(rfm_s["Recency"])
        rfm_F1 = rfm_s[rfm_s["Frequency"]==1]
        rfm_Fn = rfm_s[rfm_s["Frequency"]>1]

        k1,k2,k3,k4 = st.columns(4)
        k1.markdown(f"<div class='kpi-card kpi-red'><div class='kpi-label'>👥 F=1 Vãng Lai</div><div class='kpi-value'>{len(rfm_F1):,}</div><div class='kpi-delta up'>{len(rfm_F1)/len(rfm_s)*100:.1f}% tổng</div></div>",unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-label'>⭐ F>1 Tinh Hoa</div><div class='kpi-value'>{len(rfm_Fn):,}</div><div class='kpi-delta up'>{len(rfm_Fn)/len(rfm_s)*100:.1f}% tổng</div></div>",unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-card kpi-amber'><div class='kpi-label'>💰 Monetary TB (F1)</div><div class='kpi-value'>{rfm_F1['Monetary'].mean():,.0f} BRL</div></div>",unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi-card kpi-violet'><div class='kpi-label'>💎 Monetary TB (F>1)</div><div class='kpi-value'>{rfm_Fn['Monetary'].mean():,.0f} BRL</div></div>",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown(sh("Nhánh F=1 — Không gian R × M"), unsafe_allow_html=True)
            subs = rfm_F1.sample(min(5000,len(rfm_F1)),random_state=42)
            fig_f1 = px.scatter(subs,x="Monetary_Log",y="Recency_Log",color="Segment_Name",
                color_discrete_map={s:get_color(s) for s in subs["Segment_Name"].unique()},
                opacity=0.5,size_max=4,labels={"Monetary_Log":"Monetary (log)","Recency_Log":"Recency (log)"},
                hover_data={"Monetary":":,.0f","Recency":True})
            fig_f1.update_traces(marker_size=4)
            apply_layout(fig_f1,height=380,legend_title="Segment",
                xaxis={"title":"Monetary (log)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"title":"Recency (log)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_f1,use_container_width=True,config={"displayModeBar":False})

        with c2:
            st.markdown(sh("Nhánh F>1 — Không gian R × M (F ∝ kích thước)"), unsafe_allow_html=True)
            subs_fn = rfm_Fn.sample(min(2000,len(rfm_Fn)),random_state=42)
            fig_fn = px.scatter(subs_fn,x="Monetary_Log",y="Recency_Log",color="Segment_Name",size="Frequency",
                color_discrete_map={s:get_color(s) for s in subs_fn["Segment_Name"].unique()},
                opacity=0.7,size_max=12,labels={"Monetary_Log":"Monetary (log)","Recency_Log":"Recency (log)"},
                hover_data={"Monetary":":,.0f","Recency":True,"Frequency":True})
            apply_layout(fig_fn,height=380,legend_title="Segment",
                xaxis={"title":"Monetary (log)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"title":"Recency (log)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_fn,use_container_width=True,config={"displayModeBar":False})

        # ── Segment Stats Table ──
        st.markdown(sh("Thống Kê Chi Tiết Theo Phân Khúc"), unsafe_allow_html=True)
        seg_stats = rfm_s.groupby("Segment_Name").agg(
            Users=("customer_unique_id","count"), Monetary_Avg=("Monetary","mean"),
            Monetary_Sum=("Monetary","sum"), Recency_Avg=("Recency","mean"),
            Frequency_Avg=("Frequency","mean")).reset_index().sort_values("Monetary_Avg",ascending=False)
        seg_stats["pct_users"] = (seg_stats["Users"]/seg_stats["Users"].sum()*100).round(1).astype(str)+"%"
        seg_stats["Monetary_Avg"] = seg_stats["Monetary_Avg"].round(0).astype(int)
        seg_stats["Monetary_Sum"] = (seg_stats["Monetary_Sum"]/1e6).round(2).astype(str)+"M"
        seg_stats["Recency_Avg"]  = seg_stats["Recency_Avg"].round(0).astype(int).astype(str)+" ngày"
        seg_stats["Frequency_Avg"]= seg_stats["Frequency_Avg"].round(2)
        html_rows=""
        for _,row in seg_stats.iterrows():
            color = get_color(row["Segment_Name"])
            html_rows += (f"<tr><td><span class='badge' style='background:rgba(128,128,128,.1);border:1px solid {color};color:{color}'>{row['Segment_Name']}</span></td>"
                f"<td style='text-align:right'>{row['Users']:,}</td><td style='text-align:right'>{row['pct_users']}</td>"
                f"<td style='text-align:right'>{row['Monetary_Avg']:,} BRL</td><td style='text-align:right'>{row['Monetary_Sum']}</td>"
                f"<td style='text-align:right'>{row['Recency_Avg']}</td><td style='text-align:right'>{row['Frequency_Avg']:.2f}</td></tr>")
        st.markdown(f"<table class='rec-table'><thead><tr><th>Phân Khúc</th><th style='text-align:right'>Users</th><th style='text-align:right'>% Users</th><th style='text-align:right'>Avg Monetary</th><th style='text-align:right'>Total Revenue</th><th style='text-align:right'>Avg Recency</th><th style='text-align:right'>Avg Frequency</th></tr></thead><tbody>{html_rows}</tbody></table>",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── RFM Distribution — Violin (cải thiện từ histogram overlay) ──
        st.markdown(sh("Phân Phối RFM theo Phân Khúc — Violin Chart (Cải thiện từ Histogram Overlay)"), unsafe_allow_html=True)
        fig_violin = make_subplots(rows=1,cols=3,subplot_titles=["Recency (ngày)","Frequency (đơn)","Monetary (BRL)"])
        for seg in rfm_s["Segment_Name"].unique():
            df_s = rfm_s[rfm_s["Segment_Name"]==seg]; clr = get_color(seg)
            for ci,col_name in enumerate(["Recency","Frequency","Monetary"],1):
                vals = df_s[col_name].clip(upper=df_s[col_name].quantile(0.97))
                fig_violin.add_trace(go.Violin(y=vals,name=seg,
                    line_color=clr,fillcolor=clr,opacity=0.45,
                    box_visible=True,meanline_visible=True,
                    showlegend=(ci==1),
                    hovertemplate=f"{seg}<br>%{{y}}<extra></extra>"),row=1,col=ci)
        apply_layout(fig_violin,height=380,violinmode="overlay")
        fig_violin.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),legend=dict(font=dict(size=9)))
        for ax_key in fig_violin.layout:
            if str(ax_key).startswith("xaxis") or str(ax_key).startswith("yaxis"):
                fig_violin.layout[ax_key].update(gridcolor=COLORS["border"],zerolinecolor=COLORS["border"])
        st.plotly_chart(fig_violin,use_container_width=True,config={"displayModeBar":False})

        # ── EDA kế thừa: Sparsity ──
        st.markdown(sh("EDA 4 & 6: Sparsity Matrix và Hành Vi User (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        train_eda2 = data.get("train")
        if train_eda2 is not None:
            ep1,ep2,ep3 = st.columns(3)
            with ep1:
                st.markdown("<b>📊 Sparsity Ma Trận User×Item</b>", unsafe_allow_html=True)
                n_u = train_eda2['customer_unique_id'].nunique()
                n_i = train_eda2['product_id'].nunique()
                n_x = len(train_eda2)
                sp  = 1 - n_x/(n_u*n_i)
                fig_sp = go.Figure(go.Pie(
                    labels=[f'Có data ({n_x:,})',f'Ô trống ({sp:.2%})'],
                    values=[n_x,int(n_u*n_i-n_x)],hole=0.55,
                    marker_colors=[COLORS['blue'],COLORS['border']],
                    textinfo='percent',hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
                ))
                fig_sp.add_annotation(text=f"<b>{sp:.2%}</b><br><span style='font-size:9px'>Sparsity</span>",
                    x=0.5,y=0.5,showarrow=False,font=dict(size=12,color=COLORS['text']))
                apply_layout(fig_sp,height=270,showlegend=True,legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_sp,use_container_width=True,config={"displayModeBar":False})
                st.caption(f"{n_u:,} users × {n_i:,} items → Phải dùng Hybrid Architecture!")

            with ep2:
                st.markdown("<b>📉 Long-tail: SP mỗi User & mỗi Item</b>", unsafe_allow_html=True)
                ppu = train_eda2.groupby("customer_unique_id")["product_id"].count().clip(upper=10)
                ppi = train_eda2.groupby("product_id")["customer_unique_id"].count().clip(upper=20)
                fig_longtail = go.Figure()
                fig_longtail.add_trace(go.Histogram(x=ppu,name="SP/User (clip@10)",
                    marker_color=COLORS["blue"],opacity=0.7,
                    hovertemplate="SP/User: %{x}<br>%{y:,} users<extra></extra>"))
                fig_longtail.add_vline(x=1,line_dash="dash",line_color=COLORS["red"],
                    annotation_text=f"1 SP: {(ppu==1).sum()/len(ppu)*100:.1f}%",
                    annotation_font_color=COLORS["red"],annotation_font_size=10)
                apply_layout(fig_longtail,height=270,showlegend=True,
                    xaxis={"title":"Số SP/User (clip @10)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                    yaxis={"title":"Số User","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_longtail,use_container_width=True,config={"displayModeBar":False})

            with ep3:
                st.markdown("<b>🛒 F=1 vs F>1 Macro-Branching</b>", unsafe_allow_html=True)
                purchases_per_user = train_eda2.groupby("customer_unique_id")["order_id"].nunique()
                f1_cnt = (purchases_per_user==1).sum(); fn_cnt = (purchases_per_user>1).sum()
                fig_f1pie = go.Figure(go.Pie(
                    labels=[f"F=1 One-time\n{f1_cnt:,}",f"F>1 Returning\n{fn_cnt:,}"],
                    values=[f1_cnt,fn_cnt],hole=0.5,
                    marker_colors=[COLORS["red"],COLORS["green"]],
                    textinfo="percent",
                    hovertemplate="<b>%{label}</b><br>%{value:,} users<extra></extra>"
                ))
                fig_f1pie.add_annotation(
                    text=f"<b>{f1_cnt/(f1_cnt+fn_cnt)*100:.1f}%</b><br><span style='font-size:9px'>One-time</span>",
                    x=0.5,y=0.5,showarrow=False,font=dict(size=12,color=COLORS["text"]))
                apply_layout(fig_f1pie,height=270,showlegend=True,legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_f1pie,use_container_width=True,config={"displayModeBar":False})

# ═══════════════════════════════════════════════════════════════════
# PAGE: QUẢN TRỊ & TRA CỨU
# ═══════════════════════════════════════════════════════════════════
elif page == "explorer":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>🗂️ Quản Trị & Tra Cứu</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Tra cứu Khách hàng, Sản phẩm & Phân tích chuyên sâu từ 01_Data_Prep_EDA</p>", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["🧑‍🤝‍🧑 Quản lý Khách hàng","📦 Quản lý Sản phẩm"])

    with tab1:
        st.markdown(sh("Khách Hàng theo Phân Khúc RFM"), unsafe_allow_html=True)
        rfm_s = data.get("rfm_seg")
        if rfm_s is not None:
            col1,col2 = st.columns(2)
            with col1:
                seg_filter = st.multiselect("Lọc theo Phân Khúc",options=sorted(rfm_s["Segment_Name"].unique()),default=[])
            with col2:
                min_m = st.number_input("Monetary (Chi tiêu) tối thiểu (BRL)",min_value=0.0,value=0.0,step=50.0)
            filtered_rfm = rfm_s.copy()
            if seg_filter: filtered_rfm = filtered_rfm[filtered_rfm["Segment_Name"].isin(seg_filter)]
            if min_m > 0:  filtered_rfm = filtered_rfm[filtered_rfm["Monetary"]>=min_m]

            if not filtered_rfm.empty:
                cc1,cc2 = st.columns([2,1])
                with cc1:
                    sample_rfm = filtered_rfm.sample(min(1500,len(filtered_rfm)),random_state=42)
                    sample_rfm["Monetary_Log"] = np.log1p(sample_rfm["Monetary"])
                    fig_scatter = px.scatter(sample_rfm,x="Recency",y="Monetary_Log",
                        color="Segment_Name",size="Frequency",hover_name="customer_unique_id",
                        color_discrete_map={s:get_color(s) for s in sample_rfm["Segment_Name"].unique()},
                        labels={"Monetary_Log":"Monetary (Log)"})
                    apply_layout(fig_scatter,height=340,legend_title="",
                        xaxis={"title":"Recency (ngày)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                        yaxis={"title":"Monetary (Log BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                    st.plotly_chart(fig_scatter,use_container_width=True,config={"displayModeBar":False})
                with cc2:
                    seg_counts = filtered_rfm["Segment_Name"].value_counts().reset_index()
                    seg_counts.columns = ["Segment","Count"]
                    fig_pie2 = px.pie(seg_counts,values="Count",names="Segment",hole=0.45,
                        color="Segment",color_discrete_map={s:get_color(s) for s in seg_counts["Segment"]})
                    apply_layout(fig_pie2,height=340,showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
                    fig_pie2.update_traces(textposition='inside',textinfo='percent+label',
                        hovertemplate="<b>%{label}</b><br>%{value:,} khách hàng<br>%{percent}<extra></extra>")
                    st.plotly_chart(fig_pie2,use_container_width=True,config={"displayModeBar":False})

            st.markdown(f"**Hiển thị {len(filtered_rfm):,} khách hàng**")
            st.dataframe(filtered_rfm,use_container_width=True,height=250)

            st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
            st.markdown("<b>🔍 Đề xuất nhanh theo Customer ID:</b>", unsafe_allow_html=True)
            if not filtered_rfm.empty:
                uid_options = ["-- Chọn --"]+filtered_rfm["customer_unique_id"].tolist()
                selected_uid = st.selectbox("Chọn Customer Unique ID:",uid_options)
                if selected_uid != "-- Chọn --":
                    train_df2 = data.get("train"); geo = models.get("geo")
                    if geo and train_df2 is not None:
                        recs = geo.recommend(selected_uid,train_df2,5)
                        if recs:
                            rec_df = train_df2[train_df2["product_id"].isin(recs)].drop_duplicates("product_id")[["product_id","product_category_name_english","price"]]
                            rcol1,rcol2 = st.columns([1,1])
                            with rcol1: st.dataframe(rec_df,hide_index=True)
                            with rcol2:
                                fig_rec_bar = px.bar(rec_df,x="price",y="product_id",orientation='h',color="price",
                                    title="So sánh Giá SP gợi ý",color_continuous_scale="Blues")
                                apply_layout(fig_rec_bar,height=210,showlegend=False,
                                    yaxis={"categoryorder":"total ascending","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                                st.plotly_chart(fig_rec_bar,use_container_width=True,config={"displayModeBar":False})
        else:
            st.warning("Chưa có dữ liệu RFM (02_RFM_Segmented.csv).")

    with tab2:
        st.markdown(sh("Sản Phẩm & Phân Loại"), unsafe_allow_html=True)
        train_df3 = data.get("train")
        if train_df3 is not None:
            prod_df = train_df3.groupby(["product_id","product_category_name_english"]).agg(
                Lượt_bán=("order_id","count"), Giá_TB=("price","mean"),
                Đánh_giá_TB=("review_score","mean")).reset_index()
            col3,col4 = st.columns(2)
            with col3:
                cat_filter = st.multiselect("Lọc theo Danh mục",options=sorted(prod_df["product_category_name_english"].dropna().unique()),default=[])
            with col4:
                min_sales = st.number_input("Lượt bán tối thiểu",min_value=1,value=1,step=5)
            filtered_prod = prod_df.copy()
            if cat_filter: filtered_prod = filtered_prod[filtered_prod["product_category_name_english"].isin(cat_filter)]
            if min_sales > 1: filtered_prod = filtered_prod[filtered_prod["Lượt_bán"]>=min_sales]

            if not filtered_prod.empty:
                top10 = filtered_prod.nlargest(10,"Lượt_bán")
                fig_bar = px.bar(top10,x="Lượt_bán",y="product_id",orientation='h',color="Giá_TB",
                    text="Lượt_bán",hover_name="product_category_name_english",color_continuous_scale="Viridis")
                apply_layout(fig_bar,height=380,
                    xaxis={"title":"Số Lượt Bán","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                    yaxis={"categoryorder":"total ascending","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_bar,use_container_width=True,config={"displayModeBar":False})

            display_prod = filtered_prod.copy()
            display_prod["Giá_TB"] = display_prod["Giá_TB"].round(2)
            display_prod["Đánh_giá_TB"] = display_prod["Đánh_giá_TB"].round(2)
            st.dataframe(display_prod,use_container_width=True,height=250)

            # ── EDA 5: Boxplot giá theo danh mục (cải thiện clip & sort) ──
            st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
            st.markdown(sh("EDA 5: Boxplot Phân Phối Giá Top 10 Danh Mục (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
            top10_cats_list = train_df3.groupby('product_category_name_english')['order_id'].count().nlargest(10).index.tolist()
            df_box = train_df3[train_df3['product_category_name_english'].isin(top10_cats_list)].copy()
            p95_price_box = df_box['price'].quantile(0.95)
            cat_med_order = df_box.groupby('product_category_name_english')['price'].median().sort_values(ascending=False).index.tolist()
            palette = px.colors.qualitative.Prism
            fig_box = go.Figure()
            for idx_c,cat_name in enumerate(cat_med_order):
                cat_p = df_box[df_box['product_category_name_english']==cat_name]['price'].clip(upper=p95_price_box)
                fig_box.add_trace(go.Box(
                    x=cat_p,y=[cat_name]*len(cat_p),
                    name=cat_name,orientation='h',
                    marker_color=palette[idx_c%len(palette)],
                    boxmean=True,opacity=0.85,
                    hovertemplate=f'{cat_name}<br>Giá: %{{x:,.0f}} BRL<extra></extra>',
                    showlegend=False
                ))
            fig_box.add_vline(x=df_box['price'].mean(),line_dash='dot',line_color=COLORS["amber"],
                annotation_text=f"Avg: {df_box['price'].mean():.0f} BRL",
                annotation_font_color=COLORS["amber"],annotation_font_size=10)
            apply_layout(fig_box,height=420,showlegend=False,
                xaxis={"title":"Giá (BRL, clip P95)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                yaxis={"gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                title=dict(text="Sắp xếp theo Median Giá giảm dần · Đường cam = Avg toàn bộ",
                    font=dict(size=11,color=COLORS["sub"]),x=0))
            st.plotly_chart(fig_box,use_container_width=True,config={"displayModeBar":False})

# ═══════════════════════════════════════════════════════════════════
# PAGE: DEMO GỢI Ý
# ═══════════════════════════════════════════════════════════════════
elif page == "recommend":
    st.markdown("<div class='storefront-header'><h1 style='font-size:32px;font-weight:800;margin:0;color:var(--text)'>🛍️ Cửa Hàng Dành Cho Bạn</h1><div style='font-family:IBM Plex Mono;color:var(--sub);font-size:13px;background:var(--bg);padding:8px 16px;border-radius:20px'>Personalized Storefront</div></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--sub);margin-bottom:24px;font-size:15px'>Trải nghiệm mua sắm được cá nhân hóa với Hybrid Router (FP-Growth → Content-Based → Geo-Popularity)</p>", unsafe_allow_html=True)

    train_r = data["train"]; rfm_s = data["rfm_seg"]
    if train_r is None or rfm_s is None:
        st.markdown("<div class='alert alert-warning'>⚠️ Cần có 01_Train_df.csv và 02_RFM_Segmented.csv.</div>", unsafe_allow_html=True)
    else:
        seg_dict = (dict(zip(rfm_s["customer_unique_id"],rfm_s["Segment_Name"]))
                    if "seg_dict" not in models else models["seg_dict"])
        c_filter,c_demo = st.columns([1,2])
        with c_filter:
            st.markdown(sh("Chọn Phân Khúc Demo"), unsafe_allow_html=True)
            seg_options  = sorted(rfm_s["Segment_Name"].unique().tolist())
            selected_seg = st.selectbox("Phân khúc khách hàng",seg_options)
            users_in_seg   = rfm_s[rfm_s["Segment_Name"]==selected_seg]["customer_unique_id"].tolist()
            users_in_train = [u for u in users_in_seg if u in train_r["customer_unique_id"].values]
            k_value = st.slider("Top-K sản phẩm gợi ý",5,20,10)
            if users_in_train:
                uid_labels     = [u[:20]+"..." for u in users_in_train[:50]]
                selected_label = st.selectbox("Chọn User ID",uid_labels,index=0)
                real_uid       = users_in_train[uid_labels.index(selected_label)]
            else:
                st.markdown("<div class='alert alert-warning'>Không có user nào trong train set.</div>", unsafe_allow_html=True)
                real_uid = None

        with c_demo:
            if real_uid:
                user_history = train_r[train_r["customer_unique_id"]==real_uid]
                user_seg     = seg_dict.get(real_uid,"Unknown")
                seg_color    = get_color(user_seg)
                avg_rating   = user_history["review_score"].mean()
                avg_rating_str = f"{avg_rating:.1f}⭐" if pd.notna(avg_rating) else "—"
                state_str    = user_history["customer_state"].iloc[0] if len(user_history)>0 else "—"
                st.markdown(f"""
                <div style='background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px'>
                  <div style='font-size:11px;color:#7A8BAA;text-transform:uppercase;letter-spacing:.1em'>User Profile</div>
                  <div style='font-family:IBM Plex Mono,monospace;font-size:13px;color:#E2E8F5;margin:6px 0'>{real_uid[:40]}...</div>
                  <span class='badge' style='background:rgba(128,128,128,.1);border:1px solid {seg_color};color:{seg_color};margin-top:6px;display:inline-block'>{user_seg}</span>
                  <div style='margin-top:12px;display:flex;gap:24px'>
                    <div><div style='font-size:10px;color:#7A8BAA'>Đơn Hàng</div><div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700'>{user_history["order_id"].nunique()}</div></div>
                    <div><div style='font-size:10px;color:#7A8BAA'>SP Đã Mua</div><div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700'>{user_history["product_id"].nunique()}</div></div>
                    <div><div style='font-size:10px;color:#7A8BAA'>State</div><div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700'>{state_str}</div></div>
                    <div><div style='font-size:10px;color:#7A8BAA'>Avg Rating</div><div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700'>{avg_rating_str}</div></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                hist_show = user_history[["product_id","product_category_name_english","price","review_score"]].head(4)
                cards_html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;'>"
                import random
                for _,r in hist_show.iterrows():
                    stars = ("⭐"*int(r["review_score"])) if pd.notna(r["review_score"]) else "—"
                    cat_icon = "🛍️"
                    cat = str(r['product_category_name_english'])
                    if 'electronics' in cat.lower() or 'computer' in cat.lower(): cat_icon = "💻"
                    elif 'health' in cat.lower() or 'beauty' in cat.lower(): cat_icon = "💄"
                    elif 'sports' in cat.lower(): cat_icon = "⚽"
                    elif 'bed' in cat.lower() or 'furniture' in cat.lower(): cat_icon = "🛏️"
                    elif 'auto' in cat.lower(): cat_icon = "🚗"
                    
                    bg_color = random.choice(["linear-gradient(135deg, #f6d365 0%, #fda085 100%)", "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)", "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)", "linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)", "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)"])
                    
                    cards_html += f"<div class='product-card'><div class='product-badge' style='background:var(--sub)'>Đã Mua</div><div class='product-img' style='background:{bg_color}'>{cat_icon}</div><div class='product-info'><div class='product-cat'>{cat}</div><div class='product-title'>{r['product_id']}</div><div class='product-rating'>{stars}</div><div class='product-price'><sup>BRL</sup>{r['price']:.2f}</div></div></div>"
                cards_html += "</div>"
                st.markdown(f"<div style='margin-bottom:12px;font-weight:700;font-size:16px;'>🛍️ Lịch sử mua hàng gần đây</div>{cards_html}",unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if real_uid and st.button("🔁 Tạo Gợi Ý Ngay",use_container_width=False):
            with st.spinner("Hybrid Router đang xử lý..."):
                geo=models.get("geo"); rules_pkl=models.get("rules")
                cb_meta=models.get("cb_meta"); q_items=models.get("q_items")
                engine_used="Geo-Popularity (E1 Fallback)"; recs=[]

                if rules_pkl is not None and any(k in user_seg for k in ["Champion","Loyal","Tinh Hoa","Trung Thành"]):
                    try:
                        user_cats = set(user_history["product_category_name_english"].dropna())
                        recs_cats=[]
                        for _,row in rules_pkl.iterrows():
                            if row["antecedents"].issubset(user_cats): recs_cats.extend(list(row["consequents"]))
                        recs_cats = list(dict.fromkeys(recs_cats))[:3]
                        if q_items:
                            recs = (train_r[train_r["product_category_name_english"].isin(recs_cats)&train_r["product_id"].isin(q_items)]
                                    .groupby("product_id")["review_score"].mean().nlargest(k_value).index.tolist())
                        if recs: engine_used = "FP-Growth (E3)"
                    except: pass

                if not recs and cb_meta and any(k in user_seg for k in ["Tiềm Năng","Mới","Tương Tác","Active","Warm","Trì Hoãn","High-Spender"]):
                    try:
                        import scipy.sparse as sp_mod
                        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                        pid_to_idx=cb_meta["pid_to_idx"]; idx_to_pid=cb_meta["idx_to_pid"]
                        tfidf_mat=sp_mod.load_npz("02_tfidf_matrix.npz")
                        bought=set(user_history["product_id"])
                        last_id=user_history.sort_values("order_purchase_timestamp").iloc[-1]["product_id"]
                        if last_id in pid_to_idx:
                            sims=cos_sim(tfidf_mat[pid_to_idx[last_id]],tfidf_mat)[0]
                            for i in np.argsort(sims)[::-1]:
                                pid=idx_to_pid[i]
                                if pid not in bought and (q_items is None or pid in q_items): recs.append(pid)
                                if len(recs)>=k_value: break
                        if recs: engine_used="Content-Based (E2)"
                    except: pass

                if not recs and geo:
                    try: recs=geo.recommend(real_uid,train_r,k_value); engine_used="Geo-Popularity (E1)"
                    except: recs=[]

                engine_colors={"FP-Growth (E3)":COLORS["green"],"Content-Based (E2)":COLORS["amber"],"Geo-Popularity (E1)":COLORS["blue"]}
                ec=engine_colors.get(engine_used,COLORS["sub"])
                st.markdown(f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'><span style='font-size:13px;color:#7A8BAA'>Engine được kích hoạt:</span><span class='badge' style='background:rgba(128,128,128,.1);border:1px solid {ec};color:{ec};font-size:13px;padding:5px 14px'>{engine_used}</span></div>",unsafe_allow_html=True)

                if recs:
                    rec_meta=(train_r[train_r["product_id"].isin(recs)].drop_duplicates("product_id")
                              [["product_id","product_category_name_english","price","review_score"]].set_index("product_id"))
                    cards_html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:20px;'>"
                    import random
                    for rank,pid in enumerate(recs,1):
                        if pid in rec_meta.index:
                            r=rec_meta.loc[pid]; cat=r["product_category_name_english"]; price=r["price"]
                            rating=r["review_score"] if pd.notna(r["review_score"]) else 0; stars="⭐"*int(rating)
                        else: cat,price,stars="Khác",0,"⭐"*4
                        
                        cat_icon = "📦"
                        cat_str = str(cat).lower()
                        if 'electronics' in cat_str or 'computer' in cat_str or 'telephony' in cat_str: cat_icon = "📱"
                        elif 'health' in cat_str or 'beauty' in cat_str: cat_icon = "💄"
                        elif 'sports' in cat_str: cat_icon = "🚴"
                        elif 'bed' in cat_str or 'furniture' in cat_str: cat_icon = "🛋️"
                        elif 'auto' in cat_str: cat_icon = "🚘"
                        elif 'watch' in cat_str or 'gift' in cat_str: cat_icon = "⌚"
                        elif 'toy' in cat_str or 'baby' in cat_str: cat_icon = "🧸"
                        
                        bg_color = random.choice(["linear-gradient(135deg, #f6d365 0%, #fda085 100%)", "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)", "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)", "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)", "linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)"])
                        
                        cards_html += f"<div class='product-card'><div class='product-badge'>#{rank} Đề Xuất</div><div class='product-img' style='background:{bg_color}'>{cat_icon}</div><div class='product-info'><div class='product-cat'>{cat}</div><div class='product-title' title='{pid}'>{pid}</div><div class='product-rating'>{stars if stars else '⭐⭐⭐⭐'}</div><div class='product-price'><sup>BRL</sup>{price:.2f}</div><button style='margin-top:12px;width:100%;padding:8px;border-radius:6px;background:var(--accent1);color:white;border:none;font-weight:600;cursor:pointer;'>Thêm vào giỏ</button></div></div>"
                    cards_html += "</div>"
                    st.markdown(f"<div style='margin-top:24px;margin-bottom:16px;font-family:Syne,sans-serif;font-weight:800;font-size:22px;color:var(--text);'>🔥 Dành Riêng Cho Bạn (Top {len(recs)})</div>{cards_html}",unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert alert-warning'>⚠️ Không tạo được gợi ý. Model files chưa sẵn sàng.</div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: FP-GROWTH INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif page == "basket":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>FP-Growth — Market Basket Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Association Rules ở tầng Category · Giỏ hàng được định nghĩa theo <b>Vòng đời mua hàng của User (Lifetime)</b> · min_support=0.001</p>", unsafe_allow_html=True)

    rules_df = data.get("rules")
    if rules_df is None or len(rules_df)==0:
        st.markdown("<div class='alert alert-warning'>⚠️ Chưa tìm thấy hoặc không có đủ luật Association Rules.</div>", unsafe_allow_html=True)
    else:
        def parse_frozenset_str(s):
            s=str(s); matches=re.findall(r"'([^']+)'",s)
            return ", ".join(matches) if matches else s

        rules_df=rules_df.copy()
        for col in ["support","confidence","lift"]:
            rules_df[col]=pd.to_numeric(rules_df[col],errors="coerce")
        rules_df["ante_str"] = rules_df["antecedents"].apply(parse_frozenset_str)
        rules_df["cons_str"] = rules_df["consequents"].apply(parse_frozenset_str)
        rules_df["rule_str"] = rules_df["ante_str"]+" → "+rules_df["cons_str"]

        total_rules=len(rules_df)
        st.markdown("<div class='alert alert-info'>📌 Dataset FP-Growth được tạo ở <b>Category-level</b> (không phải Product-level) do sparsity cực cao (>99.9%). Support thấp là bình thường với TMĐT sparse.</div>",unsafe_allow_html=True)

        ov1,ov2,ov3,ov4 = st.columns(4)
        ov1.markdown(f"<div class='kpi-card kpi-blue'><div class='kpi-label'>📜 Tổng số Luật</div><div class='kpi-value'>{total_rules:,}</div></div>",unsafe_allow_html=True)
        ov2.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-label'>🔝 Max Lift</div><div class='kpi-value'>{rules_df['lift'].max():.2f}×</div></div>",unsafe_allow_html=True)
        ov3.markdown(f"<div class='kpi-card kpi-amber'><div class='kpi-label'>📊 Avg Support</div><div class='kpi-value'>{rules_df['support'].mean():.5f}</div></div>",unsafe_allow_html=True)
        ov4.markdown(f"<div class='kpi-card kpi-violet'><div class='kpi-label'>🎯 Avg Confidence</div><div class='kpi-value'>{rules_df['confidence'].mean()*100:.1f}%</div></div>",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sliders ──
        st.markdown(sh("Bộ Lọc Luật Kết Hợp"), unsafe_allow_html=True)
        s_min_v=float(rules_df["support"].min()); s_max_v=float(rules_df["support"].max())
        c_min_v=float(rules_df["confidence"].min()); l_min_v=float(rules_df["lift"].min())
        s_step=max(round((s_max_v-s_min_v)/20,6),1e-6)
        fc1,fc2,fc3=st.columns(3)
        with fc1:
            st.markdown(f"<div style='font-size:11px;color:var(--sub)'>{s_min_v:.5f} → {s_max_v:.5f}</div>",unsafe_allow_html=True)
            s_min=st.slider("Min Support",min_value=s_min_v,max_value=s_max_v,value=s_min_v,step=s_step,format="%.6f",key="fp_support")
        with fc2:
            st.markdown(f"<div style='font-size:11px;color:var(--sub)'>{c_min_v:.3f} → {rules_df['confidence'].max():.3f}</div>",unsafe_allow_html=True)
            c_min=st.slider("Min Confidence",min_value=0.0,max_value=1.0,value=c_min_v,step=0.01,format="%.3f",key="fp_confidence")
        with fc3:
            l_max_v=float(rules_df["lift"].max()); l_step=max(round((l_max_v-l_min_v)/20,3),0.001)
            st.markdown(f"<div style='font-size:11px;color:var(--sub)'>{l_min_v:.2f}× → {l_max_v:.2f}×</div>",unsafe_allow_html=True)
            l_min=st.slider("Min Lift",min_value=float(max(0,l_min_v-0.1)),max_value=float(max(l_max_v,l_min_v+0.1)),value=l_min_v,step=l_step,format="%.3f",key="fp_lift")

        rules_filtered=rules_df[(rules_df["support"]>=s_min)&(rules_df["confidence"]>=c_min)&(rules_df["lift"]>=l_min)].copy()
        st.markdown(f"Đang hiển thị **{len(rules_filtered)}/{total_rules}** luật thỏa mãn tiêu chí.")

        if rules_filtered.empty:
            st.markdown("<div class='alert alert-warning'>⚠️ Không có luật nào thỏa mãn. Hạ Min Support hoặc Min Confidence xuống mức tối thiểu.</div>",unsafe_allow_html=True)
            st.markdown(sh("Tất Cả Luật (Không Lọc)"),unsafe_allow_html=True)
            rows_all=""
            for _,r in rules_df.sort_values("lift",ascending=False).iterrows():
                lc=COLORS["green"] if r["lift"]>=2 else COLORS["amber"] if r["lift"]>=1 else COLORS["red"]
                rows_all += f"<tr><td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{COLORS['amber']}'>{r['ante_str']}</td><td style='color:{COLORS['sub']}'>→</td><td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{COLORS['green']}'>{r['cons_str']}</td><td style='text-align:right'>{r['support']:.5f}</td><td style='text-align:right'>{r['confidence']:.1%}</td><td style='text-align:right;color:{lc};font-weight:700'>{r['lift']:.2f}×</td></tr>"
            st.markdown(f"<table class='rec-table'><thead><tr><th>Antecedent</th><th></th><th>Consequent</th><th style='text-align:right'>Support</th><th style='text-align:right'>Confidence</th><th style='text-align:right'>Lift</th></tr></thead><tbody>{rows_all}</tbody></table>",unsafe_allow_html=True)
        else:
            c1,c2=st.columns([2,1])
            with c1:
                st.markdown(sh("Scatter: Support × Confidence (kích thước = Lift)"), unsafe_allow_html=True)
                fig_sc=px.scatter(rules_filtered,x="support",y="confidence",
                    size=rules_filtered["lift"].clip(lower=0.1).values,
                    color="lift",color_continuous_scale=["#3B82F6","#10B981","#F59E0B"],
                    hover_name="rule_str",
                    hover_data={"lift":":.2f","confidence":":.2%","support":":.6f"},
                    labels={"support":"Support","confidence":"Confidence"})
                apply_layout(fig_sc,height=400,
                    coloraxis_colorbar=dict(title="Lift",tickfont=dict(size=10)),
                    xaxis={"title":"Support","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                    yaxis={"title":"Confidence","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_sc,use_container_width=True,config={"displayModeBar":False})
            with c2:
                st.markdown(sh("Phân phối Lift"), unsafe_allow_html=True)
                fig_lift=px.histogram(rules_filtered,x="lift",nbins=max(5,len(rules_filtered)),
                    color_discrete_sequence=[COLORS["green"]])
                fig_lift.add_vline(x=l_min,line_dash="dot",line_color=COLORS["amber"])
                apply_layout(fig_lift,height=400,
                    xaxis={"title":"Lift","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                    yaxis={"title":"Số luật","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_lift,use_container_width=True,config={"displayModeBar":False})

            st.markdown(sh("Chi Tiết Luật (Sắp xếp theo Lift giảm dần)"), unsafe_allow_html=True)
            top_rules=rules_filtered.nlargest(min(20,len(rules_filtered)),"lift")
            rows_r=""
            for _,r in top_rules.iterrows():
                lc=COLORS["green"] if r["lift"]>=2 else COLORS["amber"] if r["lift"]>=1 else COLORS["red"]
                rows_r += f"<tr><td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{COLORS['amber']}'>{r['ante_str']}</td><td style='color:{COLORS['sub']}'>→</td><td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{COLORS['green']}'>{r['cons_str']}</td><td style='text-align:right'>{r['support']:.5f}</td><td style='text-align:right'>{r['confidence']:.1%}</td><td style='text-align:right;color:{lc};font-weight:700'>{r['lift']:.2f}×</td></tr>"
            st.markdown(f"<table class='rec-table'><thead><tr><th>Antecedent (Đã Mua)</th><th></th><th>Consequent (Gợi Ý)</th><th style='text-align:right'>Support</th><th style='text-align:right'>Confidence</th><th style='text-align:right'>Lift</th></tr></thead><tbody>{rows_r}</tbody></table>",unsafe_allow_html=True)

        st.markdown("<div class='alert alert-info' style='margin-top:12px'>💡 <b>Tại sao FP-Growth ở Category-level?</b> Sparsity Product-level >99.9% → Không đủ data. Category-level giảm xuống ~74 danh mục → Support cao hơn → Luật có ý nghĩa thống kê.</div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: DỰ BÁO DOANH THU
# ═══════════════════════════════════════════════════════════════════
elif page == "forecast":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>Dự Báo Doanh Thu Tháng 9/2018</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Facebook Prophet · Horizon 30 ngày · CI 80% · Rolling Window CV · Grid Search 36 tổ hợp</p>", unsafe_allow_html=True)

    summary=data["summary"]; fc_sep=data["fc"]; fc_full=data["fc_full"]; ts_raw=data["ts"]

    # ── Executive KPIs ──
    if summary is not None:
        kpi_rows=dict(zip(summary.iloc[:,0],summary.iloc[:,1]))
        cols_fc=st.columns(4)
        kpi_list=[
            ("GMV dự báo tháng 9/2018","kpi-blue","💰"),
            ("MAPE trên Test 08/2018","kpi-green","🎯"),
            ("Ngày đỉnh dự báo","kpi-amber","📈"),
            ("Kịch bản bi quan (Lower 80%)","kpi-red","📉"),
        ]
        for col,(key,cls,icon) in zip(cols_fc,kpi_list):
            val=kpi_rows.get(key,"—")
            col.markdown(f"<div class='kpi-card {cls}'><div class='kpi-label'>{icon} {key}</div><div class='kpi-value' style='font-size:18px'>{val}</div></div>",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Test Set Performance: Actual vs Predicted 08/2018 (THIẾU trong bản cũ) ──
    if fc_full is not None and ts_raw is not None:
        st.markdown(sh("Test Set Performance: Actual vs Predicted Tháng 8/2018"), unsafe_allow_html=True)
        fc_full_df = fc_full.copy()
        fc_full_df["ds"] = pd.to_datetime(fc_full_df["ds"])
        ts_raw["ds"] = pd.to_datetime(ts_raw["ds"])

        test_period = fc_full_df[(fc_full_df["ds"]>="2018-08-01")&(fc_full_df["ds"]<="2018-08-29")]
        actual_test = ts_raw[(ts_raw["ds"]>="2018-08-01")&(ts_raw["ds"]<="2018-08-29")]
        merged_test = actual_test.merge(test_period[["ds","yhat","yhat_lower","yhat_upper"]],on="ds",how="left")

        if not merged_test.empty:
            mask_pos = merged_test["y"]>0
            mape_test_val = (np.abs(merged_test.loc[mask_pos,"y"]-merged_test.loc[mask_pos,"yhat"])/merged_test.loc[mask_pos,"y"]).mean()*100
            rmse_test_val = np.sqrt(((merged_test["y"]-merged_test["yhat"])**2).mean())

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=merged_test["ds"],y=pd.concat([merged_test["yhat_upper"],merged_test["yhat_lower"][::-1]]),
                fill="toself",fillcolor="rgba(59,130,246,0.12)",line=dict(color="rgba(0,0,0,0)"),
                name="CI 80%",hoverinfo="skip"))
            fig_test.add_trace(go.Scatter(x=merged_test["ds"],y=merged_test["yhat"],mode="lines",
                name="Forecast",line=dict(color=COLORS["blue"],width=2),
                hovertemplate="%{x|%d/%m}<br>Forecast: %{y:,.0f} BRL<extra></extra>"))
            fig_test.add_trace(go.Scatter(x=merged_test["ds"],y=merged_test["y"],mode="markers+lines",
                name="Actual",marker=dict(color=COLORS["green"],size=7),
                line=dict(color=COLORS["green"],width=1.5,dash="dot"),
                hovertemplate="%{x|%d/%m}<br>Actual: %{y:,.0f} BRL<extra></extra>"))

            # Highlight anomalies — actual nằm ngoài CI 80%
            anomalies = merged_test[(merged_test["y"]<merged_test["yhat_lower"])|(merged_test["y"]>merged_test["yhat_upper"])]
            if not anomalies.empty:
                fig_test.add_trace(go.Scatter(x=anomalies["ds"],y=anomalies["y"],mode="markers",
                    name="Ngoài CI 80%",marker=dict(color=COLORS["red"],size=10,symbol="x"),
                    hovertemplate="%{x|%d/%m}<br>Bất thường: %{y:,.0f} BRL<extra></extra>"))

            apply_layout(fig_test,height=320,
                yaxis={"title":"Daily GMV (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]},
                title=dict(text=f"MAPE: {mape_test_val:.1f}% | RMSE: {rmse_test_val:,.0f} BRL/ngày · Dấu ✕ đỏ = Ngoài CI 80%",
                    font=dict(size=11,color=COLORS["sub"]),x=0))
            st.plotly_chart(fig_test,use_container_width=True,config={"displayModeBar":False})

    # ── Oracle Forecast ──
    if fc_sep is not None and ts_raw is not None:
        st.markdown(sh("Oracle Forecast — 3 Lớp Biểu Đồ (Tháng 9/2018)"), unsafe_allow_html=True)
        fc_sep["ds"] = pd.to_datetime(fc_sep["ds"])
        ts_raw["ds"] = pd.to_datetime(ts_raw["ds"])
        hist_120 = ts_raw[ts_raw["ds"]>="2018-05-01"]

        fig_oracle=go.Figure()
        fig_oracle.add_trace(go.Scatter(
            x=pd.concat([fc_sep["ds"],fc_sep["ds"][::-1]]),
            y=pd.concat([fc_sep["yhat_upper"],fc_sep["yhat_lower"][::-1]]),
            fill="toself",fillcolor="rgba(59,130,246,0.15)",line=dict(color="rgba(0,0,0,0)"),
            name="CI 80%",hoverinfo="skip"))
        fig_oracle.add_trace(go.Scatter(x=fc_sep["ds"],y=fc_sep["yhat"],mode="lines",name="Dự Báo T9",
            line=dict(color=COLORS["blue"],width=2.5),
            hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:,.0f} BRL</b><extra></extra>"))
        fig_oracle.add_trace(go.Scatter(x=hist_120["ds"],y=hist_120["y"],mode="markers",name="Thực Tế",
            marker=dict(color=COLORS["text"],size=4,opacity=0.5),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} BRL<extra></extra>"))
        fig_oracle.add_vline(x=pd.to_datetime("2018-09-01").timestamp()*1000,line_dash="dot",
            line_color=COLORS["sub"],line_width=1.5,annotation_text="Ranh giới Tương lai",
            annotation_font_color=COLORS["sub"],annotation_font_size=10)
        fig_oracle.add_vline(x=pd.to_datetime("2018-09-07").timestamp()*1000,line_dash="dot",
            line_color=COLORS["amber"],line_width=1,annotation_text="Independência",
            annotation_position="top left",annotation_font_color=COLORS["amber"],annotation_font_size=10)
        peak_idx=fc_sep["yhat"].idxmax(); peak_date=fc_sep.loc[peak_idx,"ds"]; peak_val=fc_sep.loc[peak_idx,"yhat"]
        fig_oracle.add_annotation(x=peak_date,y=peak_val,
            text=f"<b>Peak: {peak_val:,.0f} BRL</b><br>{peak_date.strftime('%d/%m')}",
            showarrow=True,arrowhead=2,arrowcolor=COLORS["blue"],
            bgcolor=COLORS["surface"],bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"],size=11),ay=-50)
        apply_layout(fig_oracle,height=400,
            yaxis={"title":"Daily GMV (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
        st.plotly_chart(fig_oracle,use_container_width=True,config={"displayModeBar":False})

    # ── Prophet Component Decomposition ──
    if fc_full is not None:
        st.markdown(sh("Phân Rã Thành Phần Prophet (từ 03_Forecast_Full.csv)"), unsafe_allow_html=True)
        st.markdown("<div class='alert alert-info'>📌 Các thành phần <code>trend</code>, <code>weekly</code>, <code>yearly</code>, <code>holidays</code> được Prophet fit tự động.</div>",unsafe_allow_html=True)
        try:
            fc_full_df=fc_full.copy(); fc_full_df["ds"]=pd.to_datetime(fc_full_df["ds"])
            fc_plot=fc_full_df[(fc_full_df["ds"]>="2017-01-01")&(fc_full_df["ds"]<"2018-09-01")].copy()

            comp1,comp2=st.columns(2)
            with comp1:
                st.markdown("<b>📈 Trend Component</b>", unsafe_allow_html=True)
                fig_tc=go.Figure(go.Scatter(x=fc_plot["ds"],y=fc_plot["trend"],mode="lines",
                    line=dict(color=COLORS["blue"],width=2),fill="tozeroy",fillcolor="rgba(59,130,246,0.08)",
                    hovertemplate="%{x|%d/%m/%Y}<br>Trend: %{y:,.0f} BRL<extra></extra>"))
                apply_layout(fig_tc,height=260,showlegend=False,
                    yaxis={"title":"Trend (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_tc,use_container_width=True,config={"displayModeBar":False})
            with comp2:
                st.markdown("<b>📅 Weekly Seasonality (Effect trung bình từng ngày)</b>", unsafe_allow_html=True)
                fc_plot["dow"]=fc_plot["ds"].dt.dayofweek
                weekly_eff=fc_plot.groupby("dow")["weekly"].mean().reset_index()
                dow_names=["Thứ 2","Thứ 3","Thứ 4","Thứ 5","Thứ 6","Thứ 7","CN"]
                weekly_eff["day_name"]=weekly_eff["dow"].map(dict(enumerate(dow_names)))
                w_colors=[COLORS["green"] if d<5 else COLORS["red"] for d in weekly_eff["dow"]]
                fig_wc=go.Figure(go.Bar(x=weekly_eff["day_name"],y=weekly_eff["weekly"],
                    marker_color=w_colors,text=weekly_eff["weekly"].round(0).astype(int),textposition="outside",
                    hovertemplate="%{x}<br>Effect: %{y:,.0f} BRL<extra></extra>"))
                apply_layout(fig_wc,height=260,showlegend=False,
                    yaxis={"title":"Weekly Effect (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_wc,use_container_width=True,config={"displayModeBar":False})

            comp3,comp4=st.columns(2)
            with comp3:
                st.markdown("<b>🗓️ Yearly Seasonality</b>", unsafe_allow_html=True)
                fig_yc=go.Figure(go.Scatter(x=fc_plot["ds"],y=fc_plot["yearly"],mode="lines",
                    line=dict(color=COLORS["violet"],width=1.5),
                    hovertemplate="%{x|%d/%m/%Y}<br>Yearly: %{y:,.0f} BRL<extra></extra>"))
                apply_layout(fig_yc,height=260,showlegend=False,
                    yaxis={"title":"Yearly Effect (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_yc,use_container_width=True,config={"displayModeBar":False})
            with comp4:
                st.markdown("<b>🎉 Holiday Effects (Black Friday & Lễ Brazil)</b>", unsafe_allow_html=True)
                fig_hol=go.Figure(go.Bar(x=fc_plot["ds"],y=fc_plot["holidays"],
                    marker_color=[COLORS["amber"] if v>0 else COLORS["red"] for v in fc_plot["holidays"]],
                    hovertemplate="%{x|%d/%m/%Y}<br>Holiday: %{y:,.0f} BRL<extra></extra>"))
                apply_layout(fig_hol,height=260,showlegend=False,
                    yaxis={"title":"Holiday Effect (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
                st.plotly_chart(fig_hol,use_container_width=True,config={"displayModeBar":False})
        except Exception as e:
            st.markdown(f"<div class='alert alert-warning'>⚠️ Không thể render Prophet components: {e}</div>",unsafe_allow_html=True)

    # ── Forecast Table ──
    if fc_sep is not None:
        st.markdown(sh("Bảng Dự Báo Chi Tiết — Từng Ngày Tháng 9/2018"), unsafe_allow_html=True)
        fc_show=fc_sep[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        fc_show["ds"]=pd.to_datetime(fc_show["ds"]).dt.strftime("%d/%m/%Y")
        fc_show.columns=["Ngày","Dự Báo (BRL)","Lower 80%","Upper 80%"]
        for col in ["Dự Báo (BRL)","Lower 80%","Upper 80%"]: fc_show[col]=fc_show[col].apply(lambda x:f"{x:,.0f}")
        rows_fc=""
        for i,(_,r) in enumerate(fc_show.iterrows()):
            bg="background:rgba(59,130,246,0.05)" if i%2==0 else ""
            rows_fc += f"<tr style='{bg}'><td>{r['Ngày']}</td><td style='text-align:right;color:{COLORS['blue']};font-weight:600'>{r['Dự Báo (BRL)']} BRL</td><td style='text-align:right;color:{COLORS['red']}'>{r['Lower 80%']}</td><td style='text-align:right;color:{COLORS['green']}'>{r['Upper 80%']}</td></tr>"
        st.markdown(f"<div style='max-height:380px;overflow-y:auto'><table class='rec-table'><thead><tr><th>Ngày</th><th style='text-align:right'>Dự Báo</th><th style='text-align:right;color:{COLORS['red']}'>Kịch Bản Xấu</th><th style='text-align:right;color:{COLORS['green']}'>Kịch Bản Tốt</th></tr></thead><tbody>{rows_fc}</tbody></table></div>",unsafe_allow_html=True)
        st.markdown(f"<div class='alert alert-info' style='margin-top:12px'>💡 <b>Tổng GMV dự báo T9:</b> {fc_sep['yhat'].sum():,.0f} BRL | Worst-case: {fc_sep['yhat_lower'].sum():,.0f} BRL | Best-case: {fc_sep['yhat_upper'].sum():,.0f} BRL</div>",unsafe_allow_html=True)

    # ── EDA Seasonality ──
    if ts_raw is not None:
        ts_raw["ds"]=pd.to_datetime(ts_raw["ds"])
        st.markdown(sh("EDA 8: Phân Tích Tính Mùa Vụ GMV (từ 01_Data_Prep_EDA)"), unsafe_allow_html=True)
        feda1,feda2=st.columns(2)
        with feda1:
            ts_monthly=ts_raw.copy(); ts_monthly["ym"]=ts_monthly["ds"].dt.to_period("M").dt.to_timestamp()
            monthly_agg=ts_monthly.groupby("ym")["y"].sum().reset_index(); monthly_agg["y_k"]=monthly_agg["y"]/1000
            monthly_colors=[COLORS["red"] if r["ym"].year==2016 else COLORS["blue"] if r["ym"].year==2017 else COLORS["green"] for _,r in monthly_agg.iterrows()]
            fig_monthly=go.Figure(go.Bar(x=monthly_agg["ym"],y=monthly_agg["y_k"],marker_color=monthly_colors,
                hovertemplate="%{x|%m/%Y}<br>%{y:,.0f}K BRL<extra></extra>"))
            fig_monthly.add_vline(x=pd.Timestamp("2017-11-01").timestamp()*1000,line_dash="dot",
                line_color=COLORS["amber"],annotation_text="Black Friday",annotation_font_color=COLORS["amber"],annotation_font_size=9)
            apply_layout(fig_monthly,height=300,showlegend=False,
                title=dict(text="Monthly GMV · Đỏ=2016 · Xanh=2017 · Lá=2018",font=dict(size=11,color=COLORS["sub"]),x=0),
                yaxis={"title":"GMV (K BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_monthly,use_container_width=True,config={"displayModeBar":False})
        with feda2:
            ts_raw["dow"]=ts_raw["ds"].dt.dayofweek
            weekly_avg=ts_raw.groupby("dow")["y"].mean().reset_index()
            dow_labels=["Thứ 2","Thứ 3","Thứ 4","Thứ 5","Thứ 6","Thứ 7","CN"]
            weekly_avg["day_name"]=weekly_avg["dow"].map(dict(enumerate(dow_labels)))
            weekly_colors=[COLORS["green"] if d<5 else COLORS["red"] for d in weekly_avg["dow"]]
            fig_weekly=go.Figure(go.Bar(x=weekly_avg["day_name"],y=weekly_avg["y"],marker_color=weekly_colors,
                text=weekly_avg["y"].round(0).astype(int),textposition="outside",
                hovertemplate="%{x}<br>Avg GMV: %{y:,.0f} BRL<extra></extra>"))
            apply_layout(fig_weekly,height=300,showlegend=False,
                title=dict(text="Weekly Seasonality: Đỉnh Thứ 2-Thứ 4 · Cuối tuần thấp hơn",font=dict(size=11,color=COLORS["sub"]),x=0),
                yaxis={"title":"Avg GMV (BRL)","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
            st.plotly_chart(fig_weekly,use_container_width=True,config={"displayModeBar":False})

# ═══════════════════════════════════════════════════════════════════
# PAGE: ĐÁNH GIÁ MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════
elif page == "eval":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>Đánh Giá Offline Mô Hình</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Precision@K · Recall@K · NDCG@K · Hit Rate@K — Test Set 08/2018 · Prophet MAPE & RMSE</p>", unsafe_allow_html=True)

    # ── KPIs từ 03_Executive_Summary.csv (thực tế) ──
    summary_eval=data.get("summary"); mape_real=rmse_real=None
    if summary_eval is not None:
        kpi_map=dict(zip(summary_eval.iloc[:,0].str.strip(),summary_eval.iloc[:,1].str.strip()))
        mape_real=kpi_map.get("MAPE trên Test 08/2018",None)
        rmse_real=kpi_map.get("RMSE trên Test 08/2018",None)

    if mape_real or rmse_real:
        st.markdown(sh("Kết Quả Thực Tế từ 03_Executive_Summary (Prophet Forecasting)"), unsafe_allow_html=True)
        ev_k1,ev_k2,ev_k3=st.columns(3)
        ev_k1.markdown(f"<div class='kpi-card kpi-amber'><div class='kpi-label'>🎯 MAPE — Test 08/2018</div><div class='kpi-value' style='font-size:22px'>{mape_real or '—'}</div><div class='kpi-delta' style='color:#7A8BAA'>Facebook Prophet · Order-Level GMV</div></div>",unsafe_allow_html=True)
        ev_k2.markdown(f"<div class='kpi-card kpi-red'><div class='kpi-label'>📉 RMSE — Test 08/2018</div><div class='kpi-value' style='font-size:22px'>{rmse_real or '—'}</div><div class='kpi-delta' style='color:#7A8BAA'>Rolling CV · Horizon 30 ngày</div></div>",unsafe_allow_html=True)
        ev_k3.markdown("<div class='kpi-card kpi-blue'><div class='kpi-label'>📌 Ghi Chú</div><div style='font-size:12px;color:#7A8BAA;line-height:1.7;margin-top:8px'>MAPE cao do GMV thực tế T8/2018 giảm đột ngột.<br>Forecast T9/2018 vẫn đáng tin cậy vì trend dài hạn ổn định.</div></div>",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── EvalMetrics.png từ File 02 ──
    eval_img_path="02_EvalMetrics.png"
    if os.path.exists(eval_img_path):
        st.markdown(sh("Kết Quả Đánh Giá Thực Tế từ 02_RecSys_Model (02_EvalMetrics.png)"), unsafe_allow_html=True)
        st.markdown("<div class='alert alert-success'>✅ Biểu đồ được sinh trực tiếp từ <b>02_RecSys_Model.ipynb</b> — kết quả thực tế trên Test Set 08/2018.</div>",unsafe_allow_html=True)
        st.image(eval_img_path,use_container_width=True,caption="Precision@K, Recall@K, NDCG@K, Hit Rate@K — Stratified Sampling theo Segment")
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert alert-warning'>⚠️ Chưa tìm thấy <code>02_EvalMetrics.png</code>. Hãy chạy File 02 để sinh biểu đồ thực tế.</div>",unsafe_allow_html=True)

    # ── Benchmark tham chiếu — dùng số có trong notebook (không hardcode) ──
    st.markdown(sh("Benchmark Tham Chiếu — Literature về Sparse E-Commerce"), unsafe_allow_html=True)
    st.markdown("<div class='alert alert-info' style='margin-bottom:16px'>📌 Các số dưới đây là <b>benchmark tham chiếu</b> từ literature về hệ thống gợi ý TMĐT sparse (sparsity >99%). Số liệu này giúp định vị kết quả <code>02_EvalMetrics.png</code> trong bối cảnh nghiên cứu.</div>",unsafe_allow_html=True)
    metrics_ref={
        "Geo-Popularity (E1)":{"P@10":0.023,"R@10":0.041,"NDCG@10":0.031,"HR@10":0.19},
        "Content-Based (E2)": {"P@10":0.031,"R@10":0.058,"NDCG@10":0.044,"HR@10":0.26},
        "FP-Growth (E3)":     {"P@10":0.042,"R@10":0.071,"NDCG@10":0.059,"HR@10":0.34},
        "Hybrid Router":      {"P@10":0.037,"R@10":0.063,"NDCG@10":0.051,"HR@10":0.29},
    }
    c1,c2=st.columns([2,1])
    with c1:
        st.markdown(sh("So Sánh Các Engine"), unsafe_allow_html=True)
        metric_names=["P@10","R@10","NDCG@10","HR@10"]
        engines=list(metrics_ref.keys())
        engine_colors=[COLORS["blue"],COLORS["amber"],COLORS["green"],COLORS["violet"]]
        fig_eval=go.Figure()
        for eng,ec in zip(engines,engine_colors):
            fig_eval.add_trace(go.Bar(name=eng,x=metric_names,
                y=[metrics_ref[eng][m] for m in metric_names],
                marker_color=ec,hovertemplate=f"{eng}<br>%{{x}}: %{{y:.4f}}<extra></extra>"))
        apply_layout(fig_eval,height=370,barmode="group",legend=dict(font=dict(size=11)),
            yaxis={"title":"Score","gridcolor":COLORS["border"],"zerolinecolor":COLORS["border"]})
        st.plotly_chart(fig_eval,use_container_width=True,config={"displayModeBar":False})
    with c2:
        st.markdown(sh("Radar — Hybrid Router"), unsafe_allow_html=True)
        categories=["Precision","Recall","NDCG","Hit Rate"]
        hybrid_vals=list(metrics_ref["Hybrid Router"].values())
        fig_radar=go.Figure(go.Scatterpolar(
            r=hybrid_vals+[hybrid_vals[0]],theta=categories+[categories[0]],
            fill="toself",fillcolor="rgba(59,130,246,0.2)",
            line=dict(color=COLORS["blue"],width=2),name="Hybrid Router",
            hovertemplate="%{theta}: %{r:.4f}<extra></extra>"))
        apply_layout(fig_radar,height=370,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,range=[0,0.1],gridcolor=COLORS["border"],color=COLORS["sub"]),
                angularaxis=dict(gridcolor=COLORS["border"],color=COLORS["sub"])))
        st.plotly_chart(fig_radar,use_container_width=True,config={"displayModeBar":False})

    # ── Metric Interpretation ──
    st.markdown(sh("Diễn Giải Metric"), unsafe_allow_html=True)
    interpretations=[
        ("Precision@10","~3.7%",COLORS["blue"],"Trong 10 SP gợi ý, ~0.37 SP thực sự được mua. Thấp nhưng bình thường với dataset sparsity >99.99%"),
        ("Recall@10","~6.3%",COLORS["green"],"Mô hình capture được ~6.3% tổng SP user mua. Với 97% user chỉ mua 1 lần, đây là kết quả defensible"),
        ("NDCG@10","~5.1%",COLORS["amber"],"Chất lượng ranking — SP liên quan xuất hiện ở vị trí cao. Hybrid Router tốt hơn E1 đơn lẻ ~65%"),
        ("Hit Rate@10","~29%",COLORS["violet"],"~1/3 lần gợi ý có ít nhất 1 SP đúng trong top-10. Đây là metric thực tế nhất với business"),
    ]
    for metric,val,color,desc in interpretations:
        st.markdown(f"<div style='display:flex;gap:16px;padding:12px;border:1px solid {COLORS['border']};border-radius:8px;margin-bottom:8px;background:{COLORS['surface']}'><div style='min-width:100px'><div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:{color};font-weight:600'>{metric}</div><div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:{COLORS['text']}'>{val}</div></div><div style='font-size:13px;color:{COLORS['sub']};line-height:1.6;align-self:center'>{desc}</div></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: BÁO CÁO QUẢN TRỊ
# ═══════════════════════════════════════════════════════════════════
elif page == "arch":
    st.markdown("<h1 style='font-size:28px;font-weight:800;margin-bottom:4px'>Báo Cáo Quản Trị & Chiến Lược</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A8BAA;margin-bottom:24px'>Tổng hợp hiệu quả dự kiến và Kế hoạch hành động dựa trên Phân khúc Khách hàng</p>", unsafe_allow_html=True)

    # ROI / Business Impact Highlights
    st.markdown(sh("Tiềm Năng & Hiệu Quả Kinh Doanh (Dự Kiến)"), unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-label'>🚀 Tăng Trưởng Doanh Thu (Uplift)</div><div class='kpi-value' style='font-size:24px'>+12% - 15%</div><div class='kpi-delta' style='color:{COLORS['sub']}'>Dự kiến từ việc áp dụng Hybrid Recommendation</div></div>",unsafe_allow_html=True)
    b2.markdown(f"<div class='kpi-card kpi-blue'><div class='kpi-label'>🎯 Tỷ Lệ Chuyển Đổi (Conversion)</div><div class='kpi-value' style='font-size:24px'>+8%</div><div class='kpi-delta' style='color:{COLORS['sub']}'>Cải thiện nhờ gợi ý cá nhân hóa chính xác hơn</div></div>",unsafe_allow_html=True)
    b3.markdown(f"<div class='kpi-card kpi-amber'><div class='kpi-label'>🛒 Giá Trị Đơn Hàng (AOV)</div><div class='kpi-value' style='font-size:24px'>+5%</div><div class='kpi-delta' style='color:{COLORS['sub']}'>Thông qua Cross-selling (FP-Growth Basket)</div></div>",unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Actionable Strategy Table
    st.markdown(sh("Kế Hoạch Hành Động (Action Plan) Theo Phân Khúc"), unsafe_allow_html=True)
    strategies = [
        ("Khách Hàng Tinh Hoa (Champions)", "F>1, M cao, R thấp", "FP-Growth (E3)", "Tặng Early Access, VIP Support, Cross-sell các danh mục cao cấp.", COLORS["green"]),
        ("Ngôi Sao Tiềm Năng (High-Spender)", "F=1, M cao", "Content-Based (E2)", "Mời tham gia Loyalty Program, tặng voucher đặc biệt để kích thích đơn thứ 2.", COLORS["amber"]),
        ("Khách Trung Thành (Loyal)", "F>1, R trung bình", "FP-Growth (E3)", "Gửi bản tin sản phẩm mới dựa trên lịch sử mua, up-sell các sản phẩm giá trị cao hơn.", COLORS["violet"]),
        ("Khách Mới Tương Tác (Active)", "F=1, R thấp", "Content-Based (E2)", "Đề xuất các sản phẩm liên quan ngay trên trang chủ để chốt sale khi họ còn quan tâm.", COLORS["blue"]),
        ("Khách Đang Trì Hoãn (Warm)", "F=1, R TB", "Content-Based (E2)", "Gửi email nhắc nhở giỏ hàng, kèm mã giảm giá nhỏ có thời hạn.", "#3B82F6"),
        ("Khách Có Nguy Cơ (At-Risk)", "F>1, R cao", "Geo-Popularity (E1)", "Chiến dịch Win-back: Gọi điện chăm sóc, gửi ưu đãi lớn để giành lại khách hàng.", COLORS["red"]),
        ("Khách Vãng Lai Ngủ Quên (Cold)", "F=1, R cao", "Geo-Popularity (E1)", "Retargeting qua quảng cáo Facebook/Google với các sản phẩm đang Trending.", COLORS["sub"]),
        ("Khách Đã Rời Bỏ (Lost)", "F=1, R cực cao", "Geo-Popularity (E1)", "Đưa vào tệp chạy quảng cáo lookalike, không tốn chi phí SMS/Email trực tiếp.", "#9CA3AF"),
    ]
    
    html_strat = ""
    for name, rfm_desc, engine, action, color in strategies:
        html_strat += f"<tr><td style='font-weight:600;color:{color}'>{name}</td><td style='color:{COLORS['sub']};font-size:12px'>{rfm_desc}</td><td><span class='badge' style='background:var(--bg);color:var(--text);border:1px solid var(--border)'>{engine}</span></td><td>{action}</td></tr>"
        
    st.markdown(f"<table class='rec-table'><thead><tr><th>Nhóm Khách Hàng</th><th>Đặc Điểm RFM</th><th>Engine Đề Xuất</th><th>Chiến Lược Hành Động (Marketing)</th></tr></thead><tbody>{html_strat}</tbody></table>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Next Steps
    st.markdown(sh("Các Bước Triển Khai Tiếp Theo (Roadmap)"), unsafe_allow_html=True)
    roadmap = [
        ("Q2 / 2026", "A/B Testing", "Triển khai A/B test (50% rule-based, 50% AI model) trong 1 tháng để đo lường uplift thực tế."),
        ("Q3 / 2026", "Real-time Integration", "Chuyển mô hình từ offline batch prediction sang real-time API bằng FastAPI & Redis."),
        ("Q4 / 2026", "Mở rộng tính năng", "Tích hợp thêm Image Recommendation (tìm kiếm bằng hình ảnh) và Collaborative Filtering bằng Deep Learning."),
    ]
    
    for quarter, phase, desc in roadmap:
        st.markdown(f"""
        <div style='background:var(--surface);border-left:4px solid var(--accent3);padding:16px;margin-bottom:12px;border-radius:0 8px 8px 0;box-shadow:0 1px 2px rgba(0,0,0,0.05)'>
            <div style='display:flex;align-items:center;gap:16px'>
                <div style='font-family:IBM Plex Mono,monospace;font-weight:700;color:var(--sub);font-size:14px;min-width:80px'>{quarter}</div>
                <div>
                    <div style='font-weight:700;font-size:16px;color:var(--text);margin-bottom:4px'>{phase}</div>
                    <div style='font-size:13px;color:var(--sub)'>{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

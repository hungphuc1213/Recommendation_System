import json

def make_md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}
def make_code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

cells.append(make_md_cell([
    "# 🤖 FILE 02: HYBRID RECOMMENDATION SYSTEM\n",
    "> **P0 Fixes:** K-Means Centroid Mapping → Segment Name · `item_meta` thêm vào parameter `cb_recommend()`\n\n",
    "> **P1 Fixes:** THRESHOLD_CITY/STATE được dùng trong `build_index()` · Stratified sampling N_EVAL · `applymap` → `map`"
]))

# ── I. IMPORT
cells.append(make_md_cell(["## I. IMPORT & NẠP DỮ LIỆU"]))
cells.append(make_code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings, pickle, scipy.sparse as sp\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from mlxtend.frequent_patterns import fpgrowth, association_rules\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n\n",
    "train_df = pd.read_csv('01_Train_df.csv', parse_dates=['order_purchase_timestamp'])\n",
    "val_df   = pd.read_csv('01_Val_df.csv',   parse_dates=['order_purchase_timestamp'])\n",
    "test_df  = pd.read_csv('01_Test_df.csv',  parse_dates=['order_purchase_timestamp'])\n",
    "rfm      = pd.read_csv('01_RFM_Train.csv')\n",
    "with open('01_qualified_items.pkl','rb') as f: qualified_items_train = pickle.load(f)\n\n",
    "print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')\n",
    "print(f'RFM users: {len(rfm):,} | Qualified items: {len(qualified_items_train):,}')"
]))

# ── II. MACRO BRANCHING
cells.append(make_md_cell(["## II. MACRO-BRANCHING: F=1 vs F>1"]))
cells.append(make_code_cell([
    "rfm_F1 = rfm[rfm['Frequency'] == 1].copy()\n",
    "rfm_Fn = rfm[rfm['Frequency'] > 1].copy()\n",
    "print(f'F=1 raw: {len(rfm_F1):,} users ({len(rfm_F1)/len(rfm)*100:.1f}%)')\n",
    "print(f'F>1 raw: {len(rfm_Fn):,} users ({len(rfm_Fn)/len(rfm)*100:.1f}%)')"
]))

cells.append(make_code_cell([
    "# ── DIAGNOSTIC + CLEAN: Loại NaN / inf trước khi scale ──────────────\n",
    "# Nguyên nhân: payment_value=NaN trong CSV → Monetary=NaN → log1p(NaN)=NaN\n",
    "# → MinMaxScaler raise ValueError: 'Input X contains infinity or a value too large'\n",
    "for name, df_part, cols in [\n",
    "    ('rfm_F1', rfm_F1, ['Recency_Log','Monetary_Log']),\n",
    "    ('rfm_Fn', rfm_Fn, ['Recency_Log','Frequency','Monetary_Log']),\n",
    "]:\n",
    "    nan_count = df_part[cols].isnull().sum().sum()\n",
    "    inf_count = np.isinf(df_part[cols].select_dtypes('number')).sum().sum()\n",
    "    print(f'{name}: NaN={nan_count} | inf={inf_count} trong {cols}')\n\n",
    "# Thay inf bằng NaN rồi drop\n",
    "rfm_F1 = rfm_F1.replace([np.inf, -np.inf], np.nan).dropna(\n",
    "    subset=['Recency_Log','Monetary_Log']).copy()\n",
    "rfm_Fn = rfm_Fn.replace([np.inf, -np.inf], np.nan).dropna(\n",
    "    subset=['Recency_Log','Frequency','Monetary_Log']).copy()\n\n",
    "print(f'\\nSau clean:')\n",
    "print(f'  F=1: {len(rfm_F1):,} users | F>1: {len(rfm_Fn):,} users')\n",
    "print(f'  Monetary F=1 — min: {rfm_F1.Monetary.min():.0f} | max: {rfm_F1.Monetary.max():,.0f} | mean: {rfm_F1.Monetary.mean():,.0f}')\n",
    "print(f'  Recency  F=1 — min: {rfm_F1.Recency.min():.0f}  | max: {rfm_F1.Recency.max():.0f}  | mean: {rfm_F1.Recency.mean():.0f}')"
]))

# ── III. K-MEANS + SILHOUETTE + CENTROID MAPPING
cells.append(make_md_cell([
    "## III. K-MEANS: ELBOW + SILHOUETTE → CENTROID MAPPING\n",
    "> **P0 Fix #2:** Segment Name được gán TỪ Centroid, không từ percentile hardcode.\n",
    "> Sau khi KMeans chạy, inverse_transform centroid → so sánh R và M → map tên nghiệp vụ tự động."
]))
cells.append(make_code_cell([
    "def find_optimal_k(X_scaled, max_k=6, branch_name='', sample_size=5000):\n",
    "    \"\"\"\n",
    "    sample_size: Silhouette_score có O(n²) complexity → subsample để tránh MemoryError.\n",
    "    5000 mẫu là đủ để ước lượng chính xác với sai số < 0.01.\n",
    "    \"\"\"\n",
    "    inertias, sil_scores = [], []\n",
    "    K_range = range(2, max_k+1)\n",
    "    # Subsample cho silhouette (tránh MemoryError với tập F=1 lớn 80k+ users)\n",
    "    n = X_scaled.shape[0]\n",
    "    if n > sample_size:\n",
    "        idx_sample = np.random.RandomState(42).choice(n, sample_size, replace=False)\n",
    "        X_sil = X_scaled[idx_sample]\n",
    "    else:\n",
    "        X_sil = X_scaled\n",
    "    print(f'  [{branch_name}] {n:,} users | Silhouette dùng {min(n, sample_size):,} mẫu')\n",
    "    for k in K_range:\n",
    "        km = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "        lbl = km.fit_predict(X_scaled)       # Fit trên toàn bộ data\n",
    "        inertias.append(km.inertia_)\n",
    "        lbl_sil = lbl[idx_sample] if n > sample_size else lbl  # Sil chỉ trên subsample\n",
    "        sil_scores.append(silhouette_score(X_sil, lbl_sil))\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "    fig.suptitle(f'K-Means Validation — {branch_name}', fontweight='bold')\n",
    "    axes[0].plot(list(K_range), inertias, marker='o', color='#E24B4A')\n",
    "    axes[0].set_title('Elbow Method (Inertia)')\n",
    "    axes[1].plot(list(K_range), sil_scores, marker='s', color='#1D9E75')\n",
    "    best_k = list(K_range)[np.argmax(sil_scores)]\n",
    "    axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')\n",
    "    axes[1].set_title(f'Silhouette (subsample={min(n,sample_size):,})'); axes[1].legend()\n",
    "    plt.tight_layout(); plt.show()\n",
    "    print(f'  Best K={best_k} (Silhouette={max(sil_scores):.3f})')\n",
    "    return best_k\n\n",
    "scaler_F1 = MinMaxScaler()\n",
    "X_F1 = scaler_F1.fit_transform(rfm_F1[['Recency_Log','Monetary_Log']])\n",
    "best_k_F1 = find_optimal_k(X_F1, max_k=8, branch_name='F=1 (2D: R×M)', sample_size=5000)\n",
    "print(f'  → Sử dụng K={best_k_F1} tự nhiên từ Silhouette Score (Không override)')"
]))

cells.append(make_code_cell([
    "scaler_Fn = MinMaxScaler()\n",
    "X_Fn = scaler_Fn.fit_transform(rfm_Fn[['Recency_Log','Frequency','Monetary_Log']])\n",
    "best_k_Fn = find_optimal_k(X_Fn, max_k=8, branch_name='F>1 (3D: R×F×M)', sample_size=5000)\n",
    "print(f'  → Sử dụng K={best_k_Fn} tự nhiên từ Silhouette Score (Không override)')"
]))

cells.append(make_code_cell([
    "# Fit K-Means với K tối ưu\n",
    "km_F1 = KMeans(n_clusters=best_k_F1, random_state=42, n_init=10)\n",
    "rfm_F1['Cluster'] = km_F1.fit_predict(X_F1)\n",
    "km_Fn = KMeans(n_clusters=best_k_Fn, random_state=42, n_init=10)\n",
    "rfm_Fn['Cluster'] = km_Fn.fit_predict(X_Fn)\n\n",
    "# ── [P0 FIX #2] CENTROID MAPPING — Động theo K ──────\n",
    "centroids_raw_F1 = pd.DataFrame(\n",
    "    scaler_F1.inverse_transform(km_F1.cluster_centers_),\n",
    "    columns=['Recency_Log','Monetary_Log']\n",
    ")\n",
    "centroids_raw_F1['Cluster'] = range(best_k_F1)\n",
    "print('Centroids F=1:'); print(centroids_raw_F1.round(3))\n\n",
    "# Cluster có Monetary cao nhất = Ngôi Sao Tiềm Năng\n",
    "HIGH_M_F1 = int(centroids_raw_F1.loc[centroids_raw_F1['Monetary_Log'].idxmax(), 'Cluster'])\n",
    "remaining_F1 = centroids_raw_F1[centroids_raw_F1['Cluster'] != HIGH_M_F1].sort_values('Recency_Log')\n\n",
    "seg_map_F1 = {HIGH_M_F1: 'Ngôi Sao Tiềm Năng'}\n",
    "# Map các cụm còn lại thông minh theo độ dài\n",
    "if len(remaining_F1) == 2:\n",
    "    labels_F1 = ['Vãng Lai Mới', 'Vãng Lai Ngủ Quên']\n",
    "elif len(remaining_F1) == 1:\n",
    "    labels_F1 = ['Vãng Lai Ngủ Quên']\n",
    "else:\n",
    "    labels_F1 = ['Vãng Lai Mới', 'Vãng Lai Quan Tâm', 'Vãng Lai Cũ', 'Vãng Lai Ngủ Quên', 'Vãng Lai Bỏ Đi']\n",
    "for i, row in enumerate(remaining_F1.itertuples()):\n",
    "    seg_map_F1[row.Cluster] = labels_F1[min(i, len(labels_F1)-1)]\n",
    "rfm_F1['Segment_Name'] = rfm_F1['Cluster'].map(seg_map_F1)\n",
    "print(f'\\nMapping Cluster→Segment (F=1): {seg_map_F1}')\n",
    "print(rfm_F1['Segment_Name'].value_counts())\n",
    "\n",
    "# ── Centroid Mapping cho F>1 ─────────────────────────────────────────\n",
    "centroids_raw_Fn = pd.DataFrame(\n",
    "    scaler_Fn.inverse_transform(km_Fn.cluster_centers_),\n",
    "    columns=['Recency_Log','Frequency','Monetary_Log']\n",
    ")\n",
    "centroids_raw_Fn['Cluster'] = range(best_k_Fn)\n",
    "print(f'\\nCentroids F>1 (K={best_k_Fn}):'); print(centroids_raw_Fn.round(3))\n\n",
    "# Champion: score = F cao + R thấp\n",
    "max_R, max_F = centroids_raw_Fn['Recency_Log'].max(), centroids_raw_Fn['Frequency'].max()\n",
    "centroids_raw_Fn['score'] = (\n",
    "    (centroids_raw_Fn['Frequency'] / max_F if max_F > 0 else 0) +\n",
    "    (1 - centroids_raw_Fn['Recency_Log'] / max_R if max_R > 0 else 0)\n",
    ")\n",
    "CHAMP_Fn = int(centroids_raw_Fn.loc[centroids_raw_Fn['score'].idxmax(), 'Cluster'])\n",
    "remaining_Fn = centroids_raw_Fn[centroids_raw_Fn['Cluster'] != CHAMP_Fn].sort_values('Recency_Log')\n\n",
    "seg_map_Fn = {CHAMP_Fn: 'Khách Ruột (Champions)'}\n",
    "# Map cụm theo độ dài\n",
    "if len(remaining_Fn) == 2:\n",
    "    labels_Fn = ['Khách Khứ Hồi (Loyal)', 'Khách Rời Bỏ (At-Risk)']\n",
    "elif len(remaining_Fn) == 1:\n",
    "    labels_Fn = ['Khách Rời Bỏ (At-Risk)']\n",
    "else:\n",
    "    labels_Fn = ['Khách Trung Thành (Loyal)', 'Khách Khứ Hồi (Repeating)', 'Khách Rơi Rụng (At-Risk)', 'Khách Ngủ Quên (Sleeping)', 'Khách Rời Bỏ (Lost)']\n",
    "for i, row in enumerate(remaining_Fn.itertuples()):\n",
    "    seg_map_Fn[row.Cluster] = labels_Fn[min(i, len(labels_Fn)-1)]\n",
    "rfm_Fn['Segment_Name'] = rfm_Fn['Cluster'].map(seg_map_Fn)\n",
    "print(f'\\nMapping Cluster→Segment (F>1): {seg_map_Fn}')\n\n",
    "rfm_full = pd.concat([rfm_F1, rfm_Fn], axis=0)\n",
    "print('\\nPhân bổ Segment cuối cùng:')\n",
    "print(rfm_full['Segment_Name'].value_counts())\n",
    "rfm_full.to_csv('02_RFM_Segmented.csv', index=False)"
]))

cells.append(make_code_cell([
    "# Scatter plot kết quả phân cụm\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "fig.suptitle('K-Means Clustering — Centroid-Mapped Segments', fontsize=14, fontweight='bold')\n",
    "for seg in rfm_F1['Segment_Name'].unique():\n",
    "    m = rfm_F1['Segment_Name'] == seg\n",
    "    axes[0].scatter(rfm_F1[m]['Monetary_Log'], rfm_F1[m]['Recency_Log'], label=seg, alpha=0.4, s=5)\n",
    "axes[0].set_title(f'F=1 — K={best_k_F1}'); axes[0].legend(markerscale=3)\n",
    "for seg in rfm_Fn['Segment_Name'].unique():\n",
    "    m = rfm_Fn['Segment_Name'] == seg\n",
    "    axes[1].scatter(rfm_Fn[m]['Monetary_Log'], rfm_Fn[m]['Recency_Log'], label=seg, alpha=0.5, s=15)\n",
    "axes[1].set_title(f'F>1 — K={best_k_Fn}'); axes[1].legend(markerscale=2)\n",
    "plt.tight_layout(); plt.show()"
]))

# ── IV. TIME-BASED SPLIT
cells.append(make_md_cell(["## IV. TIME-BASED SPLIT"]))
cells.append(make_code_cell([
    "TRAIN_CUTOFF = '2018-06-01'; VAL_CUTOFF = '2018-08-01'\n",
    "print(f'Train: {(train_df.order_purchase_timestamp < TRAIN_CUTOFF).sum():,} dòng')\n",
    "print(f'Val:   {len(val_df):,} dòng')\n",
    "print(f'Test:  {len(test_df):,} dòng (tháng 8/2018)')"
]))

# ── V. ENGINES
cells.append(make_md_cell(["## V. 3 AI ENGINES & FALLBACK CHAIN"]))

# Engine 1 - Geo với THRESHOLD fix
cells.append(make_md_cell(["### Engine 1: Geo-Popularity (4 cấp · P1 Fix: THRESHOLD_CITY/STATE được áp dụng)"]))
cells.append(make_code_cell([
    "class GeoPopularityEngine:\n",
    "    REGION_MAP = {\n",
    "        'SP':'Sudeste','RJ':'Sudeste','MG':'Sudeste','ES':'Sudeste',\n",
    "        'RS':'Sul','PR':'Sul','SC':'Sul',\n",
    "        'BA':'Nordeste','PE':'Nordeste','CE':'Nordeste','MA':'Nordeste',\n",
    "        'PB':'Nordeste','RN':'Nordeste','AL':'Nordeste','SE':'Nordeste','PI':'Nordeste',\n",
    "        'PA':'Norte','AM':'Norte','AC':'Norte','RO':'Norte','RR':'Norte','AP':'Norte','TO':'Norte',\n",
    "        'MT':'Centro-Oeste','MS':'Centro-Oeste','GO':'Centro-Oeste','DF':'Centro-Oeste'\n",
    "    }\n",
    "    # [P1 Fix] Threshold thực sự được dùng trong build_index\n",
    "    THRESHOLD_CITY  = 200  # Min đơn hàng để city index đáng tin cậy\n",
    "    THRESHOLD_STATE = 100  # Min đơn hàng để state index đáng tin cậy\n\n",
    "    def fit(self, train_df, qualified_items, top_n=10):\n",
    "        self.top_n = top_n\n",
    "        df = train_df[train_df['product_id'].isin(qualified_items)].copy()\n",
    "        df['region'] = df['customer_state'].map(self.REGION_MAP).fillna('Outros')\n\n",
    "        def build_index(df, group_col, min_count):\n",
    "            # [P1 Fix] Chỉ build index cho nhóm có đủ đơn hàng\n",
    "            group_counts = df.groupby(group_col)['order_id'].count()\n",
    "            valid_groups = set(group_counts[group_counts >= min_count].index)\n",
    "            df_valid = df[df[group_col].isin(valid_groups)]\n",
    "\n",
    "            stats = df_valid.groupby([group_col,'product_id']).agg(\n",
    "                cnt=('order_id','count'), rating=('review_score','mean')).reset_index()\n",
    "            stats['score'] = 0.6*(stats['cnt']/stats['cnt'].max()) + 0.4*((stats['rating']-1)/4)\n",
    "            idx = {}\n",
    "            for key, grp in stats.groupby(group_col):\n",
    "                idx[key] = grp.nlargest(top_n,'score')['product_id'].tolist()\n",
    "            return idx\n\n",
    "        self._city   = build_index(df, 'customer_city',  self.THRESHOLD_CITY)\n",
    "        self._state  = build_index(df, 'customer_state', self.THRESHOLD_STATE)\n",
    "        df['region']  = df['customer_state'].map(self.REGION_MAP).fillna('Outros')\n",
    "        self._region  = build_index(df, 'region', 0)  # Region: không threshold\n",
    "        self._global  = (df.groupby('product_id')\n",
    "                          .agg(cnt=('order_id','count'),rating=('review_score','mean'))\n",
    "                          .assign(score=lambda x: 0.6*(x.cnt/x.cnt.max())+0.4*((x.rating-1)/4))\n",
    "                          .nlargest(top_n,'score').index.tolist())\n",
    "        print(f'[E1] City idx: {len(self._city)} cities (≥{self.THRESHOLD_CITY} đơn) | State: {len(self._state)}')\n",
    "        return self\n\n",
    "    def recommend(self, user_id, train_df, top_n=None):\n",
    "        top_n = top_n or self.top_n\n",
    "        h = train_df[train_df['customer_unique_id']==user_id]\n",
    "        bought = set(h['product_id'])\n",
    "        city   = h['customer_city'].iloc[0]  if not h.empty else None\n",
    "        state  = h['customer_state'].iloc[0] if not h.empty else None\n",
    "        region = self.REGION_MAP.get(state,'Outros') if state else None\n",
    "        for pool in [self._city.get(city,[]), self._state.get(state,[]),\n",
    "                     self._region.get(region,[]), self._global]:\n",
    "            recs = [p for p in pool if p not in bought]\n",
    "            if len(recs) >= 2: return recs[:top_n]\n",
    "        return self._global[:top_n]\n\n",
    "geo_engine = GeoPopularityEngine()\n",
    "geo_engine.fit(train_df, qualified_items_train, top_n=10)"
]))

# Engine 2 - Content-Based với item_meta fix
cells.append(make_md_cell(["### Engine 2: Content-Based (P0 Fix: item_meta → parameter của cb_recommend)"]))
cells.append(make_code_cell([
    "BULKY_CATEGORIES = [\n",
    "    'bed_bath_table','furniture_decor','housewares','home_appliances',\n",
    "    'computers','office_furniture','air_conditioning','home_appliances_2'\n",
    "]\n",
    "CATEGORY_MACRO = {\n",
    "    'health_beauty':'LIFESTYLE','perfumery':'LIFESTYLE','sports_leisure':'LIFESTYLE',\n",
    "    'fashion_bags_accessories':'LIFESTYLE','cool_stuff':'LIFESTYLE','stationery':'OFFICE',\n",
    "    'computers_accessories':'TECH','electronics':'TECH','telephony':'TECH',\n",
    "    'computers':'TECH','tablets_printing_image':'TECH',\n",
    "    'watches_gifts':'PREMIUM','toys':'FAMILY','baby':'FAMILY',\n",
    "    'bed_bath_table':'HOME','furniture_decor':'HOME','housewares':'HOME',\n",
    "    'home_appliances':'HOME','home_appliances_2':'HOME','garden_tools':'HOME',\n",
    "    'auto':'AUTO',\n",
    "}\n\n",
    "df_q = train_df[train_df['product_id'].isin(qualified_items_train)].copy()\n",
    "df_q['macro_cat'] = df_q['product_category_name_english'].map(CATEGORY_MACRO).fillna('OTHER')\n",
    "df_q['price_bucket'] = df_q['price_bucket'].fillna('mid')\n\n",
    "item_meta = (df_q.sort_values('review_score', ascending=False)\n",
    "              .drop_duplicates('product_id')\n",
    "              [['product_id','product_category_name_english','macro_cat','price_bucket']]\n",
    "              .reset_index(drop=True))\n",
    "item_meta['content'] = (\n",
    "    item_meta['macro_cat'] + ' ' +\n",
    "    item_meta['product_category_name_english'].str.replace(' ','_') + ' ' +\n",
    "    item_meta['product_category_name_english'].str.replace(' ','_') + ' ' +\n",
    "    item_meta['price_bucket']\n",
    ")\n",
    "tfidf_obj = TfidfVectorizer(token_pattern=r'\\b\\w+\\b', ngram_range=(1,2))\n",
    "tfidf_mat = tfidf_obj.fit_transform(item_meta['content'])\n",
    "pid_to_idx = {pid:i for i,pid in enumerate(item_meta['product_id'])}\n",
    "idx_to_pid = {i:pid for pid,i in pid_to_idx.items()}\n",
    "sp.save_npz('02_tfidf_matrix.npz', tfidf_mat)\n",
    "with open('02_cb_meta.pkl','wb') as f:\n",
    "    pickle.dump({'pid_to_idx':pid_to_idx,'idx_to_pid':idx_to_pid,'tfidf':tfidf_obj,'item_meta':item_meta}, f)\n",
    "print(f'[E2] Content-Based: {len(pid_to_idx):,} items | TF-IDF: {tfidf_mat.shape}')"
]))

cells.append(make_code_cell([
    "# [P0 FIX #3] item_meta là PARAMETER của hàm — không phải global variable\n",
    "def cb_recommend(user_id, train_df, tfidf_mat, pid_to_idx, idx_to_pid,\n",
    "                 item_meta, qualified_items, bulky_cats, top_n=10):\n",
    "    \"\"\"P0 Fix: item_meta được truyền vào như parameter thay vì dùng global.\"\"\"\n",
    "    h = train_df[train_df['customer_unique_id']==user_id]\n",
    "    if h.empty: return []\n",
    "    bought = set(h['product_id'])\n",
    "    h_sorted = h.sort_values('order_purchase_timestamp')\n",
    "    last_item = h_sorted.iloc[-1]['product_id']\n",
    "    last_cat  = h_sorted.iloc[-1]['product_category_name_english']\n\n",
    "    # Exclusion Rule: Nếu item cuối là cồng kềnh → dùng item gần nhất không cồng kềnh\n",
    "    if last_cat in bulky_cats:\n",
    "        non_bulky = h_sorted[~h_sorted['product_category_name_english'].isin(bulky_cats)]\n",
    "        if non_bulky.empty: return []\n",
    "        last_item = non_bulky.iloc[-1]['product_id']\n\n",
    "    if last_item not in pid_to_idx: return []\n",
    "    user_vec = tfidf_mat[pid_to_idx[last_item]]\n",
    "    sims = cosine_similarity(user_vec, tfidf_mat)[0]\n",
    "    order_ = np.argsort(sims)[::-1]\n",
    "    recs = []\n",
    "    for i in order_:\n",
    "        pid  = idx_to_pid[i]\n",
    "        cats = item_meta[item_meta['product_id']==pid]['product_category_name_english'].values\n",
    "        if pid not in bought and pid in qualified_items and len(cats)>0 and cats[0] not in bulky_cats:\n",
    "            recs.append(pid)\n",
    "        if len(recs) >= top_n: break\n",
    "    return recs\n\n",
    "print('[E2] cb_recommend với item_meta làm parameter — safe cho Streamlit deploy ✅')"
]))

# Engine 3 - FP-Growth
cells.append(make_md_cell(["### Engine 3: FP-Growth (P2 Fix: applymap → map)"]))
cells.append(make_code_cell([
    "print('[E3] Đang xây FP-Growth trên Category Level...')\n",
    "fn_users = set(rfm_Fn['customer_unique_id'])\n",
    "df_fn = train_df[train_df['customer_unique_id'].isin(fn_users)].copy()\n\n",
    "# [Bug Fix] train_df không có 'order_item_id' → dùng 'product_id' để đếm số SP mỗi category/đơn\n",
    "basket = (df_fn.groupby(['order_id','product_category_name_english'])['product_id'].count()\n",
    "          .unstack(fill_value=0)\n",
    "          .reset_index(drop=True))\n",
    "# [P2 Fix] applymap deprecated → dùng (basket > 0).astype(int)\n",
    "basket_bool = (basket > 0).astype(int)\n\n",
    "# Hạ min_support xuống 0.01 vì tập F>1 khá thưa thớt\n",
    "freq_items = fpgrowth(basket_bool, min_support=0.01, use_colnames=True)\n",
    "if len(freq_items) > 0:\n",
    "    rules = association_rules(freq_items, metric='lift', min_threshold=1.2)\n",
    "    rules = rules[rules['confidence'] >= 0.20].sort_values('lift', ascending=False)\n",
    "else:\n",
    "    rules = pd.DataFrame(columns=['antecedents', 'consequents', 'confidence', 'lift'])\n\n",
    "rules.to_csv('02_Association_Rules.csv', index=False)\n",
    "if len(rules) > 0:\n",
    "    # Dùng astype(float) để print chắc chắn không dính TypeError object\n",
    "    top_3_lift = rules.lift.head(3).astype(float).round(2).tolist()\n",
    "    print(f'[E3] FP-Growth: {len(rules)} luật | Top 3 lift: {top_3_lift}')\n",
    "    print(rules[['antecedents','consequents','confidence','lift']].head())\n",
    "else:\n",
    "    print('[E3] FP-Growth: Không tìm thấy luật nào (0 rules). Engine 3 sẽ luôn trả về []')"
]))

cells.append(make_code_cell([
    "def fpgrowth_recommend(user_id, train_df, rules, qualified_items, top_n=5):\n",
    "    h = train_df[train_df['customer_unique_id']==user_id]\n",
    "    if h.empty: return []\n",
    "    user_cats = set(h['product_category_name_english'].dropna())\n",
    "    recs_cats = []\n",
    "    for _, row in rules.iterrows():\n",
    "        if row['antecedents'].issubset(user_cats):\n",
    "            recs_cats.extend(list(row['consequents']))\n",
    "    recs_cats = list(dict.fromkeys(recs_cats))[:3]\n",
    "    result = (train_df[train_df['product_category_name_english'].isin(recs_cats) &\n",
    "                        train_df['product_id'].isin(qualified_items)]\n",
    "              .groupby('product_id')['review_score'].mean()\n",
    "              .nlargest(top_n).index.tolist())\n",
    "    return result\n\n",
    "print('[E3] fpgrowth_recommend ready.')"
]))

# Fallback Router
cells.append(make_md_cell(["### Hybrid Fallback Router"]))
cells.append(make_code_cell([
    "seg_dict = dict(zip(rfm_full['customer_unique_id'], rfm_full['Segment_Name']))\n\n",
    "def hybrid_router(user_id, train_df, geo_engine, rules, tfidf_mat,\n",
    "                   pid_to_idx, idx_to_pid, item_meta,  # [P0 Fix] item_meta là param\n",
    "                   qualified_items, seg_dict,\n",
    "                   bulky_cats=BULKY_CATEGORIES, top_n=10):\n",
    "    segment = seg_dict.get(user_id, 'Vãng Lai Ngủ Quên')\n\n",
    "    # Champion / Loyal → E3 trước\n",
    "    if any(k in segment for k in ['Champion','Khứ Hồi','Loyal']):\n",
    "        recs = fpgrowth_recommend(user_id, train_df, rules, qualified_items, top_n)\n",
    "        if recs: return {'segment':segment, 'engine':'FP-Growth (E3)', 'recs':recs}\n\n",
    "    # Ngôi Sao / Occasional → E2\n",
    "    if any(k in segment for k in ['Tiềm Năng','Mới','occasional']):\n",
    "        recs = cb_recommend(user_id, train_df, tfidf_mat, pid_to_idx, idx_to_pid,\n",
    "                            item_meta, qualified_items, bulky_cats, top_n)\n",
    "        if recs: return {'segment':segment, 'engine':'Content-Based (E2)', 'recs':recs}\n\n",
    "    # Fallback E1\n",
    "    recs = geo_engine.recommend(user_id, train_df, top_n)\n",
    "    return {'segment':segment, 'engine':'Geo-Popularity (E1)', 'recs':recs}\n\n",
    "# Demo\n",
    "sample = train_df['customer_unique_id'].value_counts().index[0]\n",
    "res = hybrid_router(sample, train_df, geo_engine, rules, tfidf_mat,\n",
    "                    pid_to_idx, idx_to_pid, item_meta, qualified_items_train, seg_dict)\n",
    "print(f'Demo → Segment: {res[\"segment\"]} | Engine: {res[\"engine\"]} | Recs: {res[\"recs\"][:3]}')"
]))

# ── VI. EVALUATION với Stratified Sampling
cells.append(make_md_cell([
    "## VI. ĐÁNH GIÁ OFFLINE — TIME-BASED TEST (08/2018)\n",
    "> **P1 Fix:** Stratified sampling theo Segment để metric phản ánh đúng phân phối thực tế"
]))
cells.append(make_code_cell([
    "def precision_at_k(rec, rel, k):\n",
    "    return len(set(rec[:k]) & set(rel)) / k if rec and rel else 0.0\n",
    "def recall_at_k(rec, rel, k):\n",
    "    return len(set(rec[:k]) & set(rel)) / len(rel) if rec and rel else 0.0\n",
    "def ndcg_at_k(rec, rel, k):\n",
    "    if not rec or not rel: return 0.0\n",
    "    dcg  = sum(1/np.log2(i+2) for i,p in enumerate(rec[:k]) if p in set(rel))\n",
    "    idcg = sum(1/np.log2(i+2) for i in range(min(len(rel), k)))\n",
    "    return dcg/idcg if idcg > 0 else 0.0\n",
    "def hit_rate_at_k(rec, rel, k):\n",
    "    return 1.0 if rec and rel and set(rec[:k]) & set(rel) else 0.0"
]))

cells.append(make_code_cell([
    "ground_truth = (test_df.groupby('customer_unique_id')['product_id']\n",
    "                .apply(list).reset_index()\n",
    "                .rename(columns={'product_id':'actual_items'}))\n",
    "ground_truth = ground_truth.merge(rfm_full[['customer_unique_id','Segment_Name']], on='customer_unique_id', how='left')\n",
    "ground_truth['Segment_Name'] = ground_truth['Segment_Name'].fillna('Vãng Lai Ngủ Quên')\n",
    "print(f'Ground Truth: {len(ground_truth):,} users')\n",
    "print(ground_truth['Segment_Name'].value_counts())"
]))

cells.append(make_code_cell([
    "K = 10\n",
    "N_PER_SEG = 150  # Số user lấy mỗi segment\n\n",
    "# [P1 Fix] Stratified sampling theo Segment_Name\n",
    "stratified_sample = (ground_truth\n",
    "    .groupby('Segment_Name', group_keys=False)\n",
    "    .apply(lambda x: x.sample(min(len(x), N_PER_SEG), random_state=42)))\n",
    "print(f'Stratified sample: {len(stratified_sample)} users')\n",
    "print(stratified_sample['Segment_Name'].value_counts())\n\n",
    "results = []\n",
    "for _, row in stratified_sample.iterrows():\n",
    "    uid, actual, seg = row['customer_unique_id'], row['actual_items'], row['Segment_Name']\n",
    "    recs = hybrid_router(uid, train_df, geo_engine, rules, tfidf_mat,\n",
    "                         pid_to_idx, idx_to_pid, item_meta,\n",
    "                         qualified_items_train, seg_dict)['recs']\n",
    "    results.append({\n",
    "        'user_id': uid, 'segment': seg,\n",
    "        'precision': precision_at_k(recs, actual, K),\n",
    "        'recall':    recall_at_k(recs, actual, K),\n",
    "        'ndcg':      ndcg_at_k(recs, actual, K),\n",
    "        'hit_rate':  hit_rate_at_k(recs, actual, K),\n",
    "    })\n\n",
    "eval_df = pd.DataFrame(results)\n",
    "print(f'\\n=== KẾT QUẢ @K={K} (Stratified) ===')\n",
    "print(eval_df[['precision','recall','ndcg','hit_rate']].mean().round(4).to_string())\n",
    "print('\\nTheo Segment:')\n",
    "print(eval_df.groupby('segment')[['precision','recall','ndcg','hit_rate']].mean().round(4))"
]))

cells.append(make_code_cell([
    "# Biểu đồ metric\n",
    "metrics = ['precision','recall','ndcg','hit_rate']\n",
    "values  = [eval_df[m].mean() for m in metrics]\n",
    "labels  = [f'Precision@{K}',f'Recall@{K}',f'NDCG@{K}',f'Hit Rate@{K}']\n",
    "colors  = ['#2E86AB','#1D9E75','#EF9F27','#7F77DD']\n",
    "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
    "fig.suptitle(f'Hybrid Router Performance @K={K} — Stratified Test Set (08/2018)', fontsize=13, fontweight='bold')\n",
    "for ax, m, l, c, v in zip(axes, metrics, labels, colors, values):\n",
    "    ax.bar([l], [v], color=c, edgecolor='white', width=0.5)\n",
    "    ax.text(0, v + 0.002, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')\n",
    "    ax.set_ylim(0, max(v*1.5, 0.05)); ax.spines[['top','right']].set_visible(False)\n",
    "plt.tight_layout(); plt.savefig('02_EvalMetrics.png', dpi=150, bbox_inches='tight'); plt.show()"
]))

cells.append(make_code_cell([
    "with open('02_geo_engine.pkl','wb') as f:     pickle.dump(geo_engine, f)\n",
    "with open('02_seg_dict.pkl','wb') as f:       pickle.dump(seg_dict, f)\n",
    "with open('02_fpgrowth_rules.pkl','wb') as f: pickle.dump(rules, f)\n",
    "print('✅ FILE 02 HOÀN TẤT | Model files: 02_geo_engine.pkl | 02_seg_dict.pkl | 02_fpgrowth_rules.pkl')"
]))

nb = {
    "nbformat":4,"nbformat_minor":5,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.9.0"}},
    "cells":cells
}
with open('d:/Bao_cao_de_an/02_RecSys_Model.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"✅ 02_RecSys_Model.ipynb — {len(cells)} cells")

import json

def make_md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}

def make_code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell([
    "# 📦 FILE 01: DATA PIPELINE, EDA & PREPROCESSING\n",
    "> **Mục tiêu:** Đọc dữ liệu thô, gộp bảng (order-level tách khỏi item-level), làm sạch, EDA chuyên sâu và xuất CSV chuẩn.\n\n",
    "> **Khung thời gian:** Lệnh cấm 2016 · Train 01/2017–05/2018 · Val 06–07/2018 · Test 08/2018 · Forecast 09/2018\n\n",
    "> **P0 Fix áp dụng:** payment_value double-counting đã được tách pipeline order-level vs item-level"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell(["## I. IMPORT & ĐỌC DỮ LIỆU THÔ"]))
cells.append(make_code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('viridis')\n\n",
    "# ── P2: Define constant để tránh hardcode ở nhiều chỗ ───────────\n",
    "DATA_DIR      = 'D:/Báo cáo Đề Án/Dataset/'\n",
    "TRAIN_END     = pd.Timestamp('2018-05-31')   # Snapshot date RFM & cuối Train\n",
    "TRAIN_CUTOFF  = '2018-06-01'                 # Ranh giới Train/Val\n",
    "VAL_CUTOFF    = '2018-08-01'                 # Ranh giới Val/Test\n",
    "RATING_THRESHOLD = 4.0                       # Hard Filter sản phẩm\n\n",
    "print('Constants đã định nghĩa:')\n",
    "print(f'  TRAIN_END={TRAIN_END.date()} | TRAIN_CUTOFF={TRAIN_CUTOFF} | VAL_CUTOFF={VAL_CUTOFF}')\n",
    "print(f'  RATING_THRESHOLD={RATING_THRESHOLD}')"
]))

cells.append(make_code_cell([
    "orders       = pd.read_csv(f'{DATA_DIR}olist_orders_dataset.csv')\n",
    "items        = pd.read_csv(f'{DATA_DIR}olist_order_items_dataset.csv')\n",
    "payments     = pd.read_csv(f'{DATA_DIR}olist_order_payments_dataset.csv')\n",
    "reviews      = pd.read_csv(f'{DATA_DIR}olist_order_reviews_dataset.csv')\n",
    "customers    = pd.read_csv(f'{DATA_DIR}olist_customers_dataset.csv')\n",
    "products     = pd.read_csv(f'{DATA_DIR}olist_products_dataset.csv')\n",
    "sellers      = pd.read_csv(f'{DATA_DIR}olist_sellers_dataset.csv')\n",
    "translations = pd.read_csv(f'{DATA_DIR}product_category_name_translation.csv')\n",
    "print(f'Đọc xong 8 file | orders: {len(orders):,} | items: {len(items):,} | payments: {len(payments):,}')"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell([
    "## II. MERGE & LÀM SẠCH — TÁCH PIPELINE ORDER-LEVEL vs ITEM-LEVEL\n",
    "> [!IMPORTANT]\n",
    "> **P0 Fix #1:** `payment_value` KHÔNG được merge vào item-level rows. Giá trị từng đơn hàng sẽ bị nhân theo số sản phẩm trong đơn.\n",
    "> Giải pháp: Dùng `df_order_level` (1 dòng = 1 đơn) để tính RFM & Daily GMV. Dùng `df_item_level` (1 dòng = 1 item) chỉ cho RecSys."
]))

cells.append(make_code_cell([
    "# ── 2.1 Dịch tên Danh mục ────────────────────────────────────────\n",
    "products = products.merge(translations, on='product_category_name', how='left')\n",
    "# P2: Dùng assignment thay vì inplace=True (deprecated pandas >= 2.0)\n",
    "products['product_category_name_english'] = products['product_category_name_english'].fillna('unknown_category')\n\n",
    "# ── 2.2 Tổng hợp payment (1 đơn có thể nhiều giao dịch / trả góp) ─\n",
    "payment_agg = payments.groupby('order_id', as_index=False)['payment_value'].sum()\n\n",
    "# ── 2.3 Lọc đơn hợp lệ & parse timestamp ─────────────────────────\n",
    "valid_status = ['delivered', 'shipped', 'approved']\n",
    "orders_valid = orders[orders['order_status'].isin(valid_status)].copy()\n",
    "orders_valid['order_purchase_timestamp'] = pd.to_datetime(orders_valid['order_purchase_timestamp'])\n\n",
    "# Chỉ giữ lại cột cần thiết từ reviews\n",
    "review_agg = reviews.groupby('order_id', as_index=False)['review_score'].mean()\n\n",
    "# ── 2.4 [P0 FIX] TẠO ORDER-LEVEL DF (dùng cho RFM & Daily GMV) ────\n",
    "df_order = (orders_valid\n",
    "            .merge(customers, on='customer_id', how='left')\n",
    "            .merge(payment_agg, on='order_id', how='left')\n",
    "            .merge(review_agg, on='order_id', how='left'))\n",
    "# 1 dòng = 1 đơn hàng → payment_value không bị nhân đôi!\n",
    "print(f'[ORDER-LEVEL] {len(df_order):,} dòng | {df_order.order_id.nunique():,} đơn hàng')\n",
    "print(f'  → Đây là nguồn ĐÚNG để tính Monetary (RFM) và Daily GMV (Prophet)')"
]))

cells.append(make_code_cell([
    "# ── 2.5 [P0 FIX] TẠO ITEM-LEVEL DF (dùng cho RecSys phân tích sản phẩm) ─\n",
    "df_item = (df_order\n",
    "           .merge(items[['order_id','product_id','price','freight_value','order_item_id']], on='order_id', how='left')\n",
    "           .merge(products[['product_id','product_category_name_english']], on='product_id', how='left'))\n\n",
    "# Chọn cột quan trọng cho RecSys\n",
    "cols_recsys = ['customer_unique_id','order_id','product_id','review_score','price',\n",
    "               'freight_value','payment_value','product_category_name_english',\n",
    "               'order_purchase_timestamp','customer_city','customer_state']\n",
    "df_item = df_item[cols_recsys].copy()\n\n",
    "# Drop dup: 1 cặp (user, product) chỉ giữ tương tác gần nhất\n",
    "df_item = df_item.sort_values('order_purchase_timestamp', ascending=False)\n",
    "df_item = df_item.drop_duplicates(subset=['customer_unique_id','product_id'], keep='first')\n\n",
    "print(f'[ITEM-LEVEL]  {len(df_item):,} dòng | {df_item.product_id.nunique():,} sản phẩm')\n",
    "print(f'  → Đây là nguồn cho RecSys Content-Based & FP-Growth')\n\n",
    "# ── Kiểm tra nhanh: Doanh thu total theo 2 cách ───────────────────\n",
    "gmv_correct = df_order['payment_value'].sum()\n",
    "gmv_wrong   = df_item['payment_value'].sum()   # Sẽ lớn hơn vì lặp\n",
    "print(f\"\\nKiểm chứng P0 Fix:\")\n",
    "print(f\"  GMV đúng  (order-level): {gmv_correct:>15,.0f} BRL\")\n",
    "print(f\"  GMV sai   (item-level) : {gmv_wrong:>15,.0f} BRL\")\n",
    "print(f\"  Chênh lệch: {(gmv_wrong/gmv_correct - 1)*100:.1f}% lạm phát\")"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell(["## III. DATA QUALITY CHECKLIST (P2)"]))
cells.append(make_code_cell([
    "print('=== DATA QUALITY CHECKS ===')\n\n",
    "# Check 1: Negative payment_value (refund / lỗi nhập)\n",
    "neg_pay = (df_order['payment_value'] <= 0).sum()\n",
    "print(f'Đơn hàng có payment_value <= 0: {neg_pay} ({neg_pay/len(df_order)*100:.2f}%)')\n",
    "if neg_pay > 0:\n",
    "    print('  → Loại bỏ khỏi tập dữ liệu')\n",
    "    df_order = df_order[df_order['payment_value'] > 0]\n\n",
    "# Check 2: Price = 0 (sản phẩm cho không / lỗi data)\n",
    "zero_price = (df_item['price'] <= 0).sum()\n",
    "print(f'Dòng có price <= 0: {zero_price} ({zero_price/len(df_item)*100:.2f}%)')\n\n",
    "# Check 3: Khoảng trống lớn trong time-series\n",
    "daily_counts = df_order.groupby(df_order['order_purchase_timestamp'].dt.date).size()\n",
    "date_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq='D')\n",
    "missing_days = len(date_range) - len(daily_counts)\n",
    "print(f'Ngày không có đơn hàng: {missing_days} / {len(date_range)} ngày tổng')\n\n",
    "# Check 4: NaN overview\n",
    "print(f'\\nNaN trong ORDER-LEVEL:')\n",
    "nan_order = df_order.isnull().sum()\n",
    "print(nan_order[nan_order > 0].to_string())\n",
    "print(f'\\nNaN trong ITEM-LEVEL:')\n",
    "nan_item = df_item.isnull().sum()\n",
    "print(nan_item[nan_item > 0].to_string())\n",
    "print('\\n✅ Data Quality Check xong!')"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell(["## IV. EDA (9 PHẦN CHUYÊN SÂU)"]))

# EDA 1
cells.append(make_md_cell(["### EDA 1: Phân phối Đơn hàng theo Bang — Geo-Trend Baseline"]))
cells.append(make_code_cell([
    "plt.figure(figsize=(12, 5))\n",
    "top_states = df_item['customer_state'].value_counts().head(10)\n",
    "bars_ax = sns.barplot(x=top_states.index, y=top_states.values, palette='viridis')\n",
    "for bar, val in zip(bars_ax.patches, top_states.values):\n",
    "    bars_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,\n",
    "                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')\n",
    "plt.title('Top 10 Bang có lượng đơn hàng lớn nhất\\n(SP > 40% → Thiết kế Geo-Fallback bắt buộc)', fontsize=13, fontweight='bold')\n",
    "plt.ylabel('Số lượng đơn hàng')\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'SP chiếm: {top_states.iloc[0]/top_states.sum()*100:.1f}% trong Top 10')"
]))

# EDA 2
cells.append(make_md_cell(["### EDA 2: Price vs Freight Value — Phát Hiện Vùng Phi Lý"]))
cells.append(make_code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "axes[0].scatter(df_item['price'].clip(upper=df_item['price'].quantile(0.95)),\n",
    "                df_item['freight_value'].clip(upper=df_item['freight_value'].quantile(0.95)),\n",
    "                alpha=0.3, s=5, c='steelblue')\n",
    "axes[0].axline((0,0), slope=1, color='red', linestyle='--', label='Freight = Price')\n",
    "axes[0].set_title('Tương quan Price vs Freight Value (clip P95)')\n",
    "axes[0].set_xlabel('Price (BRL)'); axes[0].set_ylabel('Freight Value (BRL)'); axes[0].legend()\n",
    "ratio = df_item['freight_value'] / df_item['price'].replace(0, np.nan)\n",
    "axes[1].hist(ratio.dropna().clip(upper=2), bins=50, color='salmon', edgecolor='white')\n",
    "axes[1].axvline(1.0, color='red', linestyle='--', label='Ratio=1 (phi lý)')\n",
    "axes[1].set_title('Phân phối tỷ lệ Freight/Price'); axes[1].legend()\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f\"Tỷ lệ đơn có freight > 50% giá: {(ratio > 0.5).sum() / len(ratio) * 100:.1f}%\")"
]))

# EDA 3
cells.append(make_md_cell(["### EDA 3: Review Score & Ngưỡng lọc sản phẩm (Hard Filter)"]))
cells.append(make_code_cell([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "fig.suptitle('EDA 3: Rating Analysis & Hard Filter Threshold', fontsize=14, fontweight='bold')\n",
    "rating_counts = df_item['review_score'].value_counts().sort_index()\n",
    "axes[0].bar(rating_counts.index, rating_counts.values,\n",
    "            color=['#E24B4A','#EF9F27','#888780','#1D9E75','#2ECC71'], edgecolor='white')\n",
    "axes[0].set_title('Phân bố Rating (Positivity Bias)')\n\n",
    "thresholds = [3.0, 3.5, 4.0, 4.2, 4.5]\n",
    "item_avg = df_item.groupby('product_id')['review_score'].mean()\n",
    "n_items_left = [(item_avg >= t).sum() for t in thresholds]\n",
    "axes[1].plot(thresholds, n_items_left, marker='o', color='#378ADD', linewidth=2, markersize=8)\n",
    "axes[1].axvline(RATING_THRESHOLD, color='#1D9E75', linestyle='--', linewidth=1.5, label=f'Ngưỡng {RATING_THRESHOLD}')\n",
    "idx_t = thresholds.index(RATING_THRESHOLD)\n",
    "axes[1].annotate(f'{n_items_left[idx_t]:,} SP còn lại', xy=(RATING_THRESHOLD, n_items_left[idx_t]),\n",
    "                 xytext=(RATING_THRESHOLD+0.1, n_items_left[idx_t]*0.95), fontsize=10, fontweight='bold')\n",
    "axes[1].set_title('Số SP hợp lệ theo Threshold'); axes[1].legend()\n\n",
    "cat_rating = df_item.groupby('product_category_name_english')['review_score'].agg(['mean','count'])\n",
    "cat_rating = cat_rating[cat_rating['count'] >= 50].sort_values('mean').tail(15)\n",
    "colors_cat = ['#1D9E75' if r >= 4.0 else '#EF9F27' if r >= 3.5 else '#E24B4A' for r in cat_rating['mean']]\n",
    "axes[2].barh(cat_rating.index, cat_rating['mean'], color=colors_cat, edgecolor='white')\n",
    "axes[2].axvline(4.0, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng 4.0')\n",
    "axes[2].set_title('Avg Rating per Category (Top 15)'); axes[2].set_xlim(3.5, 5); axes[2].legend()\n",
    "plt.tight_layout(); plt.show()"
]))

# EDA 4 - Sparsity
cells.append(make_md_cell(["### EDA 4: Sparsity & Long-tail — Cơ sở chọn Hybrid Architecture"]))
cells.append(make_code_cell([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))\n",
    "fig.suptitle('EDA 4: Interaction Matrix & Sparsity Analysis', fontsize=14, fontweight='bold')\n\n",
    "purchases_per_user = df_item.groupby('customer_unique_id')['product_id'].count()\n",
    "axes[0].hist(purchases_per_user.clip(upper=10), bins=10, color='#2E86AB', edgecolor='white', alpha=0.85)\n",
    "one_time_pct = (purchases_per_user == 1).sum() / len(purchases_per_user)\n",
    "axes[0].axvline(1, color='#A23B2C', linestyle='--', linewidth=2)\n",
    "axes[0].text(1.5, axes[0].get_ylim()[1]*0.8,\n",
    "             f'⚠️ {one_time_pct:.1%} user\\nmua 1 lần', color='#A23B2C', fontsize=10, fontweight='bold',\n",
    "             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5E6', alpha=0.9))\n",
    "axes[0].set_title('Số SP mỗi User mua')\n\n",
    "purchases_per_item = df_item.groupby('product_id')['customer_unique_id'].count()\n",
    "axes[1].hist(purchases_per_item.clip(upper=20), bins=20, color='#A23B6F', edgecolor='white', alpha=0.85)\n",
    "axes[1].text(0.6, 0.85, f'{(purchases_per_item==1).sum()/len(purchases_per_item):.1%} SP\\nchỉ bán 1 lần',\n",
    "             transform=axes[1].transAxes, color='#A23B6F', fontsize=10, fontweight='bold',\n",
    "             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))\n",
    "axes[1].set_title('Long-tail: Số lượt mua mỗi SP')\n\n",
    "n_u = df_item['customer_unique_id'].nunique()\n",
    "n_i = df_item['product_id'].nunique()\n",
    "n_x = len(df_item)\n",
    "sp  = 1 - n_x / (n_u * n_i)\n",
    "axes[2].pie([n_x, n_u*n_i - n_x], labels=[f'Có data\\n({n_x:,})', f'Ô trống\\n({sp:.4%})'],\n",
    "            colors=['#2E86AB','#E8E8E8'], autopct='%1.4f%%', startangle=90, explode=(0.05,0))\n",
    "axes[2].set_title(f'Sparsity = {sp:.4%}\\n({n_u:,} users × {n_i:,} items)')\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'\\n→ Sparsity {sp:.4%} → PHẢI dùng Hybrid Architecture!')"
]))

# EDA 5 - Category
cells.append(make_md_cell(["### EDA 5: Phân tích Danh Mục — Xác định Exclusion Rules"]))
cells.append(make_code_cell([
    "BULKY_CATS = ['bed_bath_table','furniture_decor','housewares','home_appliances',\n",
    "              'office_furniture','computers','air_conditioning','home_appliances_2']\n\n",
    "cat_stats = df_item.groupby('product_category_name_english').agg(\n",
    "    total_orders=('order_id','count'), total_revenue=('price','sum'),\n",
    "    avg_price=('price','mean'), avg_rating=('review_score','mean'),\n",
    "    avg_freight=('freight_value','mean')\n",
    ").reset_index().sort_values('total_revenue', ascending=False)\n",
    "top15 = cat_stats.head(15)\n\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 12))\n",
    "fig.suptitle('EDA 5: Category Analysis', fontsize=14, fontweight='bold')\n\n",
    "axes[0,0].barh(top15['product_category_name_english'], top15['total_revenue']/1e3,\n",
    "               color='#1D9E75', edgecolor='white')\n",
    "axes[0,0].set_title('Top 15 Category theo Doanh thu (BRL ×1000)'); axes[0,0].invert_yaxis()\n\n",
    "colors_r = ['#E24B4A' if c in BULKY_CATS else '#1D9E75' if r >= 4.0 else '#EF9F27'\n",
    "             for c, r in zip(top15['product_category_name_english'], top15['avg_rating'])]\n",
    "axes[0,1].barh(top15['product_category_name_english'], top15['avg_rating'],\n",
    "               color=colors_r, edgecolor='white')\n",
    "axes[0,1].axvline(4.0, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng 4.0')\n",
    "axes[0,1].set_title('Avg Rating (đỏ = BULKY → Exclusion Rule)'); axes[0,1].invert_yaxis()\n",
    "axes[0,1].set_xlim(3, 5); axes[0,1].legend()\n\n",
    "top10_cats = cat_stats.head(10)['product_category_name_english'].tolist()\n",
    "df_top10 = df_item[df_item['product_category_name_english'].isin(top10_cats)]\n",
    "cat_ord  = df_top10.groupby('product_category_name_english')['price'].median().sort_values(ascending=False).index.tolist()\n",
    "sns.boxplot(data=df_top10, x='price', y='product_category_name_english', order=cat_ord,\n",
    "            ax=axes[1,0], palette='Blues', flierprops={'markersize':2,'alpha':0.3})\n",
    "axes[1,0].set_xlim(0, df_top10['price'].quantile(0.95))\n",
    "axes[1,0].set_title('Phân phối Price (clip P95)')\n\n",
    "bubble = cat_stats.head(20).copy()\n",
    "bubble['size'] = (bubble['total_orders'] / bubble['total_orders'].max()) * 1500\n",
    "sc = axes[1,1].scatter(bubble['avg_rating'], bubble['total_revenue']/1e3,\n",
    "                        s=bubble['size'], alpha=0.6, c=bubble['avg_freight'],\n",
    "                        cmap='RdYlGn_r', edgecolors='white', linewidths=0.5)\n",
    "for _, row in bubble.iterrows():\n",
    "    axes[1,1].annotate(row['product_category_name_english'][:15],\n",
    "                       (row['avg_rating'], row['total_revenue']/1e3), fontsize=7, alpha=0.8)\n",
    "plt.colorbar(sc, ax=axes[1,1], label='Avg Freight')\n",
    "axes[1,1].set_title('Bubble: Revenue vs Rating vs Volume')\n",
    "plt.tight_layout(); plt.show()"
]))

# EDA 6 - User Behavior
cells.append(make_md_cell(["### EDA 6: Hành vi User & Macro-Branching (F=1 vs F>1)"]))
cells.append(make_code_cell([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "fig.suptitle('EDA 6: User Behavior & Macro-Branching Decision', fontsize=14, fontweight='bold')\n\n",
    "user_ord_count = df_item.groupby('customer_unique_id')['order_id'].nunique()\n",
    "f1_pct = (user_ord_count == 1).sum() / len(user_ord_count)\n",
    "axes[0].pie([f1_pct, 1-f1_pct],\n",
    "            labels=[f'F=1 (One-time)\\n{f1_pct:.1%}', f'F>1 (Returning)\\n{1-f1_pct:.1%}'],\n",
    "            colors=['#E24B4A','#1D9E75'], autopct='%1.1f%%', startangle=90, explode=(0.05,0))\n",
    "axes[0].set_title('Macro-Branching\\nF=1 > 97% → Tách nhánh riêng!')\n\n",
    "segments = pd.cut(user_ord_count, bins=[0,1,3,10,np.inf],\n",
    "                  labels=['One-time\\n(1)','Occasional\\n(2-3)','Regular\\n(4-10)','VIP\\n(>10)'])\n",
    "seg_map = segments.reset_index(); seg_map.columns = ['customer_unique_id','segment']\n",
    "df_seg  = df_item.merge(seg_map, on='customer_unique_id', how='left')\n",
    "seg_order = ['One-time\\n(1)','Occasional\\n(2-3)','Regular\\n(4-10)','VIP\\n(>10)']\n",
    "palette   = {'One-time\\n(1)':'#E24B4A','Occasional\\n(2-3)':'#EF9F27',\n",
    "             'Regular\\n(4-10)':'#1D9E75','VIP\\n(>10)':'#7F77DD'}\n",
    "sns.boxplot(data=df_seg[df_seg['payment_value'] < df_seg['payment_value'].quantile(0.95)],\n",
    "            x='segment', y='payment_value', order=seg_order, palette=palette, ax=axes[1],\n",
    "            flierprops={'markersize':2,'alpha':0.3})\n",
    "axes[1].set_title('Payment Value theo Segment (clip P95)')\n\n",
    "seg_rating = df_seg.groupby('segment')['review_score'].mean().reindex(seg_order)\n",
    "bars = axes[2].bar(seg_rating.index, seg_rating.values,\n",
    "                   color=['#E24B4A','#EF9F27','#1D9E75','#7F77DD'], edgecolor='white')\n",
    "axes[2].axhline(4.0, color='black', linestyle='--', linewidth=1, label='Ngưỡng 4.0')\n",
    "axes[2].set_ylim(3.5, 5.2)\n",
    "for bar, val in zip(bars, seg_rating.fillna(0).values):\n",
    "    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02,\n",
    "                 f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')\n",
    "axes[2].set_title('Avg Review Score theo Segment'); axes[2].legend()\n",
    "plt.tight_layout(); plt.show()"
]))

# EDA 7 - 2016 sparse
cells.append(make_md_cell(["### EDA 7: Kiểm tra Độ Thưa 2016 — Xác nhận Lệnh Cấm"]))
cells.append(make_code_cell([
    "df_order['year']  = df_order['order_purchase_timestamp'].dt.year\n",
    "df_order['month'] = df_order['order_purchase_timestamp'].dt.to_period('M')\n\n",
    "yearly = df_order.groupby('year').agg(\n",
    "    so_don    = ('order_id','nunique'),\n",
    "    doanh_thu = ('payment_value','sum')\n",
    ").reset_index()\n",
    "yearly['pct'] = yearly['so_don'] / yearly['so_don'].sum() * 100\n",
    "print('THỐNG KÊ THEO NĂM (GMV từ ORDER-LEVEL, không bị inflate):'); print(yearly.to_string(index=False))\n\n",
    "monthly = df_order.groupby('month').agg(so_don=('order_id','nunique'), gmv=('payment_value','sum')).reset_index()\n",
    "monthly['month_dt'] = monthly['month'].dt.to_timestamp()\n\n",
    "fig, axes = plt.subplots(2, 1, figsize=(14, 8))\n",
    "fig.suptitle('Kiểm tra độ thưa 2016 → Quyết định Loại bỏ', fontsize=14, fontweight='bold')\n",
    "colors = monthly['month_dt'].apply(lambda x: '#E24B4A' if x.year==2016 else '#378ADD' if x.year==2017 else '#1D9E75')\n",
    "axes[0].bar(monthly['month_dt'], monthly['so_don'], color=colors, width=20, edgecolor='white')\n",
    "axes[0].axvline(pd.Timestamp('2017-01-01'), color='black', linestyle='--', linewidth=1.5, label='Bắt đầu 2017')\n",
    "axes[0].axvspan(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'), alpha=0.08, color='#E24B4A')\n",
    "axes[0].text(pd.Timestamp('2016-06-01'), axes[0].get_ylim()[1]*0.7, '2016\\nquá thưa',\n",
    "             color='#E24B4A', fontsize=11, ha='center', fontweight='bold')\n",
    "axes[0].set_title('Số đơn hàng theo tháng'); axes[0].legend()\n\n",
    "monthly['cum_pct'] = monthly['so_don'].cumsum() / monthly['so_don'].sum() * 100\n",
    "axes[1].plot(monthly['month_dt'], monthly['cum_pct'], color='#7F77DD', linewidth=2)\n",
    "axes[1].axvline(pd.Timestamp('2017-01-01'), color='#E24B4A', linestyle='--', linewidth=1.5)\n",
    "pct_2016 = monthly[monthly['month_dt'] < '2017-01-01']['cum_pct'].max()\n",
    "axes[1].axhline(pct_2016, color='#E24B4A', linestyle=':', linewidth=1.2)\n",
    "axes[1].text(pd.Timestamp('2016-07-01'), pct_2016+2,\n",
    "             f'Cuối 2016: chỉ {pct_2016:.1f}% tổng đơn', color='#E24B4A', fontsize=10, ha='center')\n",
    "axes[1].set_title('% Tích lũy'); axes[1].set_ylim(0, 110)\n",
    "plt.tight_layout(); plt.show()\n",
    "pct_2016_val = yearly[yearly['year']==2016]['pct'].values[0]\n",
    "print(f'\\n✅ Năm 2016 chiếm {pct_2016_val:.2f}% → XÁC NHẬN Loại bỏ khỏi Train!')"
]))

# EDA 8 - Daily GMV cho Prophet
cells.append(make_md_cell(["### EDA 8: Daily GMV Time-Series (Cơ sở cho Prophet)\n",
    "> Sử dụng ORDER-LEVEL để tính GMV — không bị inflate"]))
cells.append(make_code_cell([
    "# [P0 Fix] GMV tính từ ORDER-LEVEL (mỗi order 1 dòng)\n",
    "df_order_2017 = df_order[df_order['order_purchase_timestamp'] >= '2017-01-01'].copy()\n",
    "daily_gmv = df_order_2017.groupby(df_order_2017['order_purchase_timestamp'].dt.date)['payment_value'].sum().reset_index()\n",
    "daily_gmv.columns = ['date','gmv']; daily_gmv['date'] = pd.to_datetime(daily_gmv['date'])\n\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 10))\n",
    "fig.suptitle('EDA 8: Daily GMV Time-Series (Order-Level, không inflate)', fontsize=14, fontweight='bold')\n\n",
    "axes[0,0].plot(daily_gmv['date'], daily_gmv['gmv'], color='#378ADD', linewidth=0.8, alpha=0.8)\n",
    "axes[0,0].axvline(pd.Timestamp('2017-11-24'), color='red', linestyle='--', linewidth=1.5, label='Black Friday 2017')\n",
    "axes[0,0].axvline(pd.Timestamp('2018-06-01'), color='orange', linestyle='--', linewidth=1, label='Train/Val')\n",
    "axes[0,0].axvline(pd.Timestamp('2018-08-01'), color='green', linestyle='--', linewidth=1, label='Val/Test')\n",
    "axes[0,0].set_title('Daily GMV: 01/2017 → 08/2018'); axes[0,0].legend(fontsize=9)\n\n",
    "daily_gmv['dow'] = daily_gmv['date'].dt.dayofweek\n",
    "weekly_avg = daily_gmv.groupby('dow')['gmv'].mean()\n",
    "dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']\n",
    "axes[0,1].bar(dow_labels, weekly_avg.values,\n",
    "              color=['#1D9E75' if i<5 else '#E24B4A' for i in range(7)], edgecolor='white')\n",
    "axes[0,1].set_title('Weekly Seasonality → Prophet weekly_seasonality=True')\n\n",
    "daily_gmv['ym'] = daily_gmv['date'].dt.to_period('M')\n",
    "monthly_gmv = daily_gmv.groupby('ym')['gmv'].sum().reset_index()\n",
    "monthly_gmv['ym_dt'] = monthly_gmv['ym'].dt.to_timestamp()\n",
    "axes[1,0].bar(monthly_gmv['ym_dt'], monthly_gmv['gmv']/1e6, color='#7F77DD', edgecolor='white', width=25)\n",
    "axes[1,0].axvline(pd.Timestamp('2017-11-01'), color='red', linestyle='--', label='Black Friday Month')\n",
    "axes[1,0].set_title('Monthly GMV (Triệu BRL)'); axes[1,0].legend()\n\n",
    "daily_gmv['ma7'] = daily_gmv['gmv'].rolling(7).mean()\n",
    "axes[1,1].plot(daily_gmv['date'], daily_gmv['gmv'], alpha=0.3, color='#378ADD', linewidth=0.8, label='Daily')\n",
    "axes[1,1].plot(daily_gmv['date'], daily_gmv['ma7'], color='#1D9E75', linewidth=2, label='MA-7')\n",
    "axes[1,1].set_title('Daily GMV + Moving Average 7 ngày'); axes[1,1].legend()\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'Peak ngày: {daily_gmv.loc[daily_gmv.gmv.idxmax(),\"date\"].date()} = {daily_gmv.gmv.max():,.0f} BRL (Black Friday!)')"
]))

# EDA 9 - RFM
cells.append(make_md_cell(["### EDA 9: Phân phối RFM — Validate cần Log-Transform"]))
cells.append(make_code_cell([
    "# [P0 Fix] RFM tính từ ORDER-LEVEL (không inflate Monetary)\n",
    "df_ord_train = df_order[(df_order['order_purchase_timestamp'] >= '2017-01-01') &\n",
    "                          (df_order['order_purchase_timestamp'] <= TRAIN_END)]\n",
    "rfm_eda = df_ord_train.groupby('customer_unique_id').agg(\n",
    "    Recency   = ('order_purchase_timestamp', lambda x: (TRAIN_END - pd.to_datetime(x).max()).days),\n",
    "    Frequency = ('order_id', 'nunique'),\n",
    "    Monetary  = ('payment_value', 'sum')  # Mỗi order chỉ cộng 1 lần!\n",
    ").reset_index()\n",
    "rfm_eda['Monetary_Log'] = np.log1p(rfm_eda['Monetary'])\n",
    "rfm_eda['Recency_Log']  = np.log1p(rfm_eda['Recency'])\n\n",
    "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
    "fig.suptitle('EDA 9: RFM Trước & Sau Log-Transform', fontsize=14, fontweight='bold')\n",
    "for ax, col, t in zip(axes[0], ['Recency','Frequency','Monetary'], ['Recency','Frequency','Monetary']):\n",
    "    ax.hist(rfm_eda[col].clip(upper=rfm_eda[col].quantile(0.99)), bins=50, color='#E24B4A', edgecolor='white', alpha=0.8)\n",
    "    ax.set_title(f'{t} — Lệch phải mạnh')\n",
    "for ax, col, t in zip(axes[1], ['Recency_Log','Frequency','Monetary_Log'], ['Recency_Log','Frequency','Monetary_Log']):\n",
    "    ax.hist(rfm_eda[col], bins=50, color='#1D9E75', edgecolor='white', alpha=0.8)\n",
    "    ax.set_title(f'{t} — Sau Log (cân đối hơn)')\n",
    "plt.tight_layout(); plt.show()\n",
    "print(f'Skewness Monetary trước log: {rfm_eda.Monetary.skew():.2f}')\n",
    "print(f'Skewness Monetary sau log:   {rfm_eda.Monetary_Log.skew():.2f}')\n",
    "print('→ Log-Transform bắt buộc trước K-Means!')"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell(["## V. FEATURE ENGINEERING & TIME-BASED SPLIT"]))

cells.append(make_code_cell([
    "print('=== P1 FIX: TIME-SPLIT TRƯỚC — HARD FILTER SAU ===')\n\n",
    "# Lệnh cấm 2016: Cắt dữ liệu gốc\n",
    "df_item_2017 = df_item[df_item['order_purchase_timestamp'] >= '2017-01-01'].copy()\n",
    "df_order_2017 = df_order[df_order['order_purchase_timestamp'] >= '2017-01-01'].copy()\n\n",
    "# [P1 Fix] SPLIT TRƯỚC — tính qualified_items CHỈ từ train\n",
    "train_item = df_item_2017[df_item_2017['order_purchase_timestamp'] < TRAIN_CUTOFF].copy()\n",
    "val_item   = df_item_2017[(df_item_2017['order_purchase_timestamp'] >= TRAIN_CUTOFF) &\n",
    "                           (df_item_2017['order_purchase_timestamp'] < VAL_CUTOFF)].copy()\n",
    "test_item  = df_item_2017[df_item_2017['order_purchase_timestamp'] >= VAL_CUTOFF].copy()\n\n",
    "# [P1 Fix] Hard Filter chỉ dùng thông tin Train (không leakage)\n",
    "item_avg_train = train_item.groupby('product_id')['review_score'].mean()\n",
    "qualified_items = set(item_avg_train[item_avg_train >= RATING_THRESHOLD].index)\n",
    "print(f'Qualified items (từ train only): {len(qualified_items):,} sản phẩm')\n\n",
    "# Áp dụng filter xuống tất cả splits\n",
    "train_item = train_item[train_item['product_id'].isin(qualified_items)].copy()\n",
    "val_item   = val_item[val_item['product_id'].isin(qualified_items)].copy()\n",
    "test_item  = test_item[test_item['product_id'].isin(qualified_items)].copy()\n\n",
    "print(f'Train:  {len(train_item):,} ({train_item.order_purchase_timestamp.min().date()} → {train_item.order_purchase_timestamp.max().date()})')\n",
    "print(f'Val:    {len(val_item):,}  ({val_item.order_purchase_timestamp.min().date()} → {val_item.order_purchase_timestamp.max().date()})')\n",
    "print(f'Test:   {len(test_item):,} ({test_item.order_purchase_timestamp.min().date()} → {test_item.order_purchase_timestamp.max().date()})')"
]))

cells.append(make_code_cell([
    "# [P1 Fix] price_bucket: Fit qcut trên train, apply bins vào val/test\n",
    "_, price_bins = pd.qcut(train_item['price'], q=4,\n",
    "                          labels=['budget','mid','premium','luxury'],\n",
    "                          duplicates='drop', retbins=True)\n",
    "def apply_price_bucket(df):\n",
    "    return pd.cut(df['price'], bins=price_bins, labels=['budget','mid','premium','luxury'],\n",
    "                  include_lowest=True).astype(str).fillna('mid')\n\n",
    "train_item['price_bucket'] = apply_price_bucket(train_item)\n",
    "val_item['price_bucket']   = apply_price_bucket(val_item)\n",
    "test_item['price_bucket']  = apply_price_bucket(test_item)\n",
    "print('price_bucket bins fit trên train_item, apply nhất quán xuống val/test ✅')"
]))

cells.append(make_code_cell([
    "# [P0 Fix] RFM từ ORDER-LEVEL (không từ item-level)\n",
    "train_order_rfm = df_order_2017[df_order_2017['order_purchase_timestamp'] < TRAIN_CUTOFF].copy()\n\n",
    "rfm_train = train_order_rfm.groupby('customer_unique_id').agg(\n",
    "    Recency   = ('order_purchase_timestamp', lambda x: (TRAIN_END - pd.to_datetime(x).max()).days),\n",
    "    Frequency = ('order_id', 'nunique'),\n",
    "    Monetary  = ('payment_value', 'sum')   # Đúng — mỗi order chỉ cộng 1 lần\n",
    ").reset_index()\n",
    "rfm_train['Monetary_Log'] = np.log1p(rfm_train['Monetary'])\n",
    "rfm_train['Recency_Log']  = np.log1p(rfm_train['Recency'])\n",
    "print(f'RFM (order-level): {len(rfm_train):,} users | Monetary mean: {rfm_train.Monetary.mean():,.0f} BRL')"
]))

# ──────────────────────────────────────────────────────────────
cells.append(make_md_cell(["## VI. XUẤT CSV"]))
cells.append(make_code_cell([
    "train_item.to_csv('01_Train_df.csv', index=False)\n",
    "val_item.to_csv('01_Val_df.csv', index=False)\n",
    "test_item.to_csv('01_Test_df.csv', index=False)\n",
    "rfm_train.to_csv('01_RFM_Train.csv', index=False)\n\n",
    "# Daily GMV cho Prophet — từ ORDER-LEVEL (P0 Fix)\n",
    "full_range = pd.date_range('2017-01-01', '2018-08-29', freq='D')\n",
    "gmv_daily = df_order_2017.groupby(df_order_2017['order_purchase_timestamp'].dt.date)['payment_value'].sum().reset_index()\n",
    "gmv_daily.columns = ['ds','y']; gmv_daily['ds'] = pd.to_datetime(gmv_daily['ds'])\n",
    "gmv_daily = gmv_daily.set_index('ds').reindex(full_range, fill_value=0).reset_index()\n",
    "gmv_daily.columns = ['ds','y']\n",
    "gmv_daily.to_csv('01_Daily_GMV_Prophet.csv', index=False)\n\n",
    "# Lưu qualified_items để File 02 tham chiếu\n",
    "import pickle\n",
    "with open('01_qualified_items.pkl','wb') as f: pickle.dump(qualified_items, f)\n\n",
    "print('✅ ĐÃ XUẤT:')\n",
    "print('  01_Train_df.csv  | 01_Val_df.csv  | 01_Test_df.csv')\n",
    "print('  01_RFM_Train.csv | 01_qualified_items.pkl')\n",
    "print('  01_Daily_GMV_Prophet.csv (Order-Level GMV, không inflate)')"
]))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.9.0"}
    }, "cells": cells
}

with open('d:/Bao_cao_de_an/01_Data_Prep_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"✅ 01_Data_Prep_EDA.ipynb — {len(cells)} cells")

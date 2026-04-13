import json

def make_md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}
def make_code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

cells = []

cells.append(make_md_cell([
    "# 🔮 FILE 03: SALES DEMAND FORECASTING (PROPHET)\n",
    "> **P0 Fix #3:** Carnaval: 1 entry duy nhất `ds=27/02`, `lower=-2`, `upper=4` (không còn double-counting)\n\n",
    "> **P1 Fix:** Bổ sung 3 ngày lễ còn thiếu: Páscoa, Labour Day, Tiradentes\n\n",
    "> **P2 Fix:** Grid Search có Heatmap visualization · Giải thích `periods=61`"
]))

# I. IMPORT
cells.append(make_md_cell(["## I. IMPORT & NẠP TIME-SERIES"]))
cells.append(make_code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "import seaborn as sns\n",
    "import warnings, pickle\n",
    "warnings.filterwarnings('ignore')\n",
    "from prophet import Prophet\n",
    "from prophet.diagnostics import cross_validation, performance_metrics\n",
    "from itertools import product as iterproduct\n\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n\n",
    "ts_df = pd.read_csv('01_Daily_GMV_Prophet.csv', parse_dates=['ds'])\n",
    "print(f'Time-series: {len(ts_df)} ngày ({ts_df.ds.min().date()} → {ts_df.ds.max().date()})')\n",
    "print(f'GMV total:   {ts_df.y.sum():,.0f} BRL (Order-Level, không inflate)')\n",
    "print(f'Ngày GMV=0:  {(ts_df.y == 0).sum()} ngày (đã fill_value=0)')\n",
    "print(f'Peak:        {ts_df.loc[ts_df.y.idxmax(),\"ds\"].date()} = {ts_df.y.max():,.0f} BRL')"
]))

# II. HOLIDAYS - P0 + P1 Fix
cells.append(make_md_cell([
    "## II. HOLIDAY EFFECTS — 15 NGÀY LỄ BRAZIL (ĐẦY ĐỦ)\n",
    "> **P0 Fix #3:** Carnaval gộp thành 1 entry `ds=27/02`, `lower_window=-2`, `upper_window=4`\n\n",
    "> **P1 Fix:** Bổ sung Páscoa, Dia do Trabalho, Tiradentes"
]))
cells.append(make_code_cell([
    "holidays_df = pd.DataFrame([\n",
    "    # ── NHÓM A: SPIKE DOANH THU ─────────────────────────────────────────\n",
    "    # Black Friday — Hiệu ứng lan toả 3 ngày trước & 3 ngày sau\n",
    "    {'holiday':'Black_Friday', 'ds':'2017-11-24', 'lower_window':-3, 'upper_window':3},\n",
    "    {'holiday':'Black_Friday', 'ds':'2018-11-23', 'lower_window':-3, 'upper_window':3},\n",
    "    # Cyber Monday — 3 ngày sau Black Friday\n",
    "    {'holiday':'Cyber_Monday', 'ds':'2017-11-27', 'lower_window':0, 'upper_window':1},\n",
    "    {'holiday':'Cyber_Monday', 'ds':'2018-11-26', 'lower_window':0, 'upper_window':1},\n",
    "    # Dia dos Namorados (Valentine Brazil) — mua quà 3 ngày trước\n",
    "    {'holiday':'Namorados',    'ds':'2017-06-12', 'lower_window':-3, 'upper_window':0},\n",
    "    {'holiday':'Namorados',    'ds':'2018-06-12', 'lower_window':-3, 'upper_window':0},\n",
    "    # Natal — tuần mua sắm trước Giáng Sinh\n",
    "    {'holiday':'Natal',        'ds':'2017-12-25', 'lower_window':-7, 'upper_window':0},\n",
    "    {'holiday':'Natal',        'ds':'2018-12-25', 'lower_window':-7, 'upper_window':0},\n",
    "    # Dia das Criancas (Thiếu Nhi)\n",
    "    {'holiday':'Criancas',     'ds':'2017-10-12', 'lower_window':-2, 'upper_window':0},\n",
    "    {'holiday':'Criancas',     'ds':'2018-10-12', 'lower_window':-2, 'upper_window':0},\n",
    "    # Dia dos Pais (Father's Day Brazil)\n",
    "    {'holiday':'Dia_dos_Pais', 'ds':'2017-08-13', 'lower_window':-2, 'upper_window':0},\n",
    "    {'holiday':'Dia_dos_Pais', 'ds':'2018-08-12', 'lower_window':-2, 'upper_window':0},\n\n",
    "    # ── NHÓM B: DROP DOANH THU ───────────────────────────────────────────\n",
    "    # [P0 Fix #3] Carnaval: 1 entry duy nhất thay vì 4 entry riêng lẻ\n",
    "    # ds = ngày đầu chính thức (Thứ 2) | lower=-2 (Th7 trước) | upper=4 (Th7 sau)\n",
    "    # Tổng: 7 ngày — không còn double-counting ngày 28/02!\n",
    "    {'holiday':'Carnaval',     'ds':'2017-02-27', 'lower_window':-2, 'upper_window':4},\n",
    "    {'holiday':'Carnaval',     'ds':'2018-02-12', 'lower_window':-2, 'upper_window':4},\n",
    "    # Ano Novo\n",
    "    {'holiday':'Ano_Novo',     'ds':'2017-01-01', 'lower_window':0, 'upper_window':1},\n",
    "    {'holiday':'Ano_Novo',     'ds':'2018-01-01', 'lower_window':0, 'upper_window':1},\n",
    "    # Independencia — rơi vào forecast horizon tháng 9/2018!\n",
    "    {'holiday':'Independencia','ds':'2017-09-07', 'lower_window':0, 'upper_window':0},\n",
    "    {'holiday':'Independencia','ds':'2018-09-07', 'lower_window':0, 'upper_window':0},\n\n",
    "    # ── [P1 Fix] BỔ SUNG 3 NGÀY LỄ CÒN THIẾU ────────────────────────────\n",
    "    # Tiradentes — nghỉ lễ nhỏ tháng 4\n",
    "    {'holiday':'Tiradentes',   'ds':'2017-04-21', 'lower_window':0, 'upper_window':0},\n",
    "    {'holiday':'Tiradentes',   'ds':'2018-04-21', 'lower_window':0, 'upper_window':0},\n",
    "    # Dia do Trabalho (Labour Day) — nghỉ lễ tháng 5\n",
    "    {'holiday':'Labour_Day',   'ds':'2017-05-01', 'lower_window':0, 'upper_window':0},\n",
    "    {'holiday':'Labour_Day',   'ds':'2018-05-01', 'lower_window':0, 'upper_window':0},\n",
    "    # Páscoa (Phục Sinh) — ngày thay đổi từng năm!\n",
    "    {'holiday':'Pascoa',       'ds':'2017-04-16', 'lower_window':-1, 'upper_window':1},\n",
    "    {'holiday':'Pascoa',       'ds':'2018-04-01', 'lower_window':-1, 'upper_window':1},\n",
    "])\n",
    "holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])\n",
    "print(f'Holiday DataFrame: {len(holidays_df)} entries | {holidays_df.holiday.nunique()} loại ngày lễ')\n",
    "print('\\nDanh sách:')\n",
    "print(holidays_df.groupby('holiday')[['lower_window','upper_window']].first().to_string())\n",
    "print('\\n✅ Carnaval: 1 entry/năm (không còn 4 entry gây double-counting!)')"
]))

# III. SPLIT
cells.append(make_md_cell(["## III. PHÂN TÁCH TẬP DỮ LIỆU"]))
cells.append(make_code_cell([
    "train_ts    = ts_df[ts_df['ds'] <= '2018-05-31'].copy()\n",
    "val_ts      = ts_df[(ts_df['ds'] > '2018-05-31') & (ts_df['ds'] <= '2018-07-31')].copy()\n",
    "test_ts     = ts_df[ts_df['ds'] > '2018-07-31'].copy()\n",
    "trainval_ts = ts_df[ts_df['ds'] <= '2018-07-31'].copy()\n\n",
    "print(f'Train:    {len(train_ts)} ngày ({train_ts.ds.min().date()} → {train_ts.ds.max().date()})')\n",
    "print(f'Val:      {len(val_ts)} ngày  ({val_ts.ds.min().date()} → {val_ts.ds.max().date()})')\n",
    "print(f'Test:     {len(test_ts)} ngày  ({test_ts.ds.min().date()} → {test_ts.ds.max().date()})')\n",
    "print(f'TrainVal: {len(trainval_ts)} ngày (dùng để fit Final Model)')"
]))

# IV. GRID SEARCH với Heatmap
cells.append(make_md_cell([
    "## IV. GRID SEARCH TÌM SIÊU THAM SỐ TỐI ƯU\n",
    "> **P2 Fix:** Kết quả được hiển thị bằng Heatmap để thấy tham số nào ảnh hưởng lớn nhất đến MAPE"
]))
cells.append(make_code_cell([
    "param_grid = {\n",
    "    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3],\n",
    "    'seasonality_prior_scale': [0.1, 1.0, 10.0],\n",
    "    'holidays_prior_scale':    [0.1, 1.0, 10.0],\n",
    "}\n",
    "all_params = [dict(zip(param_grid.keys(), v)) for v in iterproduct(*param_grid.values())]\n",
    "print(f'Tổng tổ hợp: {len(all_params)} | Chạy Grid Search (5-15 phút)...')\n\n",
    "mape_results = []\n",
    "for i, params in enumerate(all_params):\n",
    "    m = Prophet(\n",
    "        holidays=holidays_df,\n",
    "        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,\n",
    "        changepoint_prior_scale=params['changepoint_prior_scale'],\n",
    "        seasonality_prior_scale=params['seasonality_prior_scale'],\n",
    "        holidays_prior_scale=params['holidays_prior_scale'],\n",
    "    )\n",
    "    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)\n",
    "    m.fit(train_ts)\n",
    "    fc = m.predict(val_ts[['ds']])\n",
    "    merged = val_ts.merge(fc[['ds','yhat']], on='ds')\n",
    "    mask = merged['y'] > 0\n",
    "    mape = (np.abs(merged.loc[mask,'y'] - merged.loc[mask,'yhat']) / merged.loc[mask,'y']).mean()\n",
    "    mape_results.append({**params, 'mape': mape})\n",
    "    if (i+1) % 6 == 0:\n",
    "        print(f'  [{i+1}/{len(all_params)}] Best MAPE: {min(r[\"mape\"] for r in mape_results):.4f}')\n\n",
    "results_df = pd.DataFrame(mape_results).sort_values('mape')\n",
    "best_params = results_df.iloc[0].to_dict()\n",
    "print(f'\\n✅ BEST PARAMS:')\n",
    "print(f'  changepoint_prior_scale: {best_params[\"changepoint_prior_scale\"]}')\n",
    "print(f'  seasonality_prior_scale: {best_params[\"seasonality_prior_scale\"]}')\n",
    "print(f'  holidays_prior_scale:    {best_params[\"holidays_prior_scale\"]}')\n",
    "print(f'  MAPE on Val set:         {best_params[\"mape\"]:.4f} ({best_params[\"mape\"]*100:.2f}%)')"
]))

cells.append(make_code_cell([
    "# [P2 Fix] Heatmap: changepoint_prior_scale vs seasonality_prior_scale\n",
    "pivot = results_df.groupby(['changepoint_prior_scale','seasonality_prior_scale'])['mape'].min().unstack()\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "fig.suptitle('Grid Search Heatmap — MAPE theo tổ hợp tham số', fontsize=13, fontweight='bold')\n\n",
    "sns.heatmap(pivot * 100, annot=True, fmt='.2f', cmap='RdYlGn_r',\n",
    "            ax=axes[0], cbar_kws={'label':'MAPE (%)'})\n",
    "axes[0].set_title('changepoint_prior × seasonality_prior (min MAPE)')\n",
    "axes[0].set_xlabel('seasonality_prior_scale'); axes[0].set_ylabel('changepoint_prior_scale')\n\n",
    "pivot2 = results_df.groupby(['changepoint_prior_scale','holidays_prior_scale'])['mape'].min().unstack()\n",
    "sns.heatmap(pivot2 * 100, annot=True, fmt='.2f', cmap='RdYlGn_r',\n",
    "            ax=axes[1], cbar_kws={'label':'MAPE (%)'})\n",
    "axes[1].set_title('changepoint_prior × holidays_prior (min MAPE)')\n",
    "axes[1].set_xlabel('holidays_prior_scale'); axes[1].set_ylabel('changepoint_prior_scale')\n",
    "plt.tight_layout()\n",
    "plt.savefig('03_GridSearch_Heatmap.png', dpi=150, bbox_inches='tight'); plt.show()"
]))

# V. FINAL MODEL + ROLLING CV
cells.append(make_md_cell(["## V. FIT FINAL MODEL & ROLLING WINDOW CROSS-VALIDATION"]))
cells.append(make_code_cell([
    "final_model = Prophet(\n",
    "    holidays=holidays_df,\n",
    "    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,\n",
    "    changepoint_prior_scale=best_params['changepoint_prior_scale'],\n",
    "    seasonality_prior_scale=best_params['seasonality_prior_scale'],\n",
    "    holidays_prior_scale=best_params['holidays_prior_scale'],\n",
    "    interval_width=0.80,\n",
    ")\n",
    "final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)\n",
    "final_model.fit(trainval_ts)\n",
    "print('✅ Final Model fit xong trên Train+Val (01/2017 → 07/2018)')"
]))

cells.append(make_code_cell([
    "print('Rolling Window Cross-Validation...')\n",
    "df_cv = cross_validation(\n",
    "    final_model, initial='365 days', period='30 days', horizon='30 days', parallel='threads')\n",
    "df_perf = performance_metrics(df_cv)\n",
    "print(f'\\n=== CV METRICS ===')\n",
    "print(f'MAPE TB: {df_perf.mape.mean()*100:.2f}% | RMSE: {df_perf.rmse.mean():,.0f} BRL/ngày | MAE: {df_perf.mae.mean():,.0f} BRL/ngày')\n\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "fig.suptitle('Rolling CV: Sai số theo Horizon (Ngày 1→30)', fontsize=13, fontweight='bold')\n",
    "for ax, metric, label, color in zip(axes, ['mape','rmse','mae'],\n",
    "                                     ['MAPE (%)','RMSE (BRL)','MAE (BRL)'],\n",
    "                                     ['#E24B4A','#378ADD','#1D9E75']):\n",
    "    dh = df_perf.groupby('horizon')[metric].mean().reset_index()\n",
    "    dh['days'] = dh['horizon'].dt.days\n",
    "    mul = 100 if metric=='mape' else 1\n",
    "    ax.plot(dh['days'], dh[metric]*mul, color=color, linewidth=2, marker='o', markersize=4)\n",
    "    ax.fill_between(dh['days'], dh[metric]*mul, alpha=0.1, color=color)\n",
    "    ax.axvline(7, color='gray', linestyle='--', alpha=0.5, label='Day 7')\n",
    "    ax.set_title(label); ax.set_xlabel('Horizon (ngày)'); ax.legend()\n",
    "plt.tight_layout(); plt.show()"
]))

# VI. ORACLE FORECAST
cells.append(make_md_cell([
    "## VI. ORACLE FORECAST — DỰ BÁO 30 NGÀY THÁNG 9/2018\n",
    "> **P2 Fix:** periods=61 được giải thích rõ: 31 ngày tháng 8 + 30 ngày tháng 9 = 61 ngày"
]))
cells.append(make_code_cell([
    "# periods=61: 01/08 → 31/08 (31 ngày) + 01/09 → 30/09 (30 ngày)\n",
    "# TrainVal kết thúc 31/07/2018 → future từ 01/08 → 30/09/2018\n",
    "future = final_model.make_future_dataframe(periods=61, freq='D')  # 31 tháng 8 + 30 tháng 9\n",
    "forecast = final_model.predict(future)\n\n",
    "f_test = forecast[(forecast['ds'] >= '2018-08-01') & (forecast['ds'] <= '2018-08-29')]\n",
    "f_sep  = forecast[(forecast['ds'] >= '2018-09-01') & (forecast['ds'] <= '2018-09-30')]\n\n",
    "test_merged = test_ts.merge(f_test[['ds','yhat']], on='ds', how='left')\n",
    "mask_test = test_merged['y'] > 0\n",
    "mape_test = (np.abs(test_merged.loc[mask_test,'y'] - test_merged.loc[mask_test,'yhat']) /\n",
    "             test_merged.loc[mask_test,'y']).mean()\n",
    "rmse_test = np.sqrt(((test_merged['y'] - test_merged['yhat'])**2).mean())\n\n",
    "print(f'=== TEST SET (08/2018) ===')\n",
    "print(f'MAPE: {mape_test*100:.2f}% | RMSE: {rmse_test:,.0f} BRL/ngày')\n",
    "print(f'\\n=== ORACLE THÁNG 9/2018 ===')\n",
    "print(f'GMV dự báo: {f_sep.yhat.sum():,.0f} BRL')\n",
    "print(f'GMV TB/ngày: {f_sep.yhat.mean():,.0f} BRL')\n",
    "print(f'Ngày đỉnh: {f_sep.loc[f_sep.yhat.idxmax(),\"ds\"].date()} = {f_sep.yhat.max():,.0f} BRL')\n",
    "print(f'Upper 80%: {f_sep.yhat_upper.sum():,.0f} BRL | Lower 80%: {f_sep.yhat_lower.sum():,.0f} BRL')"
]))

# VII. VISUALIZATION
cells.append(make_md_cell(["## VII. ORACLE VISUALIZATION — 3 LỚP BIỂU ĐỒ"]))
cells.append(make_code_cell([
    "fig, ax = plt.subplots(figsize=(18, 7))\n",
    "hist_120 = ts_df[ts_df['ds'] >= '2018-05-01']\n",
    "ax.scatter(hist_120['ds'], hist_120['y'], color='#2C3E50', s=12, alpha=0.6, zorder=3, label='Thực tế (dots)')\n",
    "ax.fill_between(f_sep['ds'], f_sep['yhat_lower'], f_sep['yhat_upper'],\n",
    "                alpha=0.25, color='#3498DB', label='Dải sương mù 80% CI')\n",
    "ax.plot(f_sep['ds'], f_sep['yhat'], color='#3498DB', linewidth=2.5, zorder=4, label='Dự báo 09/2018')\n",
    "ax.axvline(pd.Timestamp('2018-09-01'), color='gray', linestyle='--', linewidth=1.5, label='Ranh giới Tương lai')\n",
    "ax.axvline(pd.Timestamp('2018-09-07'), color='orange', linestyle=':', linewidth=1.5, label='Independencia 07/09')\n",
    "peak = f_sep.loc[f_sep['yhat'].idxmax()]\n",
    "ax.annotate(f\"Peak: {peak['yhat']:,.0f} BRL\\n({peak['ds'].date()})\",\n",
    "            xy=(peak['ds'], peak['yhat']),\n",
    "            xytext=(peak['ds'] - pd.Timedelta(days=6), peak['yhat']*1.05),\n",
    "            fontsize=10, fontweight='bold', color='#2980B9',\n",
    "            arrowprops=dict(arrowstyle='->', color='#2980B9'))\n",
    "ax.set_title('ORACLE FORECAST: Doanh thu Olist — Tháng 9/2018\\n(GMV từ Order-Level, không inflate)', fontsize=14, fontweight='bold')\n",
    "ax.legend(fontsize=10); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))\n",
    "fig.autofmt_xdate()\n",
    "plt.tight_layout()\n",
    "plt.savefig('03_Oracle_Forecast.png', dpi=200, bbox_inches='tight'); plt.show()\n",
    "print('✅ Oracle Visualization lưu: 03_Oracle_Forecast.png')"
]))

cells.append(make_code_cell([
    "fig_c = final_model.plot_components(forecast)\n",
    "fig_c.suptitle('Phân rã Prophet: Trend | Weekly | Yearly | Holiday', fontsize=13, fontweight='bold', y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig('03_Decomposition.png', dpi=150, bbox_inches='tight'); plt.show()"
]))

# VIII. ANOMALY DETECTION
cells.append(make_md_cell(["## VIII. ANOMALY DETECTION — CẢNH BÁO 3 CẤP (🟢🟡🔴)"]))
cells.append(make_code_cell([
    "model_95 = Prophet(\n",
    "    holidays=holidays_df,\n",
    "    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,\n",
    "    changepoint_prior_scale=best_params['changepoint_prior_scale'],\n",
    "    seasonality_prior_scale=best_params['seasonality_prior_scale'],\n",
    "    holidays_prior_scale=best_params['holidays_prior_scale'],\n",
    "    interval_width=0.95,\n",
    ")\n",
    "model_95.add_seasonality(name='monthly', period=30.5, fourier_order=5)\n",
    "model_95.fit(trainval_ts)\n",
    "forecast_95 = model_95.predict(future)\n\n",
    "def detect_anomaly(y, ylo_80, yhi_80, ylo_95, yhi_95):\n",
    "    if ylo_80 <= y <= yhi_80:   return '🟢 Bình thường'\n",
    "    if ylo_95 <= y <= yhi_95:   return '🟡 Cảnh báo (vượt CI 80%)'\n",
    "    return '🔴 BẤT THƯỜNG NGHIÊM TRỌNG (vượt CI 95%)'\n\n",
    "t = test_ts.merge(forecast[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='left')\n",
    "t = t.merge(forecast_95[['ds','yhat_lower','yhat_upper']], on='ds', how='left', suffixes=('_80','_95'))\n",
    "t['alert'] = t.apply(lambda r: detect_anomaly(\n",
    "    r['y'], r['yhat_lower_80'], r['yhat_upper_80'],\n",
    "    r['yhat_lower_95'], r['yhat_upper_95']), axis=1)\n\n",
    "print('=== ANOMALY DETECTION — TEST SET 08/2018 ===')\n",
    "print(t[['ds','y','yhat','alert']].to_string(index=False))\n",
    "print(f\"\\nTóm tắt: {t['alert'].value_counts().to_dict()}\")\n",
    "if (t['alert'].str.contains('NGHIÊM')).sum() >= 5:\n",
    "    print('\\n⚠️ ≥5 ngày bất thường nghiêm trọng → Cân nhắc Retrain!')"
]))

# IX. EXECUTIVE SUMMARY + EXPORT
cells.append(make_md_cell(["## IX. EXECUTIVE SUMMARY & EXPORT"]))
cells.append(make_code_cell([
    "summary = pd.DataFrame({\n",
    "    'Chỉ số': [\n",
    "        'GMV dự báo tháng 9/2018','GMV trung bình/ngày',\n",
    "        'Ngày đỉnh dự báo','Ngày đáy dự báo',\n",
    "        'Kịch bản lạc quan (Upper 80%)','Kịch bản bi quan (Lower 80%)',\n",
    "        'MAPE trên Test 08/2018','RMSE trên Test 08/2018',\n",
    "    ],\n",
    "    'Giá trị': [\n",
    "        f\"{f_sep['yhat'].sum():,.0f} BRL\",\n",
    "        f\"{f_sep['yhat'].mean():,.0f} BRL\",\n",
    "        f\"{f_sep.loc[f_sep.yhat.idxmax(),'ds'].date()} — {f_sep['yhat'].max():,.0f} BRL\",\n",
    "        f\"{f_sep.loc[f_sep.yhat.idxmin(),'ds'].date()} — {f_sep['yhat'].min():,.0f} BRL\",\n",
    "        f\"{f_sep['yhat_upper'].sum():,.0f} BRL/tháng\",\n",
    "        f\"{f_sep['yhat_lower'].sum():,.0f} BRL/tháng\",\n",
    "        f\"{mape_test*100:.2f}%\",\n",
    "        f\"{rmse_test:,.0f} BRL/ngày\",\n",
    "    ]\n",
    "})\n",
    "print('=== EXECUTIVE SUMMARY ===')\n",
    "print(summary.to_string(index=False))\n",
    "summary.to_csv('03_Executive_Summary.csv', index=False)\n\n",
    "with open('03_prophet_model.pkl','wb') as f: pickle.dump(final_model, f)\n",
    "forecast.to_csv('03_Forecast_Full.csv', index=False)\n",
    "f_sep.to_csv('03_Forecast_September.csv', index=False)\n\n",
    "print('\\n✅ FILE 03 HOÀN TẤT:')\n",
    "print('  03_prophet_model.pkl | 03_Forecast_Full.csv | 03_Forecast_September.csv')\n",
    "print('  03_Executive_Summary.csv | 03_Oracle_Forecast.png | 03_GridSearch_Heatmap.png')"
]))

nb = {
    "nbformat":4,"nbformat_minor":5,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.9.0"}},
    "cells":cells
}
with open('d:/Bao_cao_de_an/03_Sales_Forecasting.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"✅ 03_Sales_Forecasting.ipynb — {len(cells)} cells")

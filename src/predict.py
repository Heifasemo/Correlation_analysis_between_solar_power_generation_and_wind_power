# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import torch.nn as nn

# ====== 路径 ======
INPUT_FILE  = r"data/2024_Mech_Weather_Station_Data.xlsx"
OUTPUT_FILE = r"data/2024_Mech_Weather_Station_Data_predictions.csv"
MODEL_DIR   = "model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _tload(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ====== 读取训练配置 ======
with open(os.path.join(MODEL_DIR, "model_config.json"), "r", encoding="utf-8") as f:
    CFG = json.load(f)
FEATS_ALL = CFG["features_all"]
FEATS_SUB = CFG["features_sub"]
GHI_IDX   = FEATS_ALL.index('ghi')

# ====== 定义与训练一致的 RES-MLP ======
class DualBranch_ResMLP(nn.Module):
    def __init__(self, d_all: int, d_sub: int, ghi_index: int,
                 hidden_main: int, hidden_branch: int, hidden_ghi: int,
                 fused_hidden: int, extra_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.ghi_index = ghi_index
        hm, hb, hg = hidden_main, hidden_branch, hidden_ghi

        self.fc_in = nn.Linear(d_all, hm)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hm),
                nn.Linear(hm, hm),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hm, hm)
            ) for _ in range(extra_layers)
        ])
        self.ln_out = nn.LayerNorm(hm)

        self.b1 = nn.Linear(d_sub, hb)
        self.b2 = nn.Linear(hb, hb)

        self.g1 = nn.Linear(1, hg)
        self.g2 = nn.Linear(hg, hg)

        total = hm + hb + hg
        self.f1 = nn.Linear(total, fused_hidden)
        self.f2 = nn.Linear(fused_hidden, fused_hidden // 2)
        self.out = nn.Linear(fused_hidden // 2, 1)

    def forward(self, xa, xs):
        m = torch.relu(self.fc_in(xa))
        for block in self.res_blocks:
            m = m + block(m)
        m = self.ln_out(m)

        b = torch.relu(self.b1(xs))
        b = torch.relu(self.b2(b))

        ghi = xa[:, [self.ghi_index]]
        g = torch.relu(self.g1(ghi))
        g = torch.relu(self.g2(g))

        f = torch.cat([m, b, g], dim=1)
        f = torch.relu(self.f1(f))
        f = torch.relu(self.f2(f))
        return self.out(f).squeeze(-1)

# ====== 加载标定器/模型/融合权重 ======
sa   = joblib.load(os.path.join(MODEL_DIR, "sa.pkl"))
ss   = joblib.load(os.path.join(MODEL_DIR, "ss.pkl"))
lgbm = joblib.load(os.path.join(MODEL_DIR, "lgbm.pkl"))

mlp = DualBranch_ResMLP(
    d_all=len(FEATS_ALL), d_sub=len(FEATS_SUB), ghi_index=GHI_IDX,
    hidden_main=CFG["hidden_main"], hidden_branch=CFG["hidden_branch"],
    hidden_ghi=CFG["hidden_ghi"], fused_hidden=CFG["fused_hidden"],
    extra_layers=CFG["extra_layers"], dropout=CFG["dropout"]
).to(device)
mlp.load_state_dict(_tload(os.path.join(MODEL_DIR, "mlp.pth"), device))
mlp.eval()

centroid_path = os.path.join(MODEL_DIR, "centroid_weights.pkl")
centroid = joblib.load(centroid_path) if os.path.exists(centroid_path) else None

# ====== 列名标准化 & 特征衍生 ======
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'ghi': ['ghi','GHI','global_horizontal_irradiance'],
        'ave_temperature': ['ave_temperature','MODULE_TEMPERATURE','module_temperature','T_module'],
        'temperature_ambient': ['temperature_ambient','AMBIENT_TEMPERATURE','ambient_temperature','T_ambient'],
        'wind_speed': ['wind_speed','WIND_SPEED','WindSpeed'],
        'wind_direction': ['wind_direction','WIND_DIRECTION','WindDirection'],
        'WIND_SIN': ['WIND_SIN'],
        'WIND_COS': ['WIND_COS'],
        'power_2C2': ['power_2C2','OUTPUTPOWER','power','P_out']
    }
    df = df.copy()
    lower2orig = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in lower2orig: return lower2orig[n.lower()]
        return None
    for tgt, aliases in col_map.items():
        src = pick(aliases)
        if src and src != tgt:
            df.rename(columns={src: tgt}, inplace=True)

    if ('WIND_SIN' not in df.columns or 'WIND_COS' not in df.columns) and ('wind_direction' in df.columns):
        rad = np.deg2rad(df['wind_direction'].astype(float))
        df['WIND_SIN'] = np.sin(rad)
        df['WIND_COS'] = np.cos(rad)
    if 'WIND_SIN' not in df.columns: df['WIND_SIN'] = 0.0
    if 'WIND_COS' not in df.columns: df['WIND_COS'] = 1.0
    return df

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ghi' not in df.columns:
        raise ValueError("缺少 'ghi' 列，无法派生衍生特征。")
    df['ghi_sq']   = df['ghi'] ** 2
    df['ghi_cu']   = df['ghi'] ** 3
    df['ghi_sqrt'] = np.sqrt(np.clip(df['ghi'], 0, None))
    df['ghi_log']  = np.log1p(np.clip(df['ghi'], 0, None))
    return df

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx','.xls'):
        return pd.read_excel(path, engine='openpyxl')
    if ext == '.csv':
        return pd.read_csv(path)
    try:
        return pd.read_excel(path, engine='openpyxl')
    except Exception:
        return pd.read_csv(path)

# ====== softmax 融合 ======
def fuse_softmax(pred_mlp, pred_gbm, ghi_raw, weights_dict):
    y_global = y_local = None
    if not weights_dict or 'global' not in weights_dict:
        return y_global, y_local
    wg = np.asarray(weights_dict['global']).astype(float)  # [w_mlp, w_gbm]
    y_global = wg[0]*pred_mlp + wg[1]*pred_gbm
    if 'bins' in weights_dict and 'local' in weights_dict:
        edges = np.asarray(weights_dict['bins'])
        local = weights_dict['local']
        ids = np.digitize(ghi_raw, edges, right=False)
        y_local = np.empty_like(y_global)
        for i in range(len(y_global)):
            w = np.asarray(local.get(int(ids[i]), wg)).astype(float)
            y_local[i] = w[0]*pred_mlp[i] + w[1]*pred_gbm[i]
    return y_global, y_local

# ====== 预测 ======
def predict_file(input_path, output_path):
    df0 = read_any(input_path)
    df  = derive_features(normalize_columns(df0))

    Xa = df[FEATS_ALL].values.astype(np.float32)
    Xs = df[FEATS_SUB].values.astype(np.float32)

    Xa_s = sa.transform(Xa)
    Xs_s = ss.transform(Xs)

    with torch.no_grad():
        ta, ts = torch.tensor(Xa_s, device=device), torch.tensor(Xs_s, device=device)
        pred_mlp = mlp(ta, ts).cpu().numpy().astype(float)

    df_all_s = pd.DataFrame(Xa_s, columns=FEATS_ALL)
    pred_gbm = lgbm.predict(df_all_s).astype(float)

    ghi_raw = df['ghi'].values.astype(float)
    y_soft, y_soft_local = fuse_softmax(pred_mlp, pred_gbm, ghi_raw, centroid)

    out = df.copy()
    out['pred_mlp'] = pred_mlp
    out['pred_gbm'] = pred_gbm
    if y_soft is not None:        out['pred_soft'] = y_soft
    if y_soft_local is not None:  out['pred_soft_local'] = y_soft_local

    out.to_csv(output_path, index=False, float_format='%.6f')

    if centroid is None:
        print("⚠️ 未找到 centroid_weights.pkl，仅保存 pred_mlp / pred_gbm。")
    else:
        has_local = ('bins' in centroid and 'local' in centroid)
        print("✅ 使用 softmax 融合（全局{}）。".format(" + 局部(GHI分箱)" if has_local else ""))
        print("   Global weights (MLP, GBM) =",
              np.asarray(centroid['global']).round(4).tolist())
    print("✅ 预测结果已保存：", output_path)

if __name__ == "__main__":
    predict_file(INPUT_FILE, OUTPUT_FILE)


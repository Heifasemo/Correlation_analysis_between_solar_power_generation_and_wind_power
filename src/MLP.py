# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader

# ─── 固定随机种子 & 设备 ─────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 配置 ─────────────────────────────────────────────
class Config:
    # 训练特征列表（顺序即“训练时顺序”，预测端会复用）
    features_all = [
        'ave_temperature', 'ghi', 'temperature_ambient',
        'wind_speed', 'WIND_SIN', 'WIND_COS',
        'ghi_sq', 'ghi_cu', 'ghi_sqrt', 'ghi_log'
    ]
    features_sub = ['wind_speed', 'WIND_SIN', 'WIND_COS']
    label_col    = 'power_2C2'

    hidden_main   = 512
    hidden_branch = 256
    hidden_ghi    = 256
    fused_hidden  = (hidden_main + hidden_branch + hidden_ghi) // 2
    extra_layers  = 3
    dropout       = 0.3

    nn_epochs = 500
    nn_lr     = 1e-3
    nn_bs     = 128
    patience  = 50

    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 62,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 64,
        'lambda_l2': 2.0,
        'verbose': -1,
        'seed': 42,
        'n_estimators': 5000
    }
    lgbm_es = 100

    # 分区重心（局部 softmax，按 GHI 分箱）
    ghi_bins = [0, 100, 200, 300, 400, 500, 600, 700,
                800, 900, 1000, 1100, 1200]
    min_bin_count = 80

# ─── RES-MLP（残差 + LayerNorm + 风/GHI支路） ─────────────────────────────
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

# ─── 数据加载 & 特征衍生 ─────────────────────────────
def load_and_derive(path: str):
    df = pd.read_csv(path)
    df['ghi_sq']   = df['ghi'] ** 2
    df['ghi_cu']   = df['ghi'] ** 3
    df['ghi_sqrt'] = np.sqrt(np.clip(df['ghi'], 0, None))
    df['ghi_log']  = np.log1p(np.clip(df['ghi'], 0, None))
    X_all = df[Config.features_all].values.astype(np.float32)
    X_sub = df[Config.features_sub].values.astype(np.float32)
    y     = df[Config.label_col].values.astype(np.float32)
    return X_all, X_sub, y, df

# ─── 质心/softmax 融合权重（在验证集上学习） ─────────────────────────────
def fit_softmax_weights(y_val, preds_list, loss_type="huber", max_steps=3000, lr=1e-2):
    P = np.vstack(preds_list).T.astype(np.float32)   # (N, K)
    yt = torch.tensor(y_val.astype(np.float32), device=device)
    Pt = torch.tensor(P, device=device)

    logits = torch.zeros(P.shape[1], device=device, requires_grad=True)  # K
    opt = optim.Adam([logits], lr=lr)

    for _ in range(max_steps):
        w = F.softmax(logits, dim=0)                 # w>=0, sum=1
        y_hat = Pt @ w
        if loss_type == "huber":
            loss = torch.mean(F.huber_loss(y_hat, yt, delta=1.0))
        else:
            loss = torch.mean((y_hat - yt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        w_final = F.softmax(logits, dim=0).detach().cpu().numpy()
    return w_final  # (K,)

# ─── 训练/验证/测试 & 保存 ─────────────────────────────
def train_and_eval(train_csv: str, test_csv: str):
    # 1) 加载
    Xa, Xs, y, _ = load_and_derive(train_csv)

    # 2) 切分
    Xa_tr, Xa_va, Xs_tr, Xs_va, y_tr, y_va = train_test_split(
        Xa, Xs, y, test_size=0.2, random_state=42
    )

    # 3) 标准化
    sa = StandardScaler().fit(Xa_tr)
    ss = StandardScaler().fit(Xs_tr)
    Xa_tr_s, Xa_va_s = sa.transform(Xa_tr), sa.transform(Xa_va)
    Xs_tr_s, Xs_va_s = ss.transform(Xs_tr), ss.transform(Xs_va)

    # 4) 训练 RES-MLP（早停看 R²）
    ghi_index = Config.features_all.index('ghi')
    model = DualBranch_ResMLP(
        d_all=Xa_tr_s.shape[1], d_sub=Xs_tr_s.shape[1], ghi_index=ghi_index,
        hidden_main=Config.hidden_main, hidden_branch=Config.hidden_branch,
        hidden_ghi=Config.hidden_ghi, fused_hidden=Config.fused_hidden,
        extra_layers=Config.extra_layers, dropout=Config.dropout
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=Config.nn_lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.nn_epochs, eta_min=1e-6)
    tr_ds = TensorDataset(torch.tensor(Xa_tr_s), torch.tensor(Xs_tr_s), torch.tensor(y_tr))
    tr_ld = DataLoader(tr_ds, batch_size=Config.nn_bs, shuffle=True)

    best_r2, cnt, best_state = -np.inf, 0, None
    for ep in range(1, Config.nn_epochs+1):
        model.train()
        for xb, xsb, yb in tr_ld:
            xb, xsb, yb = xb.to(device), xsb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb, xsb), yb)
            loss.backward(); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(torch.tensor(Xa_va_s).to(device),
                            torch.tensor(Xs_va_s).to(device)).cpu().numpy()
        r2v = r2_score(y_va, pred_va)
        print(f"MLP Ep{ep}/{Config.nn_epochs} Val R²={r2v:.4f}")
        if r2v > best_r2 + 1e-4:
            best_r2, cnt, best_state = r2v, 0, model.state_dict()
        else:
            cnt += 1
            if cnt >= Config.patience:
                print(f"早停 at Ep{ep} (best R²={best_r2:.4f})")
                break

    # 5) 保存 MLP & 标准化器
    os.makedirs('model', exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, 'model/mlp.pth')
    joblib.dump(sa, 'model/sa.pkl')
    joblib.dump(ss, 'model/ss.pkl')

    # 6) 训练 LightGBM
    df_tr = pd.DataFrame(Xa_tr_s, columns=Config.features_all)
    df_va = pd.DataFrame(Xa_va_s, columns=Config.features_all)
    lgbm = lgb.LGBMRegressor(**Config.lgbm_params)
    lgbm.fit(
        df_tr, y_tr,
        eval_set=[(df_va, y_va)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=Config.lgbm_es),
                   lgb.log_evaluation(period=50)]
    )
    joblib.dump(lgbm, 'model/lgbm.pkl')

    # 7) 在验证集上学习 softmax 权重（全局 + 局部）
    model.eval()
    with torch.no_grad():
        val_pred_nn  = model(torch.tensor(Xa_va_s).to(device),
                             torch.tensor(Xs_va_s).to(device)).cpu().numpy()
    val_pred_gbm = lgbm.predict(df_va)

    w_global = fit_softmax_weights(y_va, [val_pred_nn, val_pred_gbm], loss_type="huber")
    print("Global softmax weights (MLP, GBM) =", np.round(w_global, 4))

    val_GHI = Xa_va[:, ghi_index]  # 用未标准化 GHI 分箱
    bin_ids = np.digitize(val_GHI, Config.ghi_bins, right=False)

    local_weights = {}
    for b in np.unique(bin_ids):
        idx = (bin_ids == b)
        if idx.sum() >= Config.min_bin_count:
            w_b = fit_softmax_weights(y_va[idx], [val_pred_nn[idx], val_pred_gbm[idx]],
                                      loss_type="huber")
            local_weights[int(b)] = w_b

    joblib.dump({'global': w_global, 'bins': Config.ghi_bins, 'local': local_weights},
                'model/centroid_weights.pkl')

    # 8) 在测试集上输出对比 CSV
    Xte_a, Xte_s, y_te, df_out = load_and_derive(test_csv)
    Xte_a_s, Xte_s_s = sa.transform(Xte_a), ss.transform(Xte_s)
    with torch.no_grad():
        test_pred_nn  = model(torch.tensor(Xte_a_s).to(device),
                              torch.tensor(Xte_s_s).to(device)).cpu().numpy()
    df_te = pd.DataFrame(Xte_a_s, columns=Config.features_all)
    test_pred_gbm = lgbm.predict(df_te)

    pred_soft = w_global[0] * test_pred_nn + w_global[1] * test_pred_gbm

    test_GHI = Xte_a[:, ghi_index]
    test_bin = np.digitize(test_GHI, Config.ghi_bins, right=False)
    pred_soft_local = np.empty_like(pred_soft)
    for i in range(len(pred_soft)):
        b = int(test_bin[i])
        w_use = local_weights.get(b, w_global)
        pred_soft_local[i] = w_use[0]*test_pred_nn[i] + w_use[1]*test_pred_gbm[i]

    df_csv = pd.read_csv(test_csv)
    df_csv['pred_mlp']        = test_pred_nn
    df_csv['pred_gbm']        = test_pred_gbm
    df_csv['pred_soft']       = pred_soft
    df_csv['pred_soft_local'] = pred_soft_local

    out_path = os.path.splitext(test_csv)[0] + '_compare_MLP_centroid.csv'
    df_csv.to_csv(out_path, index=False, float_format='%.6f')
    print(f"对比结果保存到：{out_path}")
    print("列包含：pred_mlp, pred_gbm, pred_soft(全局), pred_soft_local(分区)")

    # 9) 落盘训练时配置（预测端严格复用）
    cfg = {
        "features_all": Config.features_all,
        "features_sub": Config.features_sub,
        "label_col": Config.label_col,
        "hidden_main": Config.hidden_main,
        "hidden_branch": Config.hidden_branch,
        "hidden_ghi": Config.hidden_ghi,
        "fused_hidden": Config.fused_hidden,
        "extra_layers": Config.extra_layers,
        "dropout": Config.dropout,
        "ghi_bins": Config.ghi_bins
    }
    with open(os.path.join('model', 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("✅ 已保存模型配置到 model/model_config.json")

if __name__ == '__main__':
    # 训练/测试数据路径可按需修改
    train_and_eval('data/train_split.csv', 'data/test_split.csv')

# =====================================================
# VAEjMLP latent-SHAP + 稳定性 + SHAP可视化 + Cox验证 (Streamlit)
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

import shap
import matplotlib.pyplot as plt


# =====================================================
# Utils
# =====================================================
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让结果更可复现（可能略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@st.cache_data(show_spinner=False)
def read_csv_cached(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


# =====================================================
# 模型定义（与你原始一致）
# =====================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, latent_dim * 2)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        mean, log_var = torch.chunk(h3, 2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var


class MLP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))


# =====================================================
# Streamlit 页面
# =====================================================
st.set_page_config(page_title="VAEjMLP latent-SHAP", layout="wide")
st.title("🧬 VAEjMLP + latent SHAP 生物标志物分析（完整整合版）")

with st.sidebar:
    st.header("参数设置")

    latent_dim = st.number_input("latent_dim", min_value=4, max_value=1024, value=128, step=4)
    n_epochs = st.number_input("训练轮数 epochs", min_value=10, max_value=2000, value=100, step=10)
    lr = st.number_input("学习率 lr", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")

    ce_weight = st.number_input("CE 权重（loss = KL + ce_weight*CE）", min_value=0.0, max_value=10.0, value=0.001, step=0.001)

    test_size = st.slider("test_size", 0.05, 0.5, 0.2, 0.05)

    st.divider()
    st.subheader("稳定性 (multi-run)")

    n_runs = st.slider("重复运行次数 n_runs", 1, 50, 10)
    top_k = st.slider("TopK 频率统计", 5, 300, 20)
    seed_base = st.number_input("seed_base", value=42, step=1)

    st.divider()
    st.subheader("SHAP 计算")

    background_n = st.slider("background 样本数", 10, 200, 50)
    shap_nsamples = st.slider("KernelExplainer nsamples", 50, 500, 100, 50)

    st.caption("提示：KernelExplainer 可能较慢；n_runs 大时建议降低 nsamples 或 background。")

    st.divider()
    st.subheader("Cox 生存验证（可选）")

    use_survival = st.checkbox("启用 Cox 生存验证", value=False)
    freq_thr = st.slider("稳定基因筛选：TopK_Freq ≥", 0.0, 1.0, 0.6, 0.05)
    cv_thr = st.number_input("稳定基因筛选：CV ≤", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    max_surv_genes = st.slider("最多用于 Cox 的基因数", 5, 300, 50)
    cox_penalizer = st.number_input("Cox L2 penalizer", min_value=0.0, max_value=10.0, value=0.1, step=0.1)

st.divider()

rna_file = st.file_uploader("上传 RNA-seq 表达矩阵（genes × samples），CSV，index=Gene，columns=Sample", type="csv")
label_file = st.file_uploader("上传 Label 文件（Sample, Label），CSV", type="csv")
surv_file = st.file_uploader("（可选）上传 生存数据（Sample, Time, Event），CSV", type="csv") if use_survival else None

run_button = st.button("🚀 运行模型")

# =====================================================
# 主流程
# =====================================================
if run_button:
    if (rna_file is None) or (label_file is None):
        st.error("请先上传 RNA 表达矩阵和 Label 文件。")
        st.stop()

    with st.spinner("读取数据中..."):
        rna = read_csv_cached(rna_file)
        labels = read_csv_cached(label_file)

    # ---- 数据校验与对齐 ----
    if rna.shape[0] < 2 or rna.shape[1] < 4:
        st.error("RNA 矩阵维度看起来不对：需要 genes×samples 且样本数至少 4。")
        st.stop()

    if "Sample" not in labels.columns or "Label" not in labels.columns:
        st.error("Label 文件必须包含列：Sample, Label")
        st.stop()

    # 默认：rna 第一列为 gene index（如果用户没设置 index_col）
    if "Unnamed: 0" in rna.columns:
        rna = rna.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")

    # 对齐样本
    samples_rna = list(rna.columns)
    samples_lab = labels["Sample"].astype(str).tolist()
    rna.columns = rna.columns.astype(str)
    labels["Sample"] = labels["Sample"].astype(str)

    if set(samples_rna) != set(samples_lab):
        st.warning("RNA 样本与 Label 样本集合不完全一致，将取交集对齐。")
        common = sorted(list(set(samples_rna).intersection(set(samples_lab))))
        if len(common) < 4:
            st.error("对齐后共同样本数太少（<4），无法训练。")
            st.stop()
        rna = rna[common]
        labels = labels.set_index("Sample").loc[common].reset_index()

    # 重新校验顺序一致
    labels = labels.set_index("Sample").loc[rna.columns].reset_index()

    genes = rna.index.astype(str).tolist()
    y = labels["Label"].values.astype(int)

    # X: samples x genes
    X = MinMaxScaler().fit_transform(rna.T.values)

    # =====================================================
    # 多次运行：收集 metrics / gene_importance / shap_z（仅最后一次用于画图）
    # =====================================================
    all_importances = []     # list[pd.Series] index=genes
    topk_lists = []          # list[list[str]]
    metrics_runs = []        # list[dict]
    last_shap_z = None
    last_z_test = None

    # 如果用户跑很多次：给个进度条
    prog = st.progress(0)
    status = st.empty()

    for run_i in range(int(n_runs)):
        seed = int(seed_base + run_i)
        set_seed(seed)

        status.write(f"Run {run_i+1}/{n_runs} | seed={seed}")

        # ---- split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=seed, stratify=y if len(np.unique(y)) == 2 else None
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        # ---- 模型 ----
        vae = VAE(X.shape[1], int(latent_dim))
        mlp = MLP(int(latent_dim))
        optimizer = optim.Adam(list(vae.parameters()) + list(mlp.parameters()), lr=float(lr))

        # ---- 训练 ----
        vae.train()
        mlp.train()
        for _ in range(int(n_epochs)):
            optimizer.zero_grad()
            z, mean, log_var = vae(X_train_t)
            y_pred = mlp(z)
            kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            ce = F.binary_cross_entropy(y_pred, y_train_t, reduction="sum")
            loss = kl + float(ce_weight) * ce
            loss.backward()
            optimizer.step()

        vae.eval()
        mlp.eval()

        # ---- 预测与指标 ----
        with torch.no_grad():
            z_test, _, _ = vae(X_test_t)
            y_pred_test = mlp(z_test).cpu().numpy().flatten()

        # 一些数据集可能导致 AUC 报错（测试集中只有一个类）
        try:
            auc = roc_auc_score(y_test, y_pred_test)
        except Exception:
            auc = np.nan

        y_hat = (y_pred_test > 0.5).astype(int)

        metrics_runs.append({
            "run": run_i,
            "seed": seed,
            "AUC": auc,
            "Accuracy": accuracy_score(y_test, y_hat),
            "Precision": precision_score(y_test, y_hat, zero_division=0),
            "Recall": recall_score(y_test, y_hat, zero_division=0),
        })

        # =====================================================
        # latent SHAP（KernelExplainer）
        # =====================================================
        with torch.no_grad():
            z_train, _, _ = vae(X_train_t)
        z_train_np = z_train.cpu().numpy()
        z_test_np = z_test.cpu().numpy()


        def mlp_predict(z_numpy):
            z_t = torch.tensor(z_numpy, dtype=torch.float32)
            with torch.no_grad():
                out = mlp(z_t).cpu().numpy()
            return out.reshape(-1)  # <-- 关键：返回 1D

        bg_n = int(min(background_n, z_train_np.shape[0]))
        background_z = shap.sample(z_train_np, bg_n)

        explainer = shap.KernelExplainer(mlp_predict, background_z)

        shap_values = explainer.shap_values(z_test_np, nsamples=int(shap_nsamples))

        # --- 兼容不同 shap 版本的返回类型/形状 ---
        if isinstance(shap_values, list):
            shap_z = shap_values[0]
        else:
            shap_z = shap_values

        # 如果是 (n, d, 1) → squeeze 成 (n, d)
        shap_z = np.array(shap_z)
        if shap_z.ndim == 3 and shap_z.shape[-1] == 1:
            shap_z = shap_z[:, :, 0]

        # 最终保证是二维 (n_samples, n_features)
        if shap_z.ndim != 2:
            raise ValueError(f"Unexpected shap_z shape: {shap_z.shape}")

        # 保险校验，便于你定位
        if shap_z.shape[0] != z_test_np.shape[0] or shap_z.shape[1] != z_test_np.shape[1]:
            raise ValueError(
                f"Shape mismatch: shap_z={shap_z.shape}, z_test={z_test_np.shape}. "
                "Check mlp_predict output shape and shap_values processing."
            )

        # =====================================================
        # latent → gene 映射（保持与你原逻辑一致）
        # =====================================================
        W_gene_hidden = vae.fc1.weight.detach().cpu().numpy()  # (1024, n_genes)
        abs_shap_z = np.mean(np.abs(shap_z), axis=0)           # (latent_dim,)

        # 这里你的原式：只差在 W_fc1 的列强弱，latent shap 只是全局缩放
        # 先保持一致，保证复现；后续你需要更论文级映射我可以再升级
        gene_importance = {}
        scale = float(np.sum(abs_shap_z))
        for i, gene in enumerate(genes):
            gene_importance[gene] = float(np.mean(np.abs(W_gene_hidden[:, i])) * scale)

        imp_s = pd.Series(gene_importance).reindex(genes)
        all_importances.append(imp_s)

        topk_lists.append(imp_s.sort_values(ascending=False).head(int(top_k)).index.tolist())

        # 保存最后一次 run 的 shap 用于画图
        if run_i == int(n_runs) - 1:
            last_shap_z = shap_z
            last_z_test = z_test_np

        prog.progress((run_i + 1) / int(n_runs))

    status.empty()
    prog.empty()

    # =====================================================
    # 汇总输出：模型性能（多次run）
    # =====================================================
    metrics_df = pd.DataFrame(metrics_runs)
    st.success("运行完成！")

    st.subheader("📊 模型性能（每次 run）")
    st.dataframe(metrics_df)

    st.subheader("📊 模型性能汇总（均值±标准差）")
    summary = metrics_df[["AUC", "Accuracy", "Precision", "Recall"]].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["Metric", "Mean", "Std"]
    st.dataframe(summary)

    st.download_button(
        "⬇ 下载所有 run 的模型指标",
        metrics_df.to_csv(index=False),
        "model_metrics_all_runs.csv",
    )

    # =====================================================
    # 汇总输出：基因重要性稳定性（Mean / CV / Frequency）
    # =====================================================
    imp_mat = pd.concat(all_importances, axis=1)
    imp_mat.columns = [f"run_{i}" for i in range(int(n_runs))]

    mean_imp = imp_mat.mean(axis=1)
    std_imp = imp_mat.std(axis=1)
    cv_imp = std_imp / (mean_imp.abs() + 1e-12)

    from collections import Counter
    freq_counter = Counter([g for lst in topk_lists for g in lst])
    freq = pd.Series({g: freq_counter.get(g, 0) / float(n_runs) for g in genes})

    stability_df = pd.DataFrame({
        "Gene": genes,
        "MeanImportance": mean_imp.values,
        "StdImportance": std_imp.values,
        "CV": cv_imp.values,
        f"Top{int(top_k)}_Freq": freq.values,
    }).sort_values([f"Top{int(top_k)}_Freq", "MeanImportance"], ascending=[False, False])

    st.subheader("📌 生物标志物稳定性（Frequency / CV）")
    st.dataframe(stability_df.head(50))

    st.download_button(
        "⬇ 下载稳定性统计表",
        stability_df.to_csv(index=False),
        "biomarker_stability.csv",
    )

    # 兼容你原来的“Top 20 biomarkers”输出：用稳定性均值排名
    st.subheader("🧬 Top 20 潜在生物标志物（MeanImportance 排名）")
    st.dataframe(stability_df.sort_values("MeanImportance", ascending=False).head(20))

    st.download_button(
        "⬇ 下载全部基因重要性（Mean/CV/Freq）",
        stability_df.to_csv(index=False),
        "latent_shap_gene_importance_stability.csv",
    )

    # =====================================================
    # SHAP 可视化（使用最后一次 run 的 shap_z）
    # =====================================================
    if last_shap_z is not None and last_z_test is not None:
        st.subheader("🔍 Latent SHAP Summary（dot）")

        last_shap_z = np.array(last_shap_z)
        if last_shap_z.ndim == 3 and last_shap_z.shape[-1] == 1:
            last_shap_z = last_shap_z[:, :, 0]

        fig1 = plt.figure()
        shap.summary_plot(last_shap_z, features=last_z_test, show=False)
        st.pyplot(fig1)

        st.subheader("📊 Latent SHAP Summary（bar）")
        fig2 = plt.figure()
        shap.summary_plot(last_shap_z, features=last_z_test, plot_type="bar", show=False)
        st.pyplot(fig2)

        abs_latent = np.mean(np.abs(last_shap_z), axis=0)
        latent_df = pd.DataFrame({"LatentDim": np.arange(len(abs_latent)), "MeanAbsSHAP": abs_latent})
        latent_df = latent_df.sort_values("MeanAbsSHAP", ascending=False)

        st.subheader("📈 Top 20 Latent 维度重要性（MeanAbsSHAP）")
        st.dataframe(latent_df.head(20))

        fig3 = plt.figure()
        plt.bar(latent_df.head(20)["LatentDim"].astype(str), latent_df.head(20)["MeanAbsSHAP"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig3)

        st.download_button(
            "⬇ 下载 latent 维度 MeanAbsSHAP",
            latent_df.to_csv(index=False),
            "latent_mean_abs_shap.csv",
        )

    # =====================================================
    # Cox 生存验证（可选）
    # =====================================================
    if use_survival:
        if surv_file is None:
            st.warning("你开启了 Cox 验证，但没有上传生存数据文件。")
        else:
            try:
                from lifelines import CoxPHFitter
                from lifelines.utils import concordance_index
            except Exception:
                st.error("未检测到 lifelines。请先安装：pip install lifelines")
                st.stop()

            surv = read_csv_cached(surv_file)

            needed = {"Sample", "Time", "Event"}
            if not needed.issubset(set(surv.columns)):
                st.error("生存数据必须包含列：Sample, Time, Event")
                st.stop()

            surv["Sample"] = surv["Sample"].astype(str)
            surv = surv.set_index("Sample")

            # 对齐到 RNA 样本
            common = list(set(rna.columns.astype(str)).intersection(set(surv.index.astype(str))))
            if len(common) < 10:
                st.error("生存数据与 RNA 的共同样本太少（<10），无法做 Cox 验证。")
                st.stop()

            common = [s for s in rna.columns.astype(str) if s in common]  # 保持 RNA 顺序
            surv_aligned = surv.loc[common]
            rna_aligned = rna[common]

            # 选稳定基因
            freq_col = f"Top{int(top_k)}_Freq"
            selected = stability_df[
                (stability_df[freq_col] >= float(freq_thr)) & (stability_df["CV"] <= float(cv_thr))
            ].sort_values(["MeanImportance"], ascending=False)["Gene"].head(int(max_surv_genes)).tolist()

            st.subheader("⏱ Cox 生存验证")
            st.write({
                "共同样本数": int(len(common)),
                "筛选阈值": f"{freq_col}≥{freq_thr}, CV≤{cv_thr}",
                "进入 Cox 的基因数": int(len(selected)),
            })

            if len(selected) < 2:
                st.warning("筛选后基因太少（<2）。请放宽阈值或增大 top_k / n_runs。")
            else:
                X_surv = rna_aligned.loc[selected].T  # samples x genes

                df_cox = pd.concat([surv_aligned[["Time", "Event"]], X_surv], axis=1).dropna()
                if df_cox.shape[0] < 10:
                    st.warning("去除缺失值后样本太少，无法稳定拟合。")
                else:
                    df_train, df_test = train_test_split(df_cox, test_size=float(test_size), random_state=int(seed_base))

                    cph = CoxPHFitter(penalizer=float(cox_penalizer))
                    cph.fit(df_train, duration_col="Time", event_col="Event")

                    risk = cph.predict_partial_hazard(df_test)
                    c_index = concordance_index(df_test["Time"], -risk.values, df_test["Event"])

                    st.write({"C-index": float(c_index)})

                    st.subheader("📌 Cox 回归系数 Top 20")
                    coef_df = cph.summary.reset_index().rename(columns={"index": "Feature"})
                    coef_df = coef_df.sort_values("coef", ascending=False)
                    st.dataframe(coef_df.head(20))

                    st.download_button(
                        "⬇ 下载 Cox summary",
                        cph.summary.to_csv(),
                        "cox_summary.csv",
                    )

                    # 风险分数导出
                    out_risk = pd.DataFrame({
                        "Sample": df_test.index.astype(str),
                        "Time": df_test["Time"].values,
                        "Event": df_test["Event"].values,
                        "RiskScore": risk.values.flatten()
                    })
                    st.download_button(
                        "⬇ 下载测试集风险分数（RiskScore）",
                        out_risk.to_csv(index=False),
                        "cox_test_risk_scores.csv",
                    )

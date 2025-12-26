# =====================================================
# VAEjMLP latent-SHAP + 稳定性 + SHAP可视化 + Cox验证 (Streamlit) —— 已修好版
# 关键修复：
# 1) download_button 触发 rerun 后结果不消失：全部结果缓存到 st.session_state
# 2) SHAP 返回形状兼容处理：避免 summary_plot 形状断言报错
# 3) lifelines 不存在时：自动跳过 Cox，不中断主流程
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


def ensure_2d_shap(shap_values, features_2d: np.ndarray) -> np.ndarray:
    """
    兼容 shap 的多种返回格式，最终保证输出 (n_samples, n_features)
    """
    # shap_values 可能是 list 或 ndarray
    if isinstance(shap_values, list):
        shap_z = shap_values[0]
    else:
        shap_z = shap_values

    shap_z = np.array(shap_z)

    # 可能是 (n, d, 1)
    if shap_z.ndim == 3 and shap_z.shape[-1] == 1:
        shap_z = shap_z[:, :, 0]

    # 有些版本可能 (1, n, d) 或其它奇怪形状，做兜底
    if shap_z.ndim == 3 and shap_z.shape[0] == 1:
        shap_z = shap_z[0]

    if shap_z.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_z.shape}")

    if shap_z.shape[0] != features_2d.shape[0] or shap_z.shape[1] != features_2d.shape[1]:
        raise ValueError(
            f"Shape mismatch: shap={shap_z.shape}, features={features_2d.shape}. "
            "Please check mlp_predict output shape and SHAP processing."
        )
    return shap_z


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


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
st.title("🧬 VAEjMLP + latent SHAP 生物标志物分析（完整整合版｜已修好）")

# ===== 顶部工具按钮：清空缓存 =====
with st.expander("🧰 工具", expanded=False):
    if st.button("🧹 清空缓存结果（不会清空上传文件）"):
        for k in list(st.session_state.keys()):
            if k.startswith("cache_"):
                st.session_state.pop(k, None)
        st.rerun()

with st.sidebar:
    st.header("参数设置")

    latent_dim = st.number_input("latent_dim", min_value=4, max_value=1024, value=128, step=4)
    n_epochs = st.number_input("训练轮数 epochs", min_value=10, max_value=2000, value=100, step=10)
    lr = st.number_input("学习率 lr", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")

    ce_weight = st.number_input(
        "CE 权重（loss = KL + ce_weight*CE）",
        min_value=0.0,
        max_value=10.0,
        value=0.001,
        step=0.001,
    )

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
# 运行主流程
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
    samples_rna = list(map(str, rna.columns.tolist()))
    rna.columns = rna.columns.astype(str)
    labels["Sample"] = labels["Sample"].astype(str)

    if set(samples_rna) != set(labels["Sample"].tolist()):
        st.warning("RNA 样本与 Label 样本集合不完全一致，将取交集对齐。")
        common = sorted(list(set(samples_rna).intersection(set(labels["Sample"].tolist()))))
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
    all_importances = []  # list[pd.Series] index=genes
    topk_lists = []       # list[list[str]]
    metrics_runs = []     # list[dict]
    last_shap_z = None
    last_z_test = None
    last_latent_df = None

    prog = st.progress(0)
    status = st.empty()

    for run_i in range(int(n_runs)):
        seed = int(seed_base + run_i)
        set_seed(seed)

        status.write(f"Run {run_i+1}/{n_runs} | seed={seed}")

        # ---- split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=seed,
            stratify=y if len(np.unique(y)) == 2 else None,
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

        try:
            auc = roc_auc_score(y_test, y_pred_test)
        except Exception:
            auc = np.nan

        y_hat = (y_pred_test > 0.5).astype(int)

        metrics_runs.append(
            {
                "run": run_i,
                "seed": seed,
                "AUC": auc,
                "Accuracy": accuracy_score(y_test, y_hat),
                "Precision": precision_score(y_test, y_hat, zero_division=0),
                "Recall": recall_score(y_test, y_hat, zero_division=0),
            }
        )

        # =====================================================
        # latent SHAP（KernelExplainer）
        # =====================================================
        with torch.no_grad():
            z_train, _, _ = vae(X_train_t)

        z_train_np = z_train.cpu().numpy()
        z_test_np = z_test.cpu().numpy()

        # ★ 建议返回 1D，减少 shap 输出歧义
        def mlp_predict(z_numpy):
            z_t = torch.tensor(z_numpy, dtype=torch.float32)
            with torch.no_grad():
                out = mlp(z_t).cpu().numpy()
            return out.reshape(-1)  # (n,)

        bg_n = int(min(background_n, z_train_np.shape[0]))
        background_z = shap.sample(z_train_np, bg_n)

        explainer = shap.KernelExplainer(mlp_predict, background_z)
        shap_values = explainer.shap_values(z_test_np, nsamples=int(shap_nsamples))
        shap_z = ensure_2d_shap(shap_values, z_test_np)  # (n_test, latent_dim)

        # =====================================================
        # latent → gene 映射（保持与你原逻辑一致）
        # =====================================================
        W_gene_hidden = vae.fc1.weight.detach().cpu().numpy()  # (1024, n_genes)
        abs_shap_z = np.mean(np.abs(shap_z), axis=0)          # (latent_dim,)

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

            abs_latent = np.mean(np.abs(shap_z), axis=0)
            last_latent_df = (
                pd.DataFrame({"LatentDim": np.arange(len(abs_latent)), "MeanAbsSHAP": abs_latent})
                .sort_values("MeanAbsSHAP", ascending=False)
                .reset_index(drop=True)
            )

        prog.progress((run_i + 1) / int(n_runs))

    status.empty()
    prog.empty()

    # =====================================================
    # 汇总输出：模型性能（多次 run）
    # =====================================================
    metrics_df = pd.DataFrame(metrics_runs)
    summary_df = metrics_df[["AUC", "Accuracy", "Precision", "Recall"]].agg(["mean", "std"]).T.reset_index()
    summary_df.columns = ["Metric", "Mean", "Std"]

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

    freq_col = f"Top{int(top_k)}_Freq"
    stability_df = (
        pd.DataFrame(
            {
                "Gene": genes,
                "MeanImportance": mean_imp.values,
                "StdImportance": std_imp.values,
                "CV": cv_imp.values,
                freq_col: freq.values,
            }
        )
        .sort_values([freq_col, "MeanImportance"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # =====================================================
    # 把结果缓存到 session_state（download 触发 rerun 也不丢）
    # =====================================================
    st.session_state["cache_metrics_df"] = metrics_df
    st.session_state["cache_summary_df"] = summary_df
    st.session_state["cache_stability_df"] = stability_df
    st.session_state["cache_latent_df"] = last_latent_df

    st.session_state["cache_csv_metrics_all"] = to_csv_bytes(metrics_df)
    st.session_state["cache_csv_summary"] = to_csv_bytes(summary_df)
    st.session_state["cache_csv_stability"] = to_csv_bytes(stability_df)
    st.session_state["cache_csv_latent"] = to_csv_bytes(last_latent_df) if last_latent_df is not None else None

    # 也把图需要的数组存起来（如果你不想存太大数据，可删除这两行）
    st.session_state["cache_last_shap_z"] = last_shap_z
    st.session_state["cache_last_z_test"] = last_z_test

    st.success("运行完成！已缓存结果，点击任意下载不会清空。")


# =====================================================
# 结果展示区：优先展示缓存结果（即使 download 触发 rerun 也能继续显示/下载）
# =====================================================
st.divider()
st.subheader("📦 当前缓存结果")

if "cache_stability_df" not in st.session_state:
    st.write("暂无缓存结果。请上传数据并点击「🚀 运行模型」。")
    st.stop()

metrics_df = st.session_state["cache_metrics_df"]
summary_df = st.session_state["cache_summary_df"]
stability_df = st.session_state["cache_stability_df"]
latent_df = st.session_state.get("cache_latent_df", None)

st.subheader("📊 模型性能（每次 run）")
st.dataframe(metrics_df, use_container_width=True)

st.subheader("📊 模型性能汇总（均值±标准差）")
st.dataframe(summary_df, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "⬇ 下载所有 run 的模型指标",
        st.session_state.get("cache_csv_metrics_all", b""),
        "model_metrics_all_runs.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "⬇ 下载指标汇总（均值±标准差）",
        st.session_state.get("cache_csv_summary", b""),
        "model_metrics_summary.csv",
        mime="text/csv",
    )
with c3:
    # 兼容你原来文件名
    st.download_button(
        "⬇ 下载全部基因重要性（Mean/CV/Freq）",
        st.session_state.get("cache_csv_stability", b""),
        "latent_shap_gene_importance_stability.csv",
        mime="text/csv",
    )

st.subheader("📌 生物标志物稳定性（Frequency / CV）")
st.dataframe(stability_df.head(50), use_container_width=True)

st.download_button(
    "⬇ 下载稳定性统计表",
    st.session_state.get("cache_csv_stability", b""),
    "biomarker_stability.csv",
    mime="text/csv",
)

st.subheader("🧬 Top 20 潜在生物标志物（MeanImportance 排名）")
st.dataframe(stability_df.sort_values("MeanImportance", ascending=False).head(20), use_container_width=True)

# =====================================================
# SHAP 可视化（用缓存的最后一次 run）
# =====================================================
last_shap_z = st.session_state.get("cache_last_shap_z", None)
last_z_test = st.session_state.get("cache_last_z_test", None)

if (last_shap_z is not None) and (last_z_test is not None):
    st.divider()
    st.subheader("🔍 Latent SHAP Summary（dot）")
    fig1 = plt.figure()
    shap.summary_plot(last_shap_z, features=last_z_test, show=False)
    st.pyplot(fig1)

    st.subheader("📊 Latent SHAP Summary（bar）")
    fig2 = plt.figure()
    shap.summary_plot(last_shap_z, features=last_z_test, plot_type="bar", show=False)
    st.pyplot(fig2)

    if latent_df is not None:
        st.subheader("📈 Top 20 Latent 维度重要性（MeanAbsSHAP）")
        st.dataframe(latent_df.head(20), use_container_width=True)

        fig3 = plt.figure()
        plt.bar(latent_df.head(20)["LatentDim"].astype(str), latent_df.head(20)["MeanAbsSHAP"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig3)

        csv_latent = st.session_state.get("cache_csv_latent", None)
        if csv_latent is not None:
            st.download_button(
                "⬇ 下载 latent 维度 MeanAbsSHAP",
                csv_latent,
                "latent_mean_abs_shap.csv",
                mime="text/csv",
            )

# =====================================================
# Cox 生存验证（可选）：lifelines 缺失则跳过，不中断
# =====================================================
if use_survival:
    st.divider()
    st.subheader("⏱ Cox 生存验证（可选）")

    if surv_file is None:
        st.warning("你开启了 Cox 验证，但没有上传生存数据文件。")
    else:
        try:
            from lifelines import CoxPHFitter
            from lifelines.utils import concordance_index

            lifelines_ok = True
        except Exception:
            lifelines_ok = False

        if not lifelines_ok:
            st.warning("未检测到 lifelines，已自动跳过 Cox 验证（可用 pip install lifelines 启用）。")
        else:
            # 重新读取文件（download 触发 rerun 时，surv_file 仍可能存在）
            surv = read_csv_cached(surv_file)

            needed = {"Sample", "Time", "Event"}
            if not needed.issubset(set(surv.columns)):
                st.error("生存数据必须包含列：Sample, Time, Event")
            else:
                surv["Sample"] = surv["Sample"].astype(str)
                surv = surv.set_index("Sample")

                # 注意：rna 只在 run_button 时存在；这里从上传文件重新读一遍以保证独立
                if (rna_file is None):
                    st.warning("当前会话未检测到 RNA 文件上传（或已刷新）。请重新上传 RNA 文件以进行 Cox 验证。")
                else:
                    rna_tmp = read_csv_cached(rna_file)
                    if "Unnamed: 0" in rna_tmp.columns:
                        rna_tmp = rna_tmp.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")
                    rna_tmp.columns = rna_tmp.columns.astype(str)

                    common = list(set(rna_tmp.columns).intersection(set(surv.index)))
                    if len(common) < 10:
                        st.error("生存数据与 RNA 的共同样本太少（<10），无法做 Cox 验证。")
                    else:
                        # 保持 RNA 顺序
                        common = [s for s in rna_tmp.columns if s in common]
                        surv_aligned = surv.loc[common]
                        rna_aligned = rna_tmp[common]

                        freq_col = f"Top{int(top_k)}_Freq"
                        selected = (
                            stability_df[
                                (stability_df[freq_col] >= float(freq_thr)) & (stability_df["CV"] <= float(cv_thr))
                            ]
                            .sort_values("MeanImportance", ascending=False)["Gene"]
                            .head(int(max_surv_genes))
                            .tolist()
                        )

                        st.write(
                            {
                                "共同样本数": int(len(common)),
                                "筛选阈值": f"{freq_col}≥{freq_thr}, CV≤{cv_thr}",
                                "进入 Cox 的基因数": int(len(selected)),
                            }
                        )

                        if len(selected) < 2:
                            st.warning("筛选后基因太少（<2）。请放宽阈值或增大 top_k / n_runs。")
                        else:
                            X_surv = rna_aligned.loc[selected].T  # samples x genes
                            df_cox = pd.concat([surv_aligned[["Time", "Event"]], X_surv], axis=1).dropna()

                            if df_cox.shape[0] < 10:
                                st.warning("去除缺失值后样本太少，无法稳定拟合。")
                            else:
                                df_train, df_test = train_test_split(
                                    df_cox, test_size=float(test_size), random_state=int(seed_base)
                                )

                                cph = CoxPHFitter(penalizer=float(cox_penalizer))
                                cph.fit(df_train, duration_col="Time", event_col="Event")

                                risk = cph.predict_partial_hazard(df_test)
                                c_index = concordance_index(df_test["Time"], -risk.values, df_test["Event"])
                                st.write({"C-index": float(c_index)})

                                st.subheader("📌 Cox 回归系数 Top 20")
                                coef_df = cph.summary.reset_index().rename(columns={"index": "Feature"})
                                coef_df = coef_df.sort_values("coef", ascending=False)
                                st.dataframe(coef_df.head(20), use_container_width=True)

                                st.download_button(
                                    "⬇ 下载 Cox summary",
                                    cph.summary.to_csv().encode("utf-8"),
                                    "cox_summary.csv",
                                    mime="text/csv",
                                )

                                out_risk = pd.DataFrame(
                                    {
                                        "Sample": df_test.index.astype(str),
                                        "Time": df_test["Time"].values,
                                        "Event": df_test["Event"].values,
                                        "RiskScore": risk.values.flatten(),
                                    }
                                )
                                st.download_button(
                                    "⬇ 下载测试集风险分数（RiskScore）",
                                    out_risk.to_csv(index=False).encode("utf-8"),
                                    "cox_test_risk_scores.csv",
                                    mime="text/csv",
                                )

# =====================================================
# app.py
# VAEjMLP latent-SHAP + 稳定性 + SHAP可视化 + Cox验证 + Top20:
#   ① 主流程：VAE+MLP、多次 run 稳定性（Freq/CV）、Top20、latent SHAP
#   ② 下载区：所有结果持久化（download 不清空）
#   ③ GO/KEGG 富集（gseapy.enrichr, 需联网）
#   ④ 差异分析（labels 两组，Welch t-test + FDR + 火山图 + 箱线图）
#   ⑤ 聚类（Top20 表达）+ 聚类分组生存（KM/logrank/Cox）
#
# ✅ 已修复：差异分析报错 KeyError 'Sample'
#   - clean_columns() 去 BOM/空白
#   - normalize_labels_df() 强制输出 Sample/Label 并清洗列名
#   - compute_de_top_genes() 使用 reindex 对齐，避免 loc KeyError
#
# 依赖：
#   基础：streamlit pandas numpy torch scikit-learn shap matplotlib
#   差异：scipy statsmodels
#   富集：gseapy（需要外网访问 Enrichr）
#   生存：lifelines
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans

import shap
import matplotlib.pyplot as plt

# ----------------- Optional deps -----------------
try:
    from scipy.stats import ttest_ind
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    import gseapy as gp
    GSEAPY_OK = True
except Exception:
    GSEAPY_OK = False

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index
    LIFELINES_OK = True
except Exception:
    LIFELINES_OK = False


# =====================================================
# Utils
# =====================================================
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@st.cache_data(show_spinner=False)
def read_csv_cached(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_rename_index_col(df: pd.DataFrame) -> pd.DataFrame:
    # 兼容：用户没设置 index_col 导致出现 Unnamed: 0
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")
    return df


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 去 BOM/空白，避免出现肉眼看是 Sample 实际是 \ufeffSample 的 KeyError
    """
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def normalize_labels_df(labels_raw: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 强制输出两列：Sample, Label
    - 自动识别列名
    - 去 BOM/空白
    """
    labels = clean_columns(labels_raw)

    if labels.shape[1] < 2:
        raise ValueError("Label 文件至少需要两列（样本列 + 标签列）。")

    col_norm_map = {_norm(c): c for c in labels.columns}

    sample_alias = ["sample", "sampleid", "sample_id", "id", "patient", "patientid", "subject", "subjectid"]
    label_alias = ["label", "group", "class", "y", "target", "phenotype", "status", "casecontrol", "case_control"]

    sample_col = None
    label_col = None

    for a in sample_alias:
        key = _norm(a)
        if key in col_norm_map:
            sample_col = col_norm_map[key]
            break

    for a in label_alias:
        key = _norm(a)
        if key in col_norm_map:
            label_col = col_norm_map[key]
            break

    if sample_col is None:
        sample_col = labels.columns[0]
    if label_col is None:
        label_col = labels.columns[1] if labels.columns[1] != sample_col else labels.columns[0]

    labels = labels.rename(columns={sample_col: "Sample", label_col: "Label"})
    labels["Sample"] = labels["Sample"].astype(str).str.strip()
    labels["Label"] = labels["Label"]

    return labels[["Sample", "Label"]]


def align_rna_labels(rna_raw: pd.DataFrame, labels_raw: pd.DataFrame):
    """
    rna: genes x samples
    labels: Sample, Label
    返回：对齐后的 rna, labels（顺序严格与 rna.columns 一致）
    """
    rna = safe_rename_index_col(rna_raw.copy())
    rna = clean_columns(rna)
    rna.columns = rna.columns.astype(str)

    labels = normalize_labels_df(labels_raw)

    samples_rna = set(rna.columns.tolist())
    samples_lab = set(labels["Sample"].tolist())
    common = sorted(list(samples_rna.intersection(samples_lab)))
    if len(common) < 4:
        raise ValueError(f"RNA 与 Label 交集样本数太少（{len(common)}），无法训练。")

    if samples_rna != samples_lab:
        st.warning("RNA 样本与 Label 样本集合不完全一致，将取交集对齐。")
        rna = rna[common]
        labels = labels.set_index("Sample").loc[common].reset_index()

    # 保证顺序一致
    labels = labels.set_index("Sample").loc[rna.columns].reset_index()

    # 保险：列名必须存在
    labels = clean_columns(labels)
    if "Sample" not in labels.columns:
        labels = labels.rename(columns={labels.columns[0]: "Sample"})
    if "Label" not in labels.columns:
        raise ValueError("labels 中未找到 Label 列，请检查上传文件。")

    return rna, labels


def ensure_2d_shap(shap_values, features_2d: np.ndarray) -> np.ndarray:
    """
    兼容 shap 多版本返回值：
    - list([array])
    - array
    - (n,d,1) / (1,n,d) 等
    最终输出 (n,d)
    """
    if isinstance(shap_values, list):
        shap_z = shap_values[0]
    else:
        shap_z = shap_values

    shap_z = np.array(shap_z)

    if shap_z.ndim == 3 and shap_z.shape[-1] == 1:
        shap_z = shap_z[:, :, 0]
    if shap_z.ndim == 3 and shap_z.shape[0] == 1:
        shap_z = shap_z[0]

    if shap_z.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_z.shape}")

    if shap_z.shape[0] != features_2d.shape[0] or shap_z.shape[1] != features_2d.shape[1]:
        raise ValueError(f"Shape mismatch: shap={shap_z.shape}, features={features_2d.shape}")

    return shap_z


def compute_de_top_genes(rna: pd.DataFrame, labels: pd.DataFrame, top_genes: list):
    """
    ✅ 两组差异：Welch t-test + log2FC + BH-FDR
    ✅ 不再用 loc[rna.columns] 直接索引（容易 KeyError），改用 reindex
    """
    if not SCIPY_OK:
        raise RuntimeError("缺少 scipy，无法做 t-test。请安装：pip install scipy")
    if not STATSMODELS_OK:
        raise RuntimeError("缺少 statsmodels，无法做 FDR。请安装：pip install statsmodels")

    rna = clean_columns(rna.copy())
    rna.columns = rna.columns.astype(str)

    lab = clean_columns(labels.copy())
    if ("Sample" not in lab.columns) or ("Label" not in lab.columns):
        lab = normalize_labels_df(lab)
    else:
        lab["Sample"] = lab["Sample"].astype(str).str.strip()

    # ✅ 用 reindex 对齐，避免 KeyError
    lab2 = lab.set_index("Sample").reindex(rna.columns)

    missing = lab2.index[lab2["Label"].isna()].tolist()
    if len(missing) > 0:
        raise ValueError(f"labels 中缺少 {len(missing)} 个样本的标签（示例前5个）：{missing[:5]}")

    groups = pd.Series(lab2["Label"].values).unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"差异分析需要两组 Label，目前发现 {len(groups)} 组：{groups}")

    g0, g1 = groups[0], groups[1]
    s0 = lab2[lab2["Label"] == g0].index.tolist()
    s1 = lab2[lab2["Label"] == g1].index.tolist()

    if len(s0) < 2 or len(s1) < 2:
        raise ValueError(f"两组样本数不足：{g0}={len(s0)}, {g1}={len(s1)}（每组至少 2）")

    eps = 1e-9
    rows = []
    for gene in top_genes:
        if gene not in rna.index:
            continue
        x0 = rna.loc[gene].reindex(s0).astype(float).values
        x1 = rna.loc[gene].reindex(s1).astype(float).values
        stat, p = ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
        m0 = np.nanmean(x0)
        m1 = np.nanmean(x1)
        log2fc = np.log2((m1 + eps) / (m0 + eps))
        rows.append([gene, m0, m1, log2fc, p])

    de = pd.DataFrame(rows, columns=["Gene", f"Mean({g0})", f"Mean({g1})", "log2FC", "p_value"])
    if len(de) == 0:
        raise ValueError("Top genes 在 RNA 中未匹配到任何基因。")

    de["FDR"] = multipletests(de["p_value"].values, method="fdr_bh")[1]
    de = de.sort_values(["FDR", "p_value"], ascending=True).reset_index(drop=True)
    return de, (g0, g1)


def run_enrichr(top_genes: list, organism: str = "Human"):
    """
    GO/KEGG via Enrichr (gseapy.enrichr) —— 需要联网
    """
    if not GSEAPY_OK:
        raise RuntimeError("缺少 gseapy，无法做 GO/KEGG。请安装：pip install gseapy")

    if organism.lower().startswith("h"):
        libs = [
            "GO_Biological_Process_2021",
            "GO_Molecular_Function_2021",
            "GO_Cellular_Component_2021",
            "KEGG_2021_Human",
        ]
    else:
        libs = [
            "GO_Biological_Process_2021",
            "GO_Molecular_Function_2021",
            "GO_Cellular_Component_2021",
            "KEGG_2021_Mouse",
        ]

    out = {}
    for lib in libs:
        enr = gp.enrichr(gene_list=top_genes, gene_sets=lib, organism=organism, outdir=None)
        out[lib] = enr.results.copy()
    return out


def cluster_samples_by_top_genes(rna: pd.DataFrame, top_genes: list, n_clusters: int = 2, seed: int = 42):
    """
    基于 Top genes 表达做 KMeans 聚类
    """
    rna = clean_columns(rna.copy())
    rna.columns = rna.columns.astype(str)

    genes_exist = [g for g in top_genes if g in rna.index]
    if len(genes_exist) < 2:
        raise ValueError("Top genes 在 RNA 中匹配到的基因太少（<2），无法聚类。")

    X = rna.loc[genes_exist].T.astype(float)
    X_scaled = StandardScaler().fit_transform(X.values)

    km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto")
    clusters = km.fit_predict(X_scaled)

    cluster_df = pd.DataFrame({"Sample": X.index.astype(str), "Cluster": clusters.astype(int)})
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index.astype(str), columns=genes_exist)
    return cluster_df, X_scaled_df


def km_plot_by_group(surv_df: pd.DataFrame, group_col: str, time_col: str = "Time", event_col: str = "Event"):
    if not LIFELINES_OK:
        raise RuntimeError("缺少 lifelines，无法做 KM/Cox。请安装：pip install lifelines")

    fig = plt.figure()
    kmf = KaplanMeierFitter()
    groups = sorted(surv_df[group_col].unique().tolist())

    for g in groups:
        dfg = surv_df[surv_df[group_col] == g]
        kmf.fit(durations=dfg[time_col], event_observed=dfg[event_col], label=f"{group_col}={g} (n={len(dfg)})")
        kmf.plot(ci_show=False)

    plt.title("Kaplan-Meier")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.tight_layout()

    p_lr = None
    if len(groups) == 2:
        g0, g1 = groups
        df0 = surv_df[surv_df[group_col] == g0]
        df1 = surv_df[surv_df[group_col] == g1]
        lr = logrank_test(
            df0[time_col], df1[time_col],
            event_observed_A=df0[event_col],
            event_observed_B=df1[event_col],
        )
        p_lr = float(lr.p_value)

    return fig, p_lr


# =====================================================
# Model
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
# Page + Navigation
# =====================================================
st.set_page_config(page_title="VAEjMLP latent-SHAP BioApp", layout="wide")
st.title("🧬 VAEjMLP + latent SHAP 生物标志物分析")

with st.expander("🧰 工具", expanded=False):
    if st.button("🧹 清空缓存结果（不会清空上传文件）"):
        for k in list(st.session_state.keys()):
            if k.startswith("cache_"):
                st.session_state.pop(k, None)
        st.rerun()

with st.sidebar:
    st.header("导航")
    module = st.radio(
        "选择模块",
        [
            "① 训练/SHAP/稳定性（主流程）",
            "② 结果下载与回显",
            "③ GO/KEGG 富集分析（Top20）",
            "④ 差异分析（labels 分组，Top20）",
            "⑤ 聚类（Top20）+ 生存分析（KM/Cox）",
        ],
        index=0,
    )

    st.divider()
    st.header("主流程参数")
    latent_dim = st.number_input("latent_dim", min_value=4, max_value=1024, value=128, step=4)
    n_epochs = st.number_input("训练轮数 epochs", min_value=10, max_value=2000, value=100, step=10)
    lr = st.number_input("学习率 lr", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f")
    ce_weight = st.number_input("CE 权重（loss = KL + ce_weight*CE）", min_value=0.0, max_value=10.0, value=0.001, step=0.001)
    test_size = st.slider("test_size", 0.05, 0.5, 0.2, 0.05)

    st.subheader("稳定性 (multi-run)")
    n_runs = st.slider("重复运行次数 n_runs", 1, 50, 10)
    top_k = st.slider("TopK 频率统计", 5, 300, 20)
    seed_base = st.number_input("seed_base", value=42, step=1)

    st.subheader("SHAP 计算")
    background_n = st.slider("background 样本数", 10, 200, 50)
    shap_nsamples = st.slider("KernelExplainer nsamples", 50, 500, 100, 50)

    st.divider()
    st.header("聚类/生存参数")
    cluster_k = st.slider("聚类簇数 K", 2, 6, 2)
    cox_penalizer = st.number_input("Cox L2 penalizer", min_value=0.0, max_value=10.0, value=0.1, step=0.1)


# =====================================================
# Upload area (global)
# =====================================================
st.divider()
u1, u2, u3 = st.columns([1, 1, 1])
with u1:
    rna_file = st.file_uploader("上传 RNA-seq（genes×samples）CSV", type="csv", key="rna_uploader")
with u2:
    label_file = st.file_uploader("上传 Label CSV（列名可不同，会自动识别 Sample/Label）", type="csv", key="label_uploader")
with u3:
    surv_file = st.file_uploader("上传 生存 CSV（Sample, Time, Event，可选）", type="csv", key="surv_uploader")

run_button = st.button("🚀 运行主流程（训练 + SHAP + 稳定性）", type="primary")


# =====================================================
# Main pipeline
# =====================================================
if run_button:
    if rna_file is None or label_file is None:
        st.error("请先上传 RNA 表达矩阵和 Label 文件。")
        st.stop()

    with st.spinner("读取并对齐数据..."):
        rna_raw = read_csv_cached(rna_file)
        labels_raw = read_csv_cached(label_file)
        try:
            rna, labels = align_rna_labels(rna_raw, labels_raw)
        except Exception as e:
            st.error(f"数据对齐失败：{e}")
            st.stop()

    if rna.shape[0] < 2 or rna.shape[1] < 4:
        st.error("RNA 矩阵维度不对：需要 genes×samples 且样本数至少 4。")
        st.stop()

    genes = rna.index.astype(str).tolist()
    y = labels["Label"].values

    X = MinMaxScaler().fit_transform(rna.T.values)

    all_importances = []
    topk_lists = []
    metrics_runs = []
    last_shap_z = None
    last_z_test = None
    last_latent_df = None

    prog = st.progress(0)
    status = st.empty()

    for run_i in range(int(n_runs)):
        seed = int(seed_base + run_i)
        set_seed(seed)
        status.write(f"Run {run_i+1}/{n_runs} | seed={seed}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=seed,
            stratify=y if len(pd.Series(y).unique()) == 2 else None,
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_train_t = torch.tensor(pd.Series(y_train).astype(float).values, dtype=torch.float32).view(-1, 1)

        vae = VAE(X.shape[1], int(latent_dim))
        mlp = MLP(int(latent_dim))
        optimizer = optim.Adam(list(vae.parameters()) + list(mlp.parameters()), lr=float(lr))

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

        with torch.no_grad():
            z_train, _, _ = vae(X_train_t)

        z_train_np = z_train.cpu().numpy()
        z_test_np = z_test.cpu().numpy()

        def mlp_predict(z_numpy):
            z_t = torch.tensor(z_numpy, dtype=torch.float32)
            with torch.no_grad():
                out = mlp(z_t).cpu().numpy()
            return out.reshape(-1)

        bg_n = int(min(background_n, z_train_np.shape[0]))
        background_z = shap.sample(z_train_np, bg_n)

        explainer = shap.KernelExplainer(mlp_predict, background_z)
        shap_values = explainer.shap_values(z_test_np, nsamples=int(shap_nsamples))
        shap_z = ensure_2d_shap(shap_values, z_test_np)

        W_gene_hidden = vae.fc1.weight.detach().cpu().numpy()
        abs_shap_z = np.mean(np.abs(shap_z), axis=0)
        scale = float(np.sum(abs_shap_z))

        gene_importance = {}
        for i, gene in enumerate(genes):
            gene_importance[gene] = float(np.mean(np.abs(W_gene_hidden[:, i])) * scale)

        imp_s = pd.Series(gene_importance).reindex(genes)
        all_importances.append(imp_s)
        topk_lists.append(imp_s.sort_values(ascending=False).head(int(top_k)).index.tolist())

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

    metrics_df = pd.DataFrame(metrics_runs)
    summary_df = metrics_df[["AUC", "Accuracy", "Precision", "Recall"]].agg(["mean", "std"]).T.reset_index()
    summary_df.columns = ["Metric", "Mean", "Std"]

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

    top20_genes = stability_df.sort_values("MeanImportance", ascending=False)["Gene"].head(20).tolist()

    st.session_state["cache_rna"] = rna
    st.session_state["cache_labels"] = labels
    st.session_state["cache_top20_genes"] = top20_genes

    st.session_state["cache_metrics_df"] = metrics_df
    st.session_state["cache_summary_df"] = summary_df
    st.session_state["cache_stability_df"] = stability_df
    st.session_state["cache_latent_df"] = last_latent_df

    st.session_state["cache_last_shap_z"] = last_shap_z
    st.session_state["cache_last_z_test"] = last_z_test

    st.session_state["cache_csv_metrics_all"] = to_csv_bytes(metrics_df)
    st.session_state["cache_csv_summary"] = to_csv_bytes(summary_df)
    st.session_state["cache_csv_stability"] = to_csv_bytes(stability_df)
    st.session_state["cache_csv_latent"] = to_csv_bytes(last_latent_df) if last_latent_df is not None else None

    for k in [
        "cache_enrich_go_kegg",
        "cache_de_df",
        "cache_de_groups",
        "cache_cluster_df",
        "cache_cluster_X_scaled",
        "cache_surv_aligned",
        "cache_cox_cluster_summary",
    ]:
        st.session_state.pop(k, None)

    st.success("✅ 主流程运行完成：已缓存结果（下载/切换模块不会丢失）。")


# =====================================================
# Module ①
# =====================================================
if module.startswith("①"):
    st.subheader("① 训练 / SHAP / 稳定性（主流程回显）")

    if "cache_stability_df" not in st.session_state:
        st.info("请先上传数据并点击「🚀 运行主流程」。")
    else:
        metrics_df = st.session_state["cache_metrics_df"]
        summary_df = st.session_state["cache_summary_df"]
        stability_df = st.session_state["cache_stability_df"]
        latent_df = st.session_state.get("cache_latent_df", None)
        top20 = st.session_state["cache_top20_genes"]

        st.markdown("### 📊 模型性能（每次 run）")
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown("### 📊 模型性能汇总（均值±标准差）")
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### 📌 生物标志物稳定性（Frequency / CV）")
        st.dataframe(stability_df.head(50), use_container_width=True)

        st.markdown("### 🧬 Top 20 潜在生物标志物（MeanImportance）")
        st.code("\n".join(top20))

        last_shap_z = st.session_state.get("cache_last_shap_z", None)
        last_z_test = st.session_state.get("cache_last_z_test", None)
        if last_shap_z is not None and last_z_test is not None:
            st.markdown("### 🔍 Latent SHAP Summary（dot）")
            fig1 = plt.figure()
            shap.summary_plot(last_shap_z, features=last_z_test, show=False)
            st.pyplot(fig1)

            st.markdown("### 📊 Latent SHAP Summary（bar）")
            fig2 = plt.figure()
            shap.summary_plot(last_shap_z, features=last_z_test, plot_type="bar", show=False)
            st.pyplot(fig2)

        if latent_df is not None:
            st.markdown("### 📈 Top 20 latent 维度重要性（MeanAbsSHAP）")
            st.dataframe(latent_df.head(20), use_container_width=True)

            fig3 = plt.figure()
            top_lat = latent_df.head(20)
            plt.bar(top_lat["LatentDim"].astype(str), top_lat["MeanAbsSHAP"])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig3)


# =====================================================
# Module ②
# =====================================================
if module.startswith("②"):
    st.subheader("② 结果下载与回显（download 不会清空）")

    if "cache_stability_df" not in st.session_state:
        st.info("暂无缓存结果。请先运行主流程。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button(
                "⬇ 模型指标（all runs）",
                st.session_state.get("cache_csv_metrics_all", b""),
                "model_metrics_all_runs.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "⬇ 指标汇总（mean±std）",
                st.session_state.get("cache_csv_summary", b""),
                "model_metrics_summary.csv",
                mime="text/csv",
            )
        with c3:
            st.download_button(
                "⬇ 基因稳定性（Mean/CV/Freq）",
                st.session_state.get("cache_csv_stability", b""),
                "latent_shap_gene_importance_stability.csv",
                mime="text/csv",
            )
        with c4:
            csv_latent = st.session_state.get("cache_csv_latent", None)
            st.download_button(
                "⬇ latent MeanAbsSHAP",
                csv_latent if csv_latent is not None else b"",
                "latent_mean_abs_shap.csv",
                mime="text/csv",
                disabled=(csv_latent is None),
            )

        st.markdown("### Top20 基因列表")
        st.code("\n".join(st.session_state["cache_top20_genes"]))


# =====================================================
# Module ③
# =====================================================
if module.startswith("③"):
    st.subheader("③ GO / KEGG 富集分析（Top20）")

    if "cache_top20_genes" not in st.session_state:
        st.info("请先运行主流程，生成 Top20 基因。")
    else:
        top_genes = st.session_state["cache_top20_genes"]
        st.markdown("### 输入基因（Top20）")
        st.code("\n".join(top_genes))

        org = st.selectbox("物种（Enrichr organism）", ["Human", "Mouse"], index=0)

        if not GSEAPY_OK:
            st.warning("未安装 gseapy，无法做 GO/KEGG。请安装：pip install gseapy")
        else:
            if st.button("🧪 运行 GO/KEGG（Enrichr）"):
                with st.spinner("富集分析运行中（需要联网访问 Enrichr）..."):
                    try:
                        res_dict = run_enrichr(top_genes, organism=org)
                        st.session_state["cache_enrich_go_kegg"] = res_dict
                        st.success("富集完成 ✅")
                    except Exception as e:
                        st.error(f"富集失败：{e}")

            if "cache_enrich_go_kegg" in st.session_state:
                res_dict = st.session_state["cache_enrich_go_kegg"]
                st.markdown("### 结果展示（每个库默认取前 20 条）")
                for lib, df in res_dict.items():
                    st.markdown(f"#### {lib}")
                    st.dataframe(df.head(20), use_container_width=True)
                    st.download_button(
                        f"⬇ 下载 {lib}",
                        df.to_csv(index=False).encode("utf-8"),
                        f"enrichr_{lib}.csv",
                        mime="text/csv",
                    )

                st.caption("提示：如果部署环境无法访问外网，Enrichr 会失败。")


# =====================================================
# Module ④
# =====================================================
if module.startswith("④"):
    st.subheader("④ 差异分析（labels 分组，Top20）")

    if "cache_rna" not in st.session_state or "cache_labels" not in st.session_state:
        st.info("请先运行主流程。")
    else:
        rna = st.session_state["cache_rna"]
        labels = st.session_state["cache_labels"]
        top_genes = st.session_state["cache_top20_genes"]

        with st.expander("🔎 调试信息（labels 真实列名 repr）", expanded=False):
            st.write([repr(c) for c in labels.columns])
            st.dataframe(labels.head(10), use_container_width=True)

        if not SCIPY_OK:
            st.warning("未安装 scipy，无法做差异分析：pip install scipy")
        if not STATSMODELS_OK:
            st.warning("未安装 statsmodels，无法做 FDR：pip install statsmodels")

        if SCIPY_OK and STATSMODELS_OK:
            if st.button("🧬 运行 Top20 差异分析（t-test + FDR）"):
                with st.spinner("差异分析计算中..."):
                    try:
                        de_df, groups = compute_de_top_genes(rna, labels, top_genes)
                        st.session_state["cache_de_df"] = de_df
                        st.session_state["cache_de_groups"] = groups
                        st.success("差异分析完成 ✅")
                    except Exception as e:
                        st.error(f"差异分析失败：{e}")

        if "cache_de_df" in st.session_state:
            de_df = st.session_state["cache_de_df"]
            g0, g1 = st.session_state.get("cache_de_groups", ("Group0", "Group1"))

            st.markdown(f"### 差异结果（{g0} vs {g1}）")
            st.dataframe(de_df, use_container_width=True)

            st.download_button(
                "⬇ 下载差异分析结果（Top20）",
                de_df.to_csv(index=False).encode("utf-8"),
                "top20_differential_expression.csv",
                mime="text/csv",
            )

            st.markdown("### 火山图（Top20）")
            figv = plt.figure()
            x = de_df["log2FC"].values
            yv = -np.log10(de_df["p_value"].values + 1e-300)
            plt.scatter(x, yv)
            for _, row in de_df.iterrows():
                plt.text(row["log2FC"], -np.log10(row["p_value"] + 1e-300), row["Gene"], fontsize=8)
            plt.xlabel("log2FC")
            plt.ylabel("-log10(p)")
            plt.title("Volcano plot (Top20)")
            plt.tight_layout()
            st.pyplot(figv)

            st.markdown("### 箱线图（选择一个基因）")
            gene_pick = st.selectbox("选择基因", de_df["Gene"].tolist(), index=0)

            # group labels（用 normalize 确保 Sample/Label 正确）
            lab = labels.copy()
            if "Sample" not in clean_columns(lab).columns or "Label" not in clean_columns(lab).columns:
                lab = normalize_labels_df(lab)
            else:
                lab = clean_columns(lab)
                lab["Sample"] = lab["Sample"].astype(str).str.strip()
            lab2 = lab.set_index("Sample").reindex(rna.columns)

            groups_u = pd.Series(lab2["Label"].values).unique().tolist()
            s0 = lab2[lab2["Label"] == groups_u[0]].index.tolist()
            s1 = lab2[lab2["Label"] == groups_u[1]].index.tolist()

            x0 = rna.loc[gene_pick].reindex(s0).astype(float).values
            x1 = rna.loc[gene_pick].reindex(s1).astype(float).values

            figb = plt.figure()
            plt.boxplot([x0, x1], labels=[str(groups_u[0]), str(groups_u[1])])
            plt.title(f"{gene_pick} expression by Label")
            plt.ylabel("Expression")
            plt.tight_layout()
            st.pyplot(figb)


# =====================================================
# Module ⑤
# =====================================================
if module.startswith("⑤"):
    st.subheader("⑤ 聚类（Top20） + 生存分析（KM / Cox）")

    if "cache_rna" not in st.session_state or "cache_top20_genes" not in st.session_state:
        st.info("请先运行主流程。")
    else:
        rna = st.session_state["cache_rna"]
        top_genes = st.session_state["cache_top20_genes"]

        st.markdown("### 聚类输入基因（Top20）")
        st.code("\n".join(top_genes))

        if st.button("🧩 运行聚类（Top20 基因表达）"):
            with st.spinner("聚类中..."):
                try:
                    cluster_df, X_scaled_df = cluster_samples_by_top_genes(
                        rna=rna,
                        top_genes=top_genes,
                        n_clusters=int(cluster_k),
                        seed=int(seed_base),
                    )
                    st.session_state["cache_cluster_df"] = cluster_df
                    st.session_state["cache_cluster_X_scaled"] = X_scaled_df
                    st.success("聚类完成 ✅")
                except Exception as e:
                    st.error(f"聚类失败：{e}")

        if "cache_cluster_df" in st.session_state:
            cluster_df = st.session_state["cache_cluster_df"]
            X_scaled_df = st.session_state.get("cache_cluster_X_scaled", None)

            st.markdown("### 聚类结果（Sample → Cluster）")
            st.dataframe(cluster_df.head(100), use_container_width=True)

            st.download_button(
                "⬇ 下载聚类结果",
                cluster_df.to_csv(index=False).encode("utf-8"),
                "top20_cluster_labels.csv",
                mime="text/csv",
            )

            if X_scaled_df is not None:
                st.markdown("### 热图（z-score，样本按 Cluster 排序）")
                df_plot = X_scaled_df.copy()
                df_plot["Cluster"] = cluster_df.set_index("Sample").loc[df_plot.index]["Cluster"].values
                df_plot = df_plot.sort_values("Cluster")
                mat = df_plot.drop(columns=["Cluster"]).values

                fig_h = plt.figure(figsize=(10, 5))
                plt.imshow(mat, aspect="auto")
                plt.colorbar(label="z-score")
                plt.yticks([])
                plt.xticks(range(df_plot.shape[1] - 1), df_plot.drop(columns=["Cluster"]).columns, rotation=90, fontsize=7)
                plt.title("Top20 genes (z-score) sorted by Cluster")
                plt.tight_layout()
                st.pyplot(fig_h)

            st.markdown("## 生存分析（用 Cluster 分组）")
            if surv_file is None:
                st.info("未上传生存数据（Sample, Time, Event），仅展示聚类结果。")
            else:
                if not LIFELINES_OK:
                    st.warning("未安装 lifelines：pip install lifelines")
                else:
                    surv = clean_columns(read_csv_cached(surv_file))
                    if "Sample" not in surv.columns:
                        # 尝试自动识别 Sample
                        surv_cols = {_norm(c): c for c in surv.columns}
                        for a in ["sample", "sampleid", "id", "subject", "patient"]:
                            if _norm(a) in surv_cols:
                                surv = surv.rename(columns={surv_cols[_norm(a)]: "Sample"})
                                break

                    if "Sample" not in surv.columns or "Time" not in surv.columns or "Event" not in surv.columns:
                        st.error("生存数据必须包含列：Sample, Time, Event（Sample 可自动识别；Time/Event 需同名）")
                    else:
                        surv["Sample"] = surv["Sample"].astype(str).str.strip()
                        surv = surv.set_index("Sample")

                        cl = cluster_df.copy()
                        cl["Sample"] = cl["Sample"].astype(str).str.strip()
                        cl = cl.set_index("Sample")

                        common = sorted(list(set(cl.index).intersection(set(surv.index))))
                        if len(common) < 10:
                            st.error("生存数据与聚类样本交集太少（<10），无法生存分析。")
                        else:
                            surv_aligned = surv.loc[common].copy()
                            surv_aligned["Cluster"] = cl.loc[common]["Cluster"].astype(int).values

                            surv_aligned["Time"] = pd.to_numeric(surv_aligned["Time"], errors="coerce")
                            surv_aligned["Event"] = pd.to_numeric(surv_aligned["Event"], errors="coerce")
                            surv_aligned = surv_aligned.dropna(subset=["Time", "Event", "Cluster"])

                            st.markdown("### 对齐后的生存数据（含 Cluster）")
                            st.dataframe(surv_aligned.reset_index().head(50), use_container_width=True)

                            fig_km, p_lr = km_plot_by_group(surv_aligned, group_col="Cluster")
                            st.pyplot(fig_km)
                            if p_lr is not None:
                                st.write({"Log-rank p-value (2 groups)": p_lr})

                            st.markdown("### Cox（Cluster 作为协变量）")
                            df_cox = surv_aligned[["Time", "Event", "Cluster"]].copy()
                            df_cox = df_cox.reset_index(drop=True)
                            df_cox = pd.get_dummies(df_cox, columns=["Cluster"], drop_first=True)

                            df_train, df_test = train_test_split(df_cox, test_size=float(test_size), random_state=int(seed_base))

                            cph = CoxPHFitter(penalizer=float(cox_penalizer))
                            cph.fit(df_train, duration_col="Time", event_col="Event")

                            risk = cph.predict_partial_hazard(df_test)
                            c_index = concordance_index(df_test["Time"], -risk.values, df_test["Event"])
                            st.write({"C-index": float(c_index)})

                            cox_sum = cph.summary.reset_index()
                            st.dataframe(cox_sum, use_container_width=True)

                            st.download_button(
                                "⬇ 下载 Cox summary（Cluster）",
                                cox_sum.to_csv(index=False).encode("utf-8"),
                                "cox_cluster_summary.csv",
                                mime="text/csv",
                            )

                            out_risk = pd.DataFrame(
                                {
                                    "Time": df_test["Time"].values,
                                    "Event": df_test["Event"].values,
                                    "RiskScore": risk.values.flatten(),
                                }
                            )
                            st.download_button(
                                "⬇ 下载 Cox 测试集风险分数",
                                out_risk.to_csv(index=False).encode("utf-8"),
                                "cox_cluster_test_risk_scores.csv",
                                mime="text/csv",
                            )


# =====================================================
# Footer
# =====================================================
st.divider()
st.caption(
    "依赖提示：基础功能需 streamlit/pandas/numpy/torch/scikit-learn/shap/matplotlib；"
    "差异分析需 scipy + statsmodels；"
    "GO/KEGG（Enrichr）需 gseapy 且需要网络；"
    "生存分析需 lifelines。"
)

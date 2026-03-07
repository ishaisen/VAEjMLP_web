# =====================================================
# app.py (FULL REPLACEABLE VERSION)
# VAEjMLP latent-SHAP + 稳定性 + SHAP可视化 + GO/KEGG(按钮切换+气泡图)
# + DE(含箱线图) + 聚类 + 生存
#
# UI:
#   - Demo gate（未开启时仅显示开关）
#   - Hero: Input / Workflow / Output
#   - 顶部 Sticky Toolbar（锚点跳转+回到顶部+模块高亮）
#   - 模块导航：按钮（替代 Tabs）
#   - 下载中心：文件列表 + 单文件下载 + ZIP + REPORT.md（不依赖 tabulate）
#   - 清除缓存：一键清空运行结果与下载缓存
#
# CHANGE:
#   - 选中模块按钮（“Tabs”替代按钮）颜色更明显
#   - 不使用对号 ✅（选中态不显示任何对号/标记）
# =====================================================

import os
import io
import zipfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

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
# Demo files (HIDDEN)
# =====================================================
DEMO_DIR = "."
DEMO_RNA = "TCGA_GTEX_tmp.csv"
DEMO_LAB = "labels.csv"
DEMO_SUR = "sur.csv"


# =====================================================
# Utils
# =====================================================
def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


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


@st.cache_data(show_spinner=False)
def read_csv_path_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def safe_rename_index_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")
    return df


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def normalize_labels_df(labels_raw: pd.DataFrame):
    labels = clean_columns(labels_raw.copy())
    if labels.shape[1] < 2:
        raise ValueError("Label 文件至少需要两列（样本列 + 标签列）。")

    col_map = {_norm(c): c for c in labels.columns}
    sample_alias = ["sample", "sampleid", "sample_id", "id", "patient", "patientid", "subject", "subjectid"]
    label_alias = ["label", "group", "class", "y", "target", "phenotype", "status", "casecontrol", "case_control"]

    sample_col = None
    label_col = None
    for a in sample_alias:
        if _norm(a) in col_map:
            sample_col = col_map[_norm(a)]
            break
    for a in label_alias:
        if _norm(a) in col_map:
            label_col = col_map[_norm(a)]
            break

    fallback_sample = labels.columns[0]
    fallback_label = labels.columns[1] if labels.columns[1] != fallback_sample else labels.columns[0]

    if sample_col is None:
        sample_col = fallback_sample
    if label_col is None:
        label_col = fallback_label

    detect_info = {
        "raw_columns": list(labels.columns),
        "detected_sample_col": sample_col,
        "detected_label_col": label_col,
    }

    labels = labels.rename(columns={sample_col: "Sample", label_col: "Label"})
    labels["Sample"] = labels["Sample"].astype(str).str.strip()
    return labels[["Sample", "Label"]], detect_info


def align_rna_labels(rna_raw: pd.DataFrame, labels_raw: pd.DataFrame):
    rna = safe_rename_index_col(rna_raw.copy())
    rna = clean_columns(rna)
    rna.columns = rna.columns.astype(str)

    labels, label_detect = normalize_labels_df(labels_raw)

    samples_rna = set(rna.columns.tolist())
    samples_lab = set(labels["Sample"].tolist())
    common = sorted(list(samples_rna.intersection(samples_lab)))

    if len(common) < 4:
        raise ValueError(f"RNA 与 Label 交集样本数太少（{len(common)}），无法训练。")

    align_info = {
        "rna_samples": len(samples_rna),
        "label_samples": len(samples_lab),
        "common_samples": len(common),
        "used_samples": len(common),
        "took_intersection": False,
    }

    if samples_rna != samples_lab:
        align_info["took_intersection"] = True
        rna = rna[common]
        labels = labels.set_index("Sample").loc[common].reset_index()

    labels = labels.set_index("Sample").loc[rna.columns].reset_index()
    labels = clean_columns(labels)

    return rna, labels, align_info, label_detect


def ensure_2d_shap(shap_values, features_2d: np.ndarray) -> np.ndarray:
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


def df_to_markdown_fallback(df: pd.DataFrame) -> str:
    df2 = df.copy().fillna("")
    cols = [str(c) for c in df2.columns.tolist()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df2.iterrows():
        rows.append("| " + " | ".join([str(x) for x in r.tolist()]) + " |")
    return "\n".join([header, sep] + rows)


# ---------------- DE ----------------
def compute_de_top_genes(rna: pd.DataFrame, labels: pd.DataFrame, top_genes: list):
    if not SCIPY_OK:
        raise RuntimeError("缺少 scipy，无法做 t-test。请安装：pip install scipy")
    if not STATSMODELS_OK:
        raise RuntimeError("缺少 statsmodels，无法做 FDR。请安装：pip install statsmodels")

    rna = clean_columns(rna.copy())
    rna.columns = rna.columns.astype(str)

    lab = clean_columns(labels.copy())
    if ("Sample" not in lab.columns) or ("Label" not in lab.columns):
        lab, _ = normalize_labels_df(lab)
    else:
        lab["Sample"] = lab["Sample"].astype(str).str.strip()

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

        _, p = ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
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


def plot_gene_boxplots(rna: pd.DataFrame, labels: pd.DataFrame, genes: list, group_order=None):
    rna = clean_columns(rna.copy())
    rna.columns = rna.columns.astype(str)

    lab = clean_columns(labels.copy())
    if ("Sample" not in lab.columns) or ("Label" not in lab.columns):
        lab, _ = normalize_labels_df(lab)
    else:
        lab["Sample"] = lab["Sample"].astype(str).str.strip()

    lab2 = lab.set_index("Sample").reindex(rna.columns)
    if lab2["Label"].isna().any():
        missing = lab2.index[lab2["Label"].isna()].tolist()
        raise ValueError(f"labels 缺少 {len(missing)} 个样本的 Label（示例前5）：{missing[:5]}")

    groups = pd.Series(lab2["Label"].values).unique().tolist()
    if group_order is not None:
        groups = [g for g in group_order if g in groups]
    else:
        groups = sorted(groups, key=lambda x: str(x))

    genes_exist = [g for g in genes if g in rna.index]
    if len(genes_exist) == 0:
        raise ValueError("选择的基因在 RNA 里都找不到。")

    n = len(genes_exist)
    height = max(4.5, min(1.0 * n + 2.0, 18.0))
    fig = plt.figure(figsize=(9.5, height))

    for i, gene in enumerate(genes_exist, start=1):
        ax = plt.subplot(n, 1, i)
        data = []
        for g in groups:
            sids = lab2.index[lab2["Label"] == g].tolist()
            vals = rna.loc[gene].reindex(sids).astype(float).values
            data.append(vals)

        ax.boxplot(data, labels=[str(g) for g in groups], showfliers=False)
        ax.set_title(gene, fontsize=10, loc="left")
        ax.set_ylabel("Expr")
        ax.grid(True, axis="y", alpha=0.2)
        if i != n:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Label group")

    plt.tight_layout()
    return fig


# ---------------- Enrich ----------------
def run_enrichr(top_genes: list, organism: str = "Human"):
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


def _pick_col(df: pd.DataFrame, candidates: list, default=None):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return default


def plot_enrich_bubble(df: pd.DataFrame, title: str, top_n: int = 20):
    if df is None or len(df) == 0:
        raise ValueError("富集结果为空，无法绘图。")

    term_col = _pick_col(df, ["Term", "term"], default=df.columns[0])
    adjp_col = _pick_col(df, ["Adjusted P-value", "Adjusted P-value "])
    p_col = _pick_col(df, ["P-value", "p_value", "p-value"])
    comb_col = _pick_col(df, ["Combined Score", "combined_score", "Combined Score "])

    d = df.copy()
    if adjp_col is not None:
        d = d.sort_values(adjp_col, ascending=True)
    elif p_col is not None:
        d = d.sort_values(p_col, ascending=True)

    d = d.head(int(top_n)).copy()
    d[term_col] = d[term_col].astype(str)

    eps = 1e-300
    if adjp_col is not None:
        d["_adjp"] = pd.to_numeric(d[adjp_col], errors="coerce").fillna(1.0)
    elif p_col is not None:
        d["_adjp"] = pd.to_numeric(d[p_col], errors="coerce").fillna(1.0)
    else:
        d["_adjp"] = 1.0

    d["_mlog10"] = -np.log10(d["_adjp"].values + eps)

    if comb_col is not None:
        d["_x"] = pd.to_numeric(d[comb_col], errors="coerce").fillna(d["_mlog10"])
        x_label = comb_col
    else:
        d["_x"] = d["_mlog10"]
        x_label = "-log10(p_adj)"

    y_labels = d[term_col].tolist()[::-1]
    x = d["_x"].values[::-1]
    size = (d["_mlog10"].values[::-1] + 1.0) ** 2 * 12
    color = d["_adjp"].values[::-1]

    fig = plt.figure(figsize=(9.5, max(4.2, 0.28 * len(y_labels) + 1.8)))
    ax = plt.gca()
    sc = ax.scatter(x, range(len(y_labels)), s=size, c=color)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label("Adjusted P-value (smaller = better)")
    plt.tight_layout()
    return fig


# ---------------- Cluster/Survival ----------------
def cluster_samples_by_top_genes(rna: pd.DataFrame, top_genes: list, n_clusters: int = 2, seed: int = 42):
    rna = clean_columns(rna.copy())
    rna.columns = rna.columns.astype(str)

    genes_exist = [g for g in top_genes if g in rna.index]
    if len(genes_exist) < 2:
        raise ValueError("Top genes 在 RNA 中匹配到的基因太少（<2），无法聚类。")

    X = rna.loc[genes_exist].T.astype(float)  # samples x genes
    X_scaled = StandardScaler().fit_transform(X.values)

    km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto")
    clusters = km.fit_predict(X_scaled)

    cluster_df = pd.DataFrame({"Sample": X.index.astype(str), "Cluster": clusters.astype(int)})
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index.astype(str), columns=genes_exist)
    return cluster_df, X_scaled_df


def km_plot_by_group(surv_df: pd.DataFrame, group_col: str, time_col: str = "Time", event_col: str = "Event"):
    if not LIFELINES_OK:
        raise RuntimeError("缺少 lifelines，无法做 KM/Cox。请安装：pip install lifelines")

    fig = plt.figure(figsize=(7.8, 4.6))
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
# Artifact manager (Downloads center)
# =====================================================
def artifacts_init():
    if "cache_artifacts" not in st.session_state:
        st.session_state["cache_artifacts"] = {}
    if "cache_fig_pngs" not in st.session_state:
        st.session_state["cache_fig_pngs"] = {}


def artifact_put_bytes(name: str, b: bytes, mime: str, kind: str = "file", note: str = ""):
    artifacts_init()
    st.session_state["cache_artifacts"][name] = {"bytes": b, "mime": mime, "kind": kind, "note": note}


def artifact_put_df_csv(name: str, df: pd.DataFrame, note: str = ""):
    artifact_put_bytes(name, to_csv_bytes(df), "text/csv", kind="csv", note=note)


def artifact_put_fig_png(name: str, fig, note: str = ""):
    b = fig_to_png_bytes(fig)
    artifact_put_bytes(name, b, "image/png", kind="png", note=note)
    st.session_state["cache_fig_pngs"][name] = b


def build_report_md() -> str:
    src = st.session_state.get("cache_data_source", "unknown")
    at = st.session_state.get("cache_cached_at", "")
    params = st.session_state.get("cache_params", {})
    top20 = st.session_state.get("cache_top20_genes", [])
    summary_df = st.session_state.get("cache_summary_df", None)

    md = []
    md.append("# VAEjMLP latent-SHAP Results Report\n\n")
    md.append(f"- Generated at: **{datetime.now().isoformat(timespec='seconds')}**\n")
    md.append(f"- Cached at: **{at}**\n")
    md.append(f"- Data source: **{src}**\n\n")

    md.append("## Parameters\n")
    if params:
        for k, v in params.items():
            md.append(f"- {k}: {v}\n")
    else:
        md.append("- (no params captured)\n")
    md.append("\n")

    md.append("## Metrics Summary (mean ± std)\n")
    if isinstance(summary_df, pd.DataFrame):
        md.append(df_to_markdown_fallback(summary_df))
        md.append("\n\n")
    else:
        md.append("- (no summary)\n\n")

    md.append("## Top 20 Biomarkers\n")
    if top20:
        for g in top20:
            md.append(f"- {g}\n")
    else:
        md.append("- (no top20)\n")

    md.append("\n## Notes\n")
    md.append("- ZIP bundle includes CSV/PNG produced in current session.\n")
    md.append("- GO/KEGG requires gseapy and network access to Enrichr.\n")
    md.append("- DE requires scipy + statsmodels.\n")
    md.append("- Survival requires lifelines.\n")
    return "".join(md)


def build_results_zip(ts: str) -> bytes:
    artifacts_init()
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, meta in st.session_state["cache_artifacts"].items():
            zf.writestr(name, meta["bytes"])
        zf.writestr("REPORT.md", build_report_md().encode("utf-8"))
        meta = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_source": st.session_state.get("cache_data_source", "unknown"),
            "cached_at": st.session_state.get("cache_cached_at", ""),
            "bundle_ts": ts,
            "artifact_count": len(st.session_state["cache_artifacts"]),
        }
        zf.writestr("README_metadata.json", pd.Series(meta).to_json())
    zbuf.seek(0)
    return zbuf.read()


def artifact_table_df():
    artifacts_init()
    rows = []
    for name, meta in st.session_state["cache_artifacts"].items():
        size_kb = len(meta["bytes"]) / 1024.0
        rows.append([name, meta.get("kind", ""), meta.get("mime", ""), f"{size_kb:.1f} KB", meta.get("note", "")])
    if not rows:
        return pd.DataFrame(columns=["File", "Type", "MIME", "Size", "Note"])
    return pd.DataFrame(rows, columns=["File", "Type", "MIME", "Size", "Note"]).sort_values("Type")


# =====================================================
# Cache clear
# =====================================================
def clear_results_cache():
    for k in list(st.session_state.keys()):
        if k.startswith("cache_"):
            st.session_state.pop(k, None)
    try:
        st.cache_data.clear()
    except Exception:
        pass


# =====================================================
# Models
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
# Sticky Toolbar (anchor nav + scroll highlight)
# =====================================================
def render_sticky_toolbar():
    run_ok = "cache_stability_df" in st.session_state
    data_source = st.session_state.get("cache_data_source", "未运行")
    cached_at = st.session_state.get("cache_cached_at", "")

    status_badge = "已运行" if run_ok else "未运行"
    status_color = "#16A34A" if run_ok else "#F59E0B"

    html = f"""
    <style>
      .block-container {{ padding-top: 6.6rem; }}
      .stickybar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        background: rgba(255,255,255,0.86);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(15,23,42,0.10);
      }}
      .sticky-inner {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 10px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}
      .left {{ display:flex; align-items:center; gap:10px; min-width: 320px; flex-wrap: wrap; }}
      .brand {{ font-weight:800; letter-spacing:-0.02em; color:#0F172A; font-size:14px; white-space:nowrap; }}
      .badge {{
        display:inline-flex; align-items:center; gap:6px;
        padding:6px 10px; border-radius:999px;
        border: 1px solid rgba(15,23,42,0.12);
        background: rgba(255,255,255,0.70);
        font-size:12px; white-space:nowrap;
      }}
      .dot {{ width:8px; height:8px; border-radius:999px; background:{status_color}; display:inline-block; }}
      .right {{ display:flex; align-items:center; gap:8px; flex-wrap: wrap; justify-content:flex-end; }}
      .btn {{
        display:inline-flex; align-items:center; gap:8px;
        padding:8px 10px; border-radius:10px;
        border: 1px solid rgba(15,23,42,0.12);
        background: rgba(255,255,255,0.70);
        color:#0F172A; text-decoration:none;
        font-size:12px; cursor:pointer;
        transition: all .12s ease; user-select:none;
      }}
      .btn:hover {{ background: rgba(246,248,252,0.95); transform: translateY(-1px); }}
      .btn.primary {{ border-color: rgba(46,125,255,0.35); background: rgba(46,125,255,0.10); }}

      /* highlight by scroll (active) */
      .btn.active {{
        border-color: rgba(37,99,235,0.60);
        background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(59,130,246,0.14));
        box-shadow: 0 10px 22px rgba(37,99,235,0.16);
        color: #1D4ED8;
      }}
      .muted {{ opacity:0.65; font-size:12px; white-space:nowrap; }}
      .sep {{ width:1px; height:20px; background: rgba(15,23,42,0.10); margin:0 4px; }}
    </style>

    <div class="stickybar">
      <div class="sticky-inner">
        <div class="left">
          <div class="brand">🧬 VAEjMLP BioApp</div>
          <div class="badge"><span class="dot"></span>{status_badge}</div>
          <div class="badge">数据源：{data_source}</div>
          <div class="badge">缓存：{cached_at if cached_at else "—"}</div>
        </div>

        <div class="right">
          <a class="btn primary" href="#run">🚀 运行区</a>
          <div class="sep"></div>
          <a class="btn nav" id="nav-main" href="#main">① 主流程</a>
          <a class="btn nav" id="nav-download" href="#download">② 下载</a>
          <a class="btn nav" id="nav-enrich" href="#enrich">③ 富集</a>
          <a class="btn nav" id="nav-de" href="#de">④ 差异</a>
          <a class="btn nav" id="nav-survival" href="#survival">⑤ 生存</a>
          <div class="sep"></div>
          <a class="btn" href="#top">⬆ 回到顶部</a>
          <span class="muted">滚动高亮</span>
        </div>
      </div>
    </div>

    <script>
      const sections = [
        ["main", "nav-main"],
        ["download", "nav-download"],
        ["enrich", "nav-enrich"],
        ["de", "nav-de"],
        ["survival", "nav-survival"],
      ];
      function setActive(btnId) {{
        sections.forEach(([sec, id]) => {{
          const el = document.getElementById(id);
          if (el) el.classList.remove("active");
        }});
        const active = document.getElementById(btnId);
        if (active) active.classList.add("active");
      }}
      sections.forEach(([secId, btnId]) => {{
        const target = document.getElementById(secId);
        if (!target) return;
        const obs = new IntersectionObserver((entries) => {{
          entries.forEach(entry => {{
            if (entry.isIntersecting) setActive(btnId);
          }});
        }}, {{
          root: null,
          threshold: 0.01,
          rootMargin: "-35% 0px -60% 0px"
        }});
        obs.observe(target);
      }});
      setActive("nav-main");
    </script>
    """
    components.html(html, height=0)


# =====================================================
# Page + Styles
# =====================================================
st.set_page_config(page_title="VAEjMLP latent-SHAP BioApp", layout="wide")
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
render_sticky_toolbar()
artifacts_init()

st.markdown(
    """
    <style>
      .block-container { padding-bottom: 2rem; max-width: 1400px; }
      h1, h2, h3 { letter-spacing: -0.01em; }

      .heroWrap {
        border: 1px solid rgba(49,51,63,0.10);
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.80), rgba(255,255,255,0.50));
        box-shadow: 0 1px 16px rgba(0,0,0,0.03);
        margin-bottom: 14px;
      }
      .heroTitle { font-size: 20px; font-weight: 800; margin: 0 0 6px 0; }
      .heroSub { opacity: 0.75; margin: 0 0 12px 0; }

      .card {
        border: 1px solid rgba(49,51,63,0.12);
        border-radius: 16px;
        padding: 14px 14px;
        background: rgba(255,255,255,0.66);
        backdrop-filter: blur(6px);
        box-shadow: 0 1px 10px rgba(0,0,0,0.02);
        margin-bottom: 14px;
      }

      .kpi {
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(6px);
        box-shadow: 0 1px 10px rgba(0,0,0,0.03);
      }
      .kpi .label { font-size: 0.85rem; opacity: 0.7; }
      .kpi .value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; }
      .kpi .hint  { font-size: 0.8rem; opacity: 0.55; margin-top: 6px; }

      .smallMuted { opacity: 0.70; font-size: 0.90rem; }
      .stDataFrame { border-radius: 12px; overflow: hidden; }

      /* ---- module buttons: make active state obvious; no checkmark used ---- */
      .modbtn-row { display:flex; gap:10px; flex-wrap: wrap; margin: 8px 0 12px 0; }
      /* Streamlit button styling hook (best-effort) */
      div[data-testid="stButton"] > button.modbtn {
        border: 1px solid rgba(15,23,42,0.14) !important;
        background: rgba(255,255,255,0.72) !important;
        color: #0F172A !important;
        border-radius: 14px !important;
        padding: 0.55rem 0.8rem !important;
        font-weight: 800 !important;
        transition: all .14s ease !important;
      }
      div[data-testid="stButton"] > button.modbtn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 22px rgba(2,6,23,0.06) !important;
        border-color: rgba(37,99,235,0.35) !important;
        background: rgba(246,248,252,0.96) !important;
      }
      div[data-testid="stButton"] > button.modbtn.active {
        border-color: rgba(37,99,235,0.60) !important;
        background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(59,130,246,0.14)) !important;
        box-shadow: 0 12px 26px rgba(37,99,235,0.16) !important;
        color: #1D4ED8 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 VAEjMLP + latent SHAP 生物标志物分析平台")
st.caption("Latent 表征学习 → 解释性（SHAP）→ 稳定性评估 → 功能富集（按钮切换+气泡图）→ 差异分析（含箱线图）→ 聚类与生存验证")


# =====================================================
# Sidebar: Demo gate + Clear cache
# =====================================================
with st.sidebar:
    st.header("示例数据")
    use_demo_gate = st.checkbox("使用示例数据（Demo）", value=False)
    st.divider()
    st.header("缓存管理")
    if st.button("🧹 清除缓存 / 清除结果", use_container_width=True):
        clear_results_cache()
        st.success("已清除运行结果与下载缓存。")
        st.rerun()
    st.caption("说明：清除不会影响页面开关，只会清掉已运行结果与下载文件缓存。")

if not use_demo_gate:
    st.markdown(
        """
        <div class="card">
          <div class="heroTitle">整体工作介绍</div>
          <div class="smallMuted">
            本工具面向 RNA-seq 表达矩阵（genes×samples）与二分类标签，训练 <b>VAE + MLP</b> 学习 latent 表征；
            使用 <b>latent SHAP</b> 解释模型决策并映射回基因层形成候选 biomarkers；
            支持多次运行做稳定性评估（频率/CV）；并对 Top20 做富集、差异（含箱线图）、聚类及生存验证。
          </div>
          <div class="smallMuted" style="margin-top:10px;">👉 请在左侧打开「使用示例数据（Demo）」进入完整页面。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# =====================================================
# Sidebar: parameters
# =====================================================
with st.sidebar:
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
# Hero section: Input / Workflow / Output
# =====================================================
st.markdown('<div class="heroWrap">', unsafe_allow_html=True)
st.markdown('<div class="heroTitle">一站式 Biomarker 发现与验证</div>', unsafe_allow_html=True)
st.markdown('<div class="heroSub">从表达矩阵到可解释性、稳定性与验证分析（富集/差异/聚类/生存）</div>', unsafe_allow_html=True)

hc1, hc2, hc3 = st.columns(3)
with hc1:
    st.markdown(
        """
        <div class="card">
          <b>Input</b>
          <div class="smallMuted" style="margin-top:8px;">
            <ul>
              <li>RNA: genes × samples（CSV）</li>
              <li>Labels: Sample / Label（CSV）</li>
              <li>Survival: Sample / Time / Event（可选）</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown(
        """
        <div class="card">
          <b>Workflow</b>
          <div class="smallMuted" style="margin-top:8px;">
            VAE 压缩 → MLP 分类 → latent SHAP → 映射回基因 → 多次运行稳定性
          </div>
          <div class="smallMuted" style="margin-top:10px;">
            下游：GO/KEGG、差异（含箱线图）、Top20 聚类、生存验证
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hc3:
    st.markdown(
        """
        <div class="card">
          <b>Output</b>
          <div class="smallMuted" style="margin-top:8px;">
            <ul>
              <li>性能指标（AUC/Acc/Prec/Recall）</li>
              <li>Top biomarkers + 稳定性（Freq/CV）</li>
              <li>SHAP 图、富集/差异/聚类/生存</li>
              <li>CSV/PNG/ZIP + REPORT.md</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Uploaders + RUN (Anchor: run)
# =====================================================
st.markdown('<div id="run"></div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 数据输入")

u1, u2, u3 = st.columns(3)
with u1:
    rna_file = st.file_uploader("RNA-seq（genes×samples）CSV（可选）", type="csv", key="rna_uploader")
with u2:
    label_file = st.file_uploader("Labels CSV（Sample/Label，可选）", type="csv", key="label_uploader")
with u3:
    surv_file = st.file_uploader("Survival CSV（Sample,Time,Event，可选）", type="csv", key="surv_uploader")

use_demo_data = st.checkbox("本次运行使用示例数据（忽略上传）", value=True)
run_button = st.button("🚀 运行主流程", type="primary")
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Main pipeline
# =====================================================
if run_button:
    st.session_state["cache_params"] = {
        "latent_dim": int(latent_dim),
        "epochs": int(n_epochs),
        "lr": float(lr),
        "ce_weight": float(ce_weight),
        "test_size": float(test_size),
        "n_runs": int(n_runs),
        "top_k": int(top_k),
        "seed_base": int(seed_base),
        "background_n": int(background_n),
        "shap_nsamples": int(shap_nsamples),
        "cluster_k": int(cluster_k),
        "cox_penalizer": float(cox_penalizer),
    }

    if use_demo_data:
        rna_path = os.path.join(DEMO_DIR, DEMO_RNA)
        lab_path = os.path.join(DEMO_DIR, DEMO_LAB)
        sur_path = os.path.join(DEMO_DIR, DEMO_SUR)

        if (not os.path.exists(rna_path)) or (not os.path.exists(lab_path)):
            st.error(
                "示例数据文件不存在。请把示例文件放在 app.py 同目录：\n"
                f"- {rna_path}\n- {lab_path}\n- {sur_path}（可选）"
            )
            st.stop()

        with st.spinner("读取示例数据中..."):
            rna_raw = read_csv_path_cached(rna_path)
            labels_raw = read_csv_path_cached(lab_path)

        st.session_state["cache_demo_surv_raw"] = read_csv_path_cached(sur_path) if os.path.exists(sur_path) else None
        st.session_state["cache_data_source"] = "Demo"
    else:
        st.session_state["cache_demo_surv_raw"] = None
        if rna_file is None or label_file is None:
            st.error("未选择示例数据时，必须上传 RNA 与 Label。")
            st.stop()
        with st.spinner("读取上传数据中..."):
            rna_raw = read_csv_cached(rna_file)
            labels_raw = read_csv_cached(label_file)
        st.session_state["cache_data_source"] = "Upload"

    with st.spinner("对齐样本与列识别中..."):
        try:
            rna, labels, align_info, label_detect = align_rna_labels(rna_raw, labels_raw)
        except Exception as e:
            st.error(f"数据对齐失败：{e}")
            st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ 输入检查与对齐信息")
    st.write({
        "样本对齐": align_info,
        "对齐后 RNA 维度 (genes×samples)": f"{rna.shape[0]} × {rna.shape[1]}",
    })
    if align_info.get("took_intersection", False):
        st.warning("RNA 与 Labels 样本不完全一致：已自动取交集并按 RNA 列顺序对齐。")
    st.markdown("</div>", unsafe_allow_html=True)

    genes = rna.index.astype(str).tolist()
    y = labels["Label"].values
    X = MinMaxScaler().fit_transform(rna.T.values)

    st.session_state["cache_artifacts"] = {}
    st.session_state["cache_fig_pngs"] = {}

    all_importances, topk_lists, metrics_runs = [], [], []
    last_shap_z, last_z_test, last_latent_df = None, None, None

    prog = st.progress(0)
    status = st.empty()

    with st.spinner("训练与 SHAP 计算中..."):
        for run_i in range(int(n_runs)):
            seed = int(seed_base + run_i)
            set_seed(seed)
            status.write(f"Run {run_i+1}/{n_runs} | seed={seed}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=seed,
                stratify=y if len(pd.Series(y).unique()) == 2 else None
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
            metrics_runs.append({
                "run": run_i, "seed": seed,
                "AUC": auc,
                "Accuracy": accuracy_score(y_test, y_hat),
                "Precision": precision_score(y_test, y_hat, zero_division=0),
                "Recall": recall_score(y_test, y_hat, zero_division=0),
            })

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
            gene_importance = {gene: float(np.mean(np.abs(W_gene_hidden[:, i])) * scale) for i, gene in enumerate(genes)}
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

    stability_df = pd.DataFrame({
        "Gene": genes,
        "MeanImportance": mean_imp.values,
        "StdImportance": std_imp.values,
        "CV": cv_imp.values,
        freq_col: freq.values,
    }).sort_values([freq_col, "MeanImportance"], ascending=[False, False]).reset_index(drop=True)

    top20_genes = stability_df.sort_values("MeanImportance", ascending=False)["Gene"].head(20).tolist()

    # cache
    st.session_state["cache_rna"] = rna
    st.session_state["cache_labels"] = labels
    st.session_state["cache_top20_genes"] = top20_genes
    st.session_state["cache_metrics_df"] = metrics_df
    st.session_state["cache_summary_df"] = summary_df
    st.session_state["cache_stability_df"] = stability_df
    st.session_state["cache_latent_df"] = last_latent_df
    st.session_state["cache_last_shap_z"] = last_shap_z
    st.session_state["cache_last_z_test"] = last_z_test
    st.session_state["cache_cached_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # artifacts
    artifact_put_df_csv("model_metrics_all_runs.csv", metrics_df, note="每次 run 的指标")
    artifact_put_df_csv("model_metrics_summary.csv", summary_df, note="指标均值±标准差")
    artifact_put_df_csv("latent_shap_gene_importance_stability.csv", stability_df, note="基因稳定性（Mean/CV/Freq）")
    if isinstance(last_latent_df, pd.DataFrame):
        artifact_put_df_csv("latent_mean_abs_shap.csv", last_latent_df, note="latent 维度 MeanAbsSHAP")

    try:
        if last_shap_z is not None and last_z_test is not None:
            fig1 = plt.figure(figsize=(9.5, 5.5))
            shap.summary_plot(last_shap_z, features=last_z_test, show=False)
            artifact_put_fig_png("shap_summary_dot.png", fig1, note="SHAP summary dot")
            plt.close(fig1)

            fig2 = plt.figure(figsize=(9.5, 5.0))
            shap.summary_plot(last_shap_z, features=last_z_test, plot_type="bar", show=False)
            artifact_put_fig_png("shap_summary_bar.png", fig2, note="SHAP summary bar")
            plt.close(fig2)
    except Exception:
        pass

    # clear downstream caches
    for k in [
        "cache_enrich_go_kegg",
        "cache_enrich_lib_idx",
        "cache_de_df",
        "cache_de_groups",
        "cache_cluster_df",
        "cache_cluster_X_scaled",
        "cache_cox_cluster_summary",
    ]:
        st.session_state.pop(k, None)

    # default page
    st.session_state["page"] = "主流程"
    st.success("✅ 主流程运行完成：结果已缓存（切换页面/下载不会丢失）。")
    st.rerun()


# =====================================================
# KPI cards
# =====================================================
if "cache_metrics_df" in st.session_state:
    sdf = st.session_state["cache_summary_df"]
    auc_mean = float(sdf.loc[sdf["Metric"] == "AUC", "Mean"].values[0]) if "AUC" in sdf["Metric"].values else np.nan
    acc_mean = float(sdf.loc[sdf["Metric"] == "Accuracy", "Mean"].values[0]) if "Accuracy" in sdf["Metric"].values else np.nan
    top20 = st.session_state.get("cache_top20_genes", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi"><div class="label">Runs</div><div class="value">{len(st.session_state["cache_metrics_df"])}</div><div class="hint">重复训练次数</div></div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="kpi"><div class="label">AUC (mean)</div><div class="value">{auc_mean:.3f}</div><div class="hint">测试集平均 AUC</div></div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="kpi"><div class="label">Accuracy (mean)</div><div class="value">{acc_mean:.3f}</div><div class="hint">测试集平均准确率</div></div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class="kpi"><div class="label">Top biomarkers</div><div class="value">{len(top20)}</div><div class="hint">用于下游分析</div></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Button Navigation (replaces Tabs) — NO CHECKMARK
# =====================================================
PAGES = ["主流程", "下载中心", "功能富集", "差异分析", "聚类&生存"]
if "page" not in st.session_state:
    st.session_state["page"] = "主流程"

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 模块导航（按钮）")

cols = st.columns([1, 1, 1, 1, 1])
for i, p in enumerate(PAGES):
    with cols[i]:
        is_active = (st.session_state["page"] == p)

        # 关键：不加 ✅，仅用颜色区分
        btn_label = p

        # 用 key 区分；并用 JS 给当前按钮加 class active（CSS 控制明显颜色）
        clicked = st.button(btn_label, use_container_width=True, key=f"nav_{p}")
        if clicked:
            st.session_state["page"] = p
            st.rerun()

        # 给渲染出来的“最后一个按钮”打标签不太可靠，采用 JS：按文本匹配并给 active 加 class
        # 这段每次都会跑一遍，确保当前选中态生效
        if is_active:
            components.html(
                f"""
                <script>
                  const btns = parent.document.querySelectorAll('button');
                  btns.forEach(b => {{
                    if (b.innerText.trim() === "{p}") {{
                      b.classList.add("modbtn");
                      b.classList.add("active");
                    }} else if (b.classList.contains("modbtn")) {{
                      // 只移除 modbtn 的 active，不影响其他按钮
                      b.classList.remove("active");
                    }}
                  }});
                </script>
                """,
                height=0,
            )
        else:
            # ensure modbtn class exists for consistent style
            components.html(
                f"""
                <script>
                  const btns = parent.document.querySelectorAll('button');
                  btns.forEach(b => {{
                    if (b.innerText.trim() === "{p}") {{
                      b.classList.add("modbtn");
                    }}
                  }});
                </script>
                """,
                height=0,
            )

st.markdown("</div>", unsafe_allow_html=True)


def _need_run():
    st.info("请先运行主流程（点击上方 🚀 运行主流程）。")


# =====================================================
# Render pages
# =====================================================
def render_main():
    st.markdown('<div id="main"></div>', unsafe_allow_html=True)
    st.subheader("① 主流程（性能 / 稳定性 / SHAP）")
    if "cache_stability_df" not in st.session_state:
        _need_run()
        return

    metrics_df = st.session_state["cache_metrics_df"]
    summary_df = st.session_state["cache_summary_df"]
    stability_df = st.session_state["cache_stability_df"]
    top20 = st.session_state["cache_top20_genes"]
    last_shap_z = st.session_state.get("cache_last_shap_z", None)
    last_z_test = st.session_state.get("cache_last_z_test", None)
    latent_df = st.session_state.get("cache_latent_df", None)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 模型性能（每次 run）")
    st.dataframe(metrics_df, use_container_width=True, height=260)
    st.markdown("#### 📊 指标汇总（均值±标准差）")
    st.dataframe(summary_df, use_container_width=True, height=210)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧬 Top20 候选生物标志物（MeanImportance）")
    st.code("\n".join(top20))
    st.markdown("#### 📌 稳定性（Frequency / CV）Top50")
    st.dataframe(stability_df.head(50), use_container_width=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    if last_shap_z is not None and last_z_test is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Latent SHAP Summary（dot / bar）")
        fig1 = plt.figure(figsize=(9.5, 5.5))
        shap.summary_plot(last_shap_z, features=last_z_test, show=False)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(9.5, 5.0))
        shap.summary_plot(last_shap_z, features=last_z_test, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    if isinstance(latent_df, pd.DataFrame):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Top latent 维度（MeanAbsSHAP）")
        st.dataframe(latent_df.head(20), use_container_width=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)


def render_download():
    st.markdown('<div id="download"></div>', unsafe_allow_html=True)
    st.subheader("② 下载中心（文件列表 / 单文件 / ZIP + REPORT）")
    if "cache_stability_df" not in st.session_state:
        _need_run()
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📦 结果文件列表（当前 session）")
    st.dataframe(artifact_table_df(), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ⬇ 单文件下载")
    cols = st.columns(3)
    artifacts = st.session_state.get("cache_artifacts", {})
    if artifacts:
        names = list(artifacts.keys())
        for i, name in enumerate(names):
            meta = artifacts[name]
            with cols[i % 3]:
                st.download_button(f"⬇ {name}", meta["bytes"], name, mime=meta["mime"])
    else:
        st.info("当前没有可下载文件。")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧾 ZIP 打包（含 REPORT.md）")
    ts = now_stamp()
    zip_name = f"results_{ts}.zip"
    zip_bytes = build_results_zip(ts)
    st.download_button("⬇ 下载 ZIP（results_*.zip）", zip_bytes, zip_name, mime="application/zip")
    with st.expander("预览 REPORT.md（会包含在 ZIP）"):
        st.markdown(build_report_md())
    st.markdown("</div>", unsafe_allow_html=True)


def render_enrich():
    st.markdown('<div id="enrich"></div>', unsafe_allow_html=True)
    st.subheader("③ GO / KEGG 富集分析（Top20，按钮切换 + 气泡图）")
    if "cache_top20_genes" not in st.session_state:
        _need_run()
        return

    top_genes = st.session_state["cache_top20_genes"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 输入基因（Top20）")
    st.code("\n".join(top_genes))
    org = st.selectbox("物种（Enrichr organism）", ["Human", "Mouse"], index=0)
    top_n = st.slider("展示条目数（Top N）", 5, 50, 20, 5)
    st.markdown('<div class="smallMuted">依赖：gseapy + 网络访问 Enrichr。</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not GSEAPY_OK:
        st.warning("未安装 gseapy，无法做 GO/KEGG：pip install gseapy")
        return

    if st.button("🧪 运行 GO/KEGG（Enrichr）", type="primary"):
        with st.spinner("富集分析运行中（需要联网访问 Enrichr）..."):
            try:
                res_dict = run_enrichr(top_genes, organism=org)
                st.session_state["cache_enrich_go_kegg"] = res_dict
                st.session_state["cache_enrich_lib_idx"] = 0
                for lib, df in res_dict.items():
                    artifact_put_df_csv(f"enrichr_{lib}.csv", df, note=f"Enrichr: {lib}")
                st.success("富集完成 ✅")
            except Exception as e:
                st.error(f"富集失败：{e}")
                return

    if "cache_enrich_go_kegg" not in st.session_state:
        st.info("点击上面的「运行 GO/KEGG」生成结果。")
        return

    res_dict = st.session_state["cache_enrich_go_kegg"]
    libs = list(res_dict.keys())
    if "cache_enrich_lib_idx" not in st.session_state:
        st.session_state["cache_enrich_lib_idx"] = 0

    idx = int(st.session_state["cache_enrich_lib_idx"])
    idx = max(0, min(idx, len(libs) - 1))
    st.session_state["cache_enrich_lib_idx"] = idx

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅ 上一个结果", use_container_width=True, disabled=(idx == 0)):
            st.session_state["cache_enrich_lib_idx"] = idx - 1
            st.rerun()
    with c2:
        st.markdown(f"#### 当前结果：**{libs[idx]}**  （{idx+1}/{len(libs)}）")
    with c3:
        if st.button("下一个结果 ➡", use_container_width=True, disabled=(idx == len(libs) - 1)):
            st.session_state["cache_enrich_lib_idx"] = idx + 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    lib = libs[idx]
    df = res_dict[lib].copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 富集结果表（Top）")
    adjp_col = _pick_col(df, ["Adjusted P-value", "Adjusted P-value "])
    p_col = _pick_col(df, ["P-value", "p_value", "p-value"])
    if adjp_col is not None:
        df_show = df.sort_values(adjp_col, ascending=True).head(int(top_n))
    elif p_col is not None:
        df_show = df.sort_values(p_col, ascending=True).head(int(top_n))
    else:
        df_show = df.head(int(top_n))
    st.dataframe(df_show, use_container_width=True, height=360)

    st.markdown("#### 气泡图（Top）")
    try:
        figb = plot_enrich_bubble(df, title=f"Enrichr Bubble: {lib}", top_n=int(top_n))
        st.pyplot(figb)
        artifact_put_fig_png(f"enrich_bubble_{lib}.png", figb, note=f"Bubble plot: {lib}")
        plt.close(figb)
    except Exception as e:
        st.error(f"气泡图绘制失败：{e}")

    st.download_button(f"⬇ 下载 {lib} CSV", df.to_csv(index=False).encode("utf-8"), f"enrichr_{lib}.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


def render_de():
    st.markdown('<div id="de"></div>', unsafe_allow_html=True)
    st.subheader("④ 差异分析（Top20 + 火山图 + 箱线图）")
    if "cache_rna" not in st.session_state or "cache_labels" not in st.session_state:
        _need_run()
        return

    rna = st.session_state["cache_rna"]
    labels = st.session_state["cache_labels"]
    top_genes = st.session_state["cache_top20_genes"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 依赖检查")
    st.write({"scipy": SCIPY_OK, "statsmodels": STATSMODELS_OK})
    st.markdown("</div>", unsafe_allow_html=True)

    if not (SCIPY_OK and STATSMODELS_OK):
        st.warning("差异分析需要：pip install scipy statsmodels")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 运行与结果")
    if st.button("🧬 运行 Top20 差异分析（t-test + FDR）", type="primary"):
        with st.spinner("差异分析计算中..."):
            try:
                de_df, groups = compute_de_top_genes(rna, labels, top_genes)
                st.session_state["cache_de_df"] = de_df
                st.session_state["cache_de_groups"] = groups
                artifact_put_df_csv("top20_differential_expression.csv", de_df, note="Top20 DE (t-test+FDR)")
                st.success("差异分析完成 ✅")
            except Exception as e:
                st.error(f"差异分析失败：{e}")
    st.markdown("</div>", unsafe_allow_html=True)

    if "cache_de_df" in st.session_state:
        de_df = st.session_state["cache_de_df"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 差异结果（Top20）")
        st.dataframe(de_df, use_container_width=True, height=360)
        st.download_button("⬇ 下载差异结果", de_df.to_csv(index=False).encode("utf-8"),
                           "top20_differential_expression.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 火山图（Top20）")
        figv = plt.figure(figsize=(7.8, 4.6))
        x = de_df["log2FC"].values
        yv = -np.log10(de_df["p_value"].values + 1e-300)
        plt.scatter(x, yv)
        for _, row in de_df.iterrows():
            plt.text(row["log2FC"], -np.log10(row["p_value"] + 1e-300), row["Gene"], fontsize=8)
        plt.xlabel("log2FC")
        plt.ylabel("-log10(p)")
        plt.title("Volcano (Top20)")
        plt.tight_layout()
        st.pyplot(figv)
        artifact_put_fig_png("de_volcano_top20.png", figv, note="Volcano plot (Top20)")
        plt.close(figv)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📦 Top 基因表达箱线图（按 Label 分组）")
    default_sel = top_genes[:6]
    sel_genes = st.multiselect("选择要画的基因（建议 3~10 个更清晰）", options=top_genes, default=default_sel)
    save_png = st.checkbox("把箱线图加入下载中心（PNG）", value=True)
    if st.button("📈 生成箱线图", type="primary"):
        try:
            fig_box = plot_gene_boxplots(rna, labels, sel_genes)
            st.pyplot(fig_box)
            if save_png:
                artifact_put_fig_png("top_genes_boxplots.png", fig_box, note="Boxplots by Label (selected genes)")
            plt.close(fig_box)
        except Exception as e:
            st.error(f"箱线图生成失败：{e}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_survival():
    st.markdown('<div id="survival"></div>', unsafe_allow_html=True)
    st.subheader("⑤ 聚类（Top20）+ 生存（KM/Cox）")
    if "cache_rna" not in st.session_state or "cache_top20_genes" not in st.session_state:
        _need_run()
        return

    rna = st.session_state["cache_rna"]
    top_genes = st.session_state["cache_top20_genes"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 聚类输入（Top20）")
    st.code("\n".join(top_genes))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("🧩 运行聚类（Top20 基因表达）", type="primary"):
        with st.spinner("聚类中..."):
            try:
                cluster_df, X_scaled_df = cluster_samples_by_top_genes(
                    rna=rna, top_genes=top_genes, n_clusters=int(cluster_k), seed=int(seed_base)
                )
                st.session_state["cache_cluster_df"] = cluster_df
                st.session_state["cache_cluster_X_scaled"] = X_scaled_df
                artifact_put_df_csv("top20_cluster_labels.csv", cluster_df, note="KMeans clusters by Top20")
                st.success("聚类完成 ✅")
            except Exception as e:
                st.error(f"聚类失败：{e}")
    st.markdown("</div>", unsafe_allow_html=True)

    if "cache_cluster_df" not in st.session_state:
        st.info("先点击「运行聚类」生成 Cluster 标签，然后再做生存分析。")
        return

    cluster_df = st.session_state["cache_cluster_df"]
    X_scaled_df = st.session_state.get("cache_cluster_X_scaled", None)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 聚类结果（Sample → Cluster）")
    st.dataframe(cluster_df.head(100), use_container_width=True, height=340)
    st.download_button("⬇ 下载聚类结果", cluster_df.to_csv(index=False).encode("utf-8"),
                       "top20_cluster_labels.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    if X_scaled_df is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 热图（z-score，按 Cluster 排序）")
        df_plot = X_scaled_df.copy()
        df_plot["Cluster"] = cluster_df.set_index("Sample").loc[df_plot.index]["Cluster"].values
        df_plot = df_plot.sort_values("Cluster")
        mat = df_plot.drop(columns=["Cluster"]).values

        fig_h = plt.figure(figsize=(10.0, 5.0))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="z-score")
        plt.yticks([])
        plt.xticks(range(df_plot.shape[1] - 1), df_plot.drop(columns=["Cluster"]).columns, rotation=90, fontsize=7)
        plt.title("Top20 genes (z-score) sorted by Cluster")
        plt.tight_layout()
        st.pyplot(fig_h)
        artifact_put_fig_png("cluster_heatmap_top20.png", fig_h, note="Heatmap by cluster")
        plt.close(fig_h)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 生存分析（按 Cluster 分组）")

    if not LIFELINES_OK:
        st.warning("未安装 lifelines：pip install lifelines")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    surv_df_source = read_csv_cached(surv_file) if surv_file is not None else st.session_state.get("cache_demo_surv_raw", None)
    if surv_df_source is None:
        st.info("未提供生存数据（上传或示例 sur.csv），跳过 KM/Cox。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    surv = clean_columns(surv_df_source.copy())
    if "Sample" not in surv.columns:
        surv_cols = {_norm(c): c for c in surv.columns}
        for a in ["sample", "sampleid", "id", "subject", "patient"]:
            if _norm(a) in surv_cols:
                surv = surv.rename(columns={surv_cols[_norm(a)]: "Sample"})
                break

    if "Sample" not in surv.columns or "Time" not in surv.columns or "Event" not in surv.columns:
        st.error("生存数据必须包含列：Sample, Time, Event")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    surv["Sample"] = surv["Sample"].astype(str).str.strip()
    surv = surv.set_index("Sample")

    cl = cluster_df.copy()
    cl["Sample"] = cl["Sample"].astype(str).str.strip()
    cl = cl.set_index("Sample")

    common = sorted(list(set(cl.index).intersection(set(surv.index))))
    if len(common) < 10:
        st.error("生存数据与聚类样本交集太少（<10），无法生存分析。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    surv_aligned = surv.loc[common].copy()
    surv_aligned["Cluster"] = cl.loc[common]["Cluster"].astype(int).values
    surv_aligned["Time"] = pd.to_numeric(surv_aligned["Time"], errors="coerce")
    surv_aligned["Event"] = pd.to_numeric(surv_aligned["Event"], errors="coerce")
    surv_aligned = surv_aligned.dropna(subset=["Time", "Event", "Cluster"])

    fig_km, p_lr = km_plot_by_group(surv_aligned, group_col="Cluster")
    st.pyplot(fig_km)
    artifact_put_fig_png("survival_km_by_cluster.png", fig_km, note="KM curves by cluster")
    plt.close(fig_km)
    if p_lr is not None:
        st.write({"Log-rank p-value (2 groups)": p_lr})

    st.markdown("##### Cox（Cluster 作为协变量）")
    df_cox = surv_aligned[["Time", "Event", "Cluster"]].copy().reset_index(drop=True)
    df_cox = pd.get_dummies(df_cox, columns=["Cluster"], drop_first=True)

    df_train, df_test = train_test_split(df_cox, test_size=float(test_size), random_state=int(seed_base))
    cph = CoxPHFitter(penalizer=float(cox_penalizer))
    cph.fit(df_train, duration_col="Time", event_col="Event")

    risk = cph.predict_partial_hazard(df_test)
    c_index = concordance_index(df_test["Time"], -risk.values, df_test["Event"])
    st.write({"C-index": float(c_index)})

    cox_sum = cph.summary.reset_index()
    st.session_state["cache_cox_cluster_summary"] = cox_sum
    artifact_put_df_csv("cox_cluster_summary.csv", cox_sum, note="Cox summary (Cluster)")
    st.dataframe(cox_sum, use_container_width=True, height=360)
    st.download_button("⬇ 下载 Cox summary（Cluster）",
                       cox_sum.to_csv(index=False).encode("utf-8"),
                       "cox_cluster_summary.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# Dispatch
# =====================================================
page = st.session_state.get("page", "主流程")
if page == "主流程":
    render_main()
elif page == "下载中心":
    render_download()
elif page == "功能富集":
    render_enrich()
elif page == "差异分析":
    render_de()
elif page == "聚类&生存":
    render_survival()

st.divider()
st.caption(
    "依赖提示：基础功能需 streamlit/pandas/numpy/torch/scikit-learn/shap/matplotlib；"
    "差异分析需 scipy + statsmodels；"
    "GO/KEGG（Enrichr）需 gseapy 且需要网络；"
    "生存分析需 lifelines。"
)

"""
Virtual Skin Platform — Interactive Dashboard
Multi-scale virtual skin model for transdermal drug delivery prediction.
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Virtual Skin Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 3px solid #e94560; padding-bottom: 10px; margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 12px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #16213e;
        border-left: 4px solid #e94560; padding-left: 12px; margin: 25px 0 15px;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f3460 0%, #16213e 100%); }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    div[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all models and data once."""
    import torch
    from virtual_skin.transport.drug_transport import DrugTransportPredictor
    from virtual_skin.transport.state_modulation import StateModulationNetwork
    from virtual_skin.data.drug_properties import DrugLibrary
    from virtual_skin.data.public_datasets.huskindb_scraper import build_expanded_literature_db

    db = build_expanded_literature_db(try_download=False)
    drug_lib = DrugLibrary.default_library()

    net = StateModulationNetwork(n_state_axes=5, n_drug_desc=8, hidden_dims=[64, 32])
    model_path = "results/training/modulation_net.pt"
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))

    predictor = DrugTransportPredictor(modulation_net=net, use_default_physics=True)
    return predictor, drug_lib, db


@st.cache_data
def load_scrna():
    """Load real scRNA-seq data (gracefully returns None if unavailable)."""
    try:
        import scanpy as sc
        path = os.path.join(os.path.dirname(__file__), "data/public/GSE147424_processed.h5ad")
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            return adata
    except ImportError:
        pass
    except Exception:
        pass
    return None


def run_prediction(predictor, drug_desc, tissue_state, t_total_h=24.0):
    from virtual_skin.atlas.state_vector import TissueStateVector
    sv = TissueStateVector(*tissue_state)
    result = predictor.predict(sv, drug_desc, t_total_h=t_total_h)
    return result


# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 Virtual Skin")
    st.markdown("*Multi-scale Transdermal*\n*Drug Delivery Platform*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "💊 Drug Prediction", "🔬 scRNA-seq Explorer",
         "⚗️ Virtual Experiment", "📊 Model Validation"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Powered by Physics-Informed<br>"
        "Neural Networks + scRNA-seq<br>"
        "© 2026 Virtual Skin Lab</small>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">Virtual Skin Platform — Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">106</div>
            <div class="metric-label">Drugs in Database</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">
            <div class="metric-value">39,600</div>
            <div class="metric-label">Single Cells (GSE147424)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#4facfe,#00f2fe)">
            <div class="metric-value">76%</div>
            <div class="metric-label">Within 3-fold (Kp)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card" style="background:linear-gradient(135deg,#43e97b,#38f9d7)">
            <div class="metric-value">3</div>
            <div class="metric-label">Skin Conditions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)

        fig = go.Figure()
        boxes = [
            (0.5, 5, "scRNA-seq\n(39,600 cells)", "#667eea"),
            (3.5, 5, "Spatial\nTranscriptomics", "#764ba2"),
            (2.0, 3.5, "Tissue State\nVector (5-axis)", "#e94560"),
            (0.5, 2, "State Modulation\nNetwork", "#f5576c"),
            (3.5, 2, "Drug Properties\n(MW, logP, ...)", "#4facfe"),
            (2.0, 0.5, "PDE Solver\n(Layered Diffusion)", "#43e97b"),
        ]
        for x, y, text, color in boxes:
            fig.add_shape(type="rect", x0=x-0.8, y0=y-0.5, x1=x+0.8, y1=y+0.5,
                          fillcolor=color, opacity=0.15, line=dict(color=color, width=2))
            fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                               font=dict(size=11, color="#1a1a2e"), align="center")

        arrows = [(0.5,4.5,2.0,4.0), (3.5,4.5,2.0,4.0), (2.0,3.0,0.5,2.5),
                  (2.0,3.0,3.5,2.5), (0.5,1.5,2.0,1.0), (3.5,1.5,2.0,1.0)]
        for x0,y0,x1,y1 in arrows:
            fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0, showarrow=True,
                               arrowhead=2, arrowsize=1.2, arrowcolor="#666")

        fig.update_layout(
            height=400, showlegend=False, plot_bgcolor="white",
            xaxis=dict(range=[-0.5,4.5], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-0.2,5.8], showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Quick Prediction</div>', unsafe_allow_html=True)

        predictor, drug_lib, db = load_models()
        from virtual_skin.data.drug_properties import DrugDescriptor

        quick_drug = st.selectbox("Drug", ["caffeine", "lidocaine", "diclofenac",
                                            "hydrocortisone", "testosterone"], key="quick_drug")
        quick_cond = st.selectbox("Condition", ["Healthy", "AD Lesional", "AD Non-lesional"], key="quick_cond")

        state_map = {
            "Healthy": [0.8, 0.15, 0.2, 0.5, 0.3],
            "AD Lesional": [0.3, 0.7, 0.6, 0.6, 0.4],
            "AD Non-lesional": [0.5, 0.3, 0.4, 0.5, 0.3],
        }

        if st.button("Predict", type="primary", use_container_width=True):
            try:
                drug = drug_lib.get(quick_drug)
            except KeyError:
                drug = DrugDescriptor(name=quick_drug, molecular_weight=194.19, logP=-0.07, solubility_mg_mL=20.0)

            result = run_prediction(predictor, drug, state_map[quick_cond])

            m1, m2, m3 = st.columns(3)
            m1.metric("Jss", f"{result.steady_state_flux:.2f}", "µg/cm²/h")
            m2.metric("Lag Time", f"{result.lag_time:.1f}", "hours")
            m3.metric("Q24", f"{result.cumulative_permeation[-1]:.1f}", "µg/cm²")

            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=result.time_h, y=result.cumulative_permeation,
                mode="lines", line=dict(color="#e94560", width=3),
                fill="tozeroy", fillcolor="rgba(233,69,96,0.1)",
            ))
            fig_q.update_layout(
                height=200, margin=dict(l=40,r=10,t=10,b=30),
                xaxis_title="Time (h)", yaxis_title="Q (µg/cm²)",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_q, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: Drug Prediction
# ════════════════════════════════════════════
elif page == "💊 Drug Prediction":
    st.markdown('<div class="main-header">Drug Permeation Prediction</div>', unsafe_allow_html=True)

    predictor, drug_lib, db = load_models()
    from virtual_skin.data.drug_properties import DrugDescriptor

    col_param, col_result = st.columns([1, 2])

    with col_param:
        st.markdown('<div class="section-header">Tissue State</div>', unsafe_allow_html=True)

        preset = st.selectbox("Preset", ["Custom", "Healthy Skin", "AD Lesional",
                                          "Psoriasis Plaque", "Tape-Stripped"])
        presets = {
            "Healthy Skin": [0.8, 0.15, 0.2, 0.5, 0.3],
            "AD Lesional": [0.3, 0.7, 0.6, 0.6, 0.4],
            "Psoriasis Plaque": [0.4, 0.8, 0.7, 0.7, 0.3],
            "Tape-Stripped": [0.1, 0.2, 0.1, 0.5, 0.5],
            "Custom": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
        defaults = presets.get(preset, presets["Custom"])

        barrier = st.slider("Barrier Integrity", 0.0, 1.0, defaults[0], 0.05, key="b")
        inflam = st.slider("Inflammatory Load", 0.0, 1.0, defaults[1], 0.05, key="i")
        ecm = st.slider("ECM Remodeling", 0.0, 1.0, defaults[2], 0.05, key="e")
        vasc = st.slider("Vascularization", 0.0, 1.0, defaults[3], 0.05, key="v")
        append = st.slider("Appendage Openness", 0.0, 1.0, defaults[4], 0.05, key="a")

        st.markdown('<div class="section-header">Drug</div>', unsafe_allow_html=True)

        drug_mode = st.radio("Input mode", ["Library", "Custom"], horizontal=True)

        if drug_mode == "Library":
            avail = list(drug_lib._drugs.keys()) if hasattr(drug_lib, '_drugs') else ["caffeine"]
            drug_name = st.selectbox("Select drug", sorted(avail))
            try:
                drug = drug_lib.get(drug_name)
            except Exception:
                drug = DrugDescriptor(name=drug_name, molecular_weight=200, logP=1.0, solubility_mg_mL=10.0)
        else:
            custom_name = st.text_input("Drug name", "custom_drug")
            custom_mw = st.number_input("MW (g/mol)", 50.0, 1500.0, 300.0)
            custom_logp = st.number_input("logP", -5.0, 12.0, 2.0)
            custom_sol = st.number_input("Solubility (mg/mL)", 0.001, 1000.0, 10.0)
            drug = DrugDescriptor(name=custom_name, molecular_weight=custom_mw,
                                  logP=custom_logp, solubility_mg_mL=custom_sol)

        t_total = st.slider("Simulation time (h)", 6, 72, 24)

    with col_result:
        if st.button("🔬 Run Simulation", type="primary", use_container_width=True):
            state = [barrier, inflam, ecm, vasc, append]
            result = run_prediction(predictor, drug, state, t_total_h=float(t_total))

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jss (µg/cm²/h)", f"{result.steady_state_flux:.3f}")
            m2.metric("Lag Time (h)", f"{result.lag_time:.1f}")
            q24 = float(result.cumulative_permeation[-1]) if len(result.cumulative_permeation) > 0 else 0
            m3.metric("Q_total (µg/cm²)", f"{q24:.1f}")
            m4.metric("Dermis AUC", f"{result.target_layer_auc:.2f}")

            # Permeation curve
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Cumulative Permeation", "Flux Profile"])

            fig.add_trace(go.Scatter(
                x=result.time_h, y=result.cumulative_permeation,
                mode="lines", line=dict(color="#e94560", width=3),
                fill="tozeroy", fillcolor="rgba(233,69,96,0.1)", name="Q(t)",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=result.time_h, y=result.flux,
                mode="lines", line=dict(color="#4facfe", width=3),
                fill="tozeroy", fillcolor="rgba(79,172,254,0.1)", name="J(t)",
            ), row=1, col=2)

            fig.update_xaxes(title_text="Time (h)", row=1, col=1)
            fig.update_xaxes(title_text="Time (h)", row=1, col=2)
            fig.update_yaxes(title_text="Q (µg/cm²)", row=1, col=1)
            fig.update_yaxes(title_text="J (µg/cm²/h)", row=1, col=2)
            fig.update_layout(height=350, showlegend=False, plot_bgcolor="white",
                              margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

            # Concentration heatmap
            if result.concentration_profile is not None:
                st.markdown('<div class="section-header">Concentration Profile C(x, t)</div>',
                            unsafe_allow_html=True)

                C = result.concentration_profile
                t_idx = np.linspace(0, len(result.time_h)-1, min(50, len(result.time_h))).astype(int)
                C_sub = C[t_idx, :]

                from virtual_skin.transport.layered_diffusion import SkinLayerGeometry
                geom = SkinLayerGeometry()
                x_grid, layers = geom.build_grid()

                fig_heat = go.Figure(data=go.Heatmap(
                    z=np.log10(C_sub + 1e-6),
                    x=x_grid,
                    y=result.time_h[t_idx],
                    colorscale="Viridis",
                    colorbar=dict(title="log₁₀(C)"),
                ))
                fig_heat.add_vline(x=geom.sc_thickness, line_dash="dash", line_color="white",
                                   annotation_text="SC|VE")
                fig_heat.add_vline(x=geom.sc_thickness + geom.ve_thickness, line_dash="dash",
                                   line_color="white", annotation_text="VE|Dermis")
                fig_heat.update_layout(
                    height=300, xaxis_title="Depth (µm)", yaxis_title="Time (h)",
                    plot_bgcolor="white", margin=dict(t=10, b=30),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # State radar
            st.markdown('<div class="section-header">Tissue State Vector</div>', unsafe_allow_html=True)
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=state + [state[0]],
                theta=["Barrier", "Inflammation", "ECM", "Vascularization", "Appendage", "Barrier"],
                fill="toself", fillcolor="rgba(233,69,96,0.2)",
                line=dict(color="#e94560", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=300, showlegend=False, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: scRNA-seq Explorer
# ════════════════════════════════════════════
elif page == "🔬 scRNA-seq Explorer":
    st.markdown('<div class="main-header">scRNA-seq Explorer — GSE147424</div>', unsafe_allow_html=True)

    adata = load_scrna()

    if adata is not None:
        st.success(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        col_umap, col_info = st.columns([2, 1])

        with col_info:
            st.markdown('<div class="section-header">Dataset Info</div>', unsafe_allow_html=True)

            if "condition" in adata.obs.columns:
                cond_counts = adata.obs["condition"].value_counts()
                for cond, cnt in cond_counts.items():
                    st.metric(str(cond).replace("_", " ").title(), f"{cnt:,} cells")

            color_by = st.selectbox("Color UMAP by",
                [c for c in ["condition", "leiden", "sample", "batch", "n_genes"]
                 if c in adata.obs.columns or c in ["n_genes"]])

        with col_umap:
            if "X_umap" in adata.obsm:
                umap = adata.obsm["X_umap"]
                n_show = min(10000, adata.n_obs)
                idx = np.random.choice(adata.n_obs, n_show, replace=False)

                if color_by in adata.obs.columns:
                    colors = adata.obs[color_by].values[idx]
                    if hasattr(colors, 'astype'):
                        colors = colors.astype(str)
                else:
                    colors = np.zeros(n_show)

                fig = px.scatter(
                    x=umap[idx, 0], y=umap[idx, 1], color=colors,
                    labels={"x": "UMAP-1", "y": "UMAP-2", "color": color_by},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    opacity=0.4,
                )
                fig.update_traces(marker=dict(size=3))
                fig.update_layout(
                    height=500, plot_bgcolor="white",
                    legend=dict(font=dict(size=10)),
                    margin=dict(t=10, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Gene expression
        st.markdown('<div class="section-header">Gene Expression</div>', unsafe_allow_html=True)
        gene_input = st.text_input("Enter gene name (e.g. FLG, IL1B, COL1A1)", "FLG")

        if gene_input and gene_input in adata.var_names:
            if "X_umap" in adata.obsm:
                n_show = min(10000, adata.n_obs)
                idx = np.random.choice(adata.n_obs, n_show, replace=False)
                umap = adata.obsm["X_umap"]

                gene_idx = list(adata.var_names).index(gene_input)
                if hasattr(adata.X, 'toarray'):
                    expr = np.array(adata.X[idx, gene_idx].toarray()).flatten()
                else:
                    expr = np.array(adata.X[idx, gene_idx]).flatten()

                fig = px.scatter(
                    x=umap[idx, 0], y=umap[idx, 1], color=expr,
                    labels={"x": "UMAP-1", "y": "UMAP-2", "color": gene_input},
                    color_continuous_scale="Viridis", opacity=0.5,
                )
                fig.update_traces(marker=dict(size=3))
                fig.update_layout(height=450, plot_bgcolor="white", margin=dict(t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)
        elif gene_input:
            st.warning(f"Gene '{gene_input}' not found in dataset")
    else:
        st.warning("scRNA-seq data not loaded. Run `scripts/run_real_data_pipeline.py` first.")


# ════════════════════════════════════════════
# PAGE: Virtual Experiment
# ════════════════════════════════════════════
elif page == "⚗️ Virtual Experiment":
    st.markdown('<div class="main-header">Virtual Experiment — Counterfactual Simulation</div>',
                unsafe_allow_html=True)

    predictor, drug_lib, db = load_models()
    from virtual_skin.data.drug_properties import DrugDescriptor

    st.markdown('<div class="section-header">Experiment Design</div>', unsafe_allow_html=True)

    exp_type = st.radio("Experiment type",
                        ["Barrier Perturbation Sweep", "Drug Comparison", "Condition Comparison"],
                        horizontal=True)

    if exp_type == "Barrier Perturbation Sweep":
        col1, col2 = st.columns(2)
        with col1:
            drug_name = st.selectbox("Drug", ["caffeine", "lidocaine", "diclofenac",
                                               "hydrocortisone", "testosterone"])
        with col2:
            axis = st.selectbox("Perturb axis", ["Barrier Integrity", "Inflammatory Load",
                                                  "ECM Remodeling", "Vascularization"])

        axis_map = {"Barrier Integrity": 0, "Inflammatory Load": 1,
                    "ECM Remodeling": 2, "Vascularization": 3}
        axis_idx = axis_map[axis]

        if st.button("🧪 Run Sweep", type="primary"):
            try:
                drug = drug_lib.get(drug_name)
            except Exception:
                drug = DrugDescriptor(name=drug_name, molecular_weight=194.19, logP=-0.07, solubility_mg_mL=20.0)

            sweep_vals = np.linspace(0.05, 0.95, 15)
            jss_vals, lag_vals, q24_vals = [], [], []

            progress = st.progress(0)
            for i, v in enumerate(sweep_vals):
                state = [0.5, 0.3, 0.3, 0.5, 0.3]
                state[axis_idx] = v
                result = run_prediction(predictor, drug, state)
                jss_vals.append(result.steady_state_flux)
                lag_vals.append(result.lag_time)
                q24_vals.append(float(result.cumulative_permeation[-1]) if len(result.cumulative_permeation) > 0 else 0)
                progress.progress((i + 1) / len(sweep_vals))

            fig = make_subplots(rows=1, cols=3, subplot_titles=["Jss", "Lag Time", "Q24"])

            fig.add_trace(go.Scatter(x=sweep_vals, y=jss_vals, mode="lines+markers",
                                     line=dict(color="#e94560", width=3), marker=dict(size=6)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sweep_vals, y=lag_vals, mode="lines+markers",
                                     line=dict(color="#4facfe", width=3), marker=dict(size=6)), row=1, col=2)
            fig.add_trace(go.Scatter(x=sweep_vals, y=q24_vals, mode="lines+markers",
                                     line=dict(color="#43e97b", width=3), marker=dict(size=6)), row=1, col=3)

            for i in range(1, 4):
                fig.update_xaxes(title_text=axis, row=1, col=i)
            fig.update_yaxes(title_text="µg/cm²/h", row=1, col=1)
            fig.update_yaxes(title_text="hours", row=1, col=2)
            fig.update_yaxes(title_text="µg/cm²", row=1, col=3)
            fig.update_layout(height=400, showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"**Insight**: {axis} sweep from 0.05→0.95 changes Jss by "
                    f"{max(jss_vals)/max(min(jss_vals),1e-6):.1f}× for {drug_name}")

    elif exp_type == "Drug Comparison":
        drugs_to_compare = st.multiselect(
            "Select drugs", ["caffeine", "lidocaine", "diclofenac", "hydrocortisone",
                             "testosterone", "ibuprofen", "nicotine", "fentanyl"],
            default=["caffeine", "lidocaine", "hydrocortisone"],
        )

        if st.button("🧪 Compare", type="primary") and drugs_to_compare:
            state = [0.8, 0.15, 0.2, 0.5, 0.3]
            results = {}

            for dn in drugs_to_compare:
                try:
                    drug = drug_lib.get(dn)
                except Exception:
                    continue
                result = run_prediction(predictor, drug, state)
                results[dn] = result

            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (dn, r) in enumerate(results.items()):
                fig.add_trace(go.Scatter(
                    x=r.time_h, y=r.cumulative_permeation,
                    mode="lines", name=dn,
                    line=dict(color=colors[i % len(colors)], width=3),
                ))

            fig.update_layout(
                height=450, xaxis_title="Time (h)", yaxis_title="Cumulative Permeation (µg/cm²)",
                plot_bgcolor="white", legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            import pandas as pd
            df = pd.DataFrame([
                {"Drug": dn, "Jss (µg/cm²/h)": f"{r.steady_state_flux:.3f}",
                 "Lag (h)": f"{r.lag_time:.1f}",
                 "Q24 (µg/cm²)": f"{float(r.cumulative_permeation[-1]):.1f}"}
                for dn, r in results.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif exp_type == "Condition Comparison":
        drug_name = st.selectbox("Drug", ["caffeine", "lidocaine", "diclofenac",
                                           "hydrocortisone", "testosterone"])

        conditions = {
            "Healthy": [0.8, 0.15, 0.2, 0.5, 0.3],
            "AD Lesional": [0.3, 0.7, 0.6, 0.6, 0.4],
            "AD Non-lesional": [0.5, 0.3, 0.4, 0.5, 0.3],
            "Psoriasis": [0.4, 0.8, 0.7, 0.7, 0.3],
            "Tape-Stripped (10×)": [0.1, 0.2, 0.1, 0.5, 0.5],
        }

        if st.button("🧪 Compare Conditions", type="primary"):
            try:
                drug = drug_lib.get(drug_name)
            except Exception:
                drug = DrugDescriptor(name=drug_name, molecular_weight=194.19, logP=-0.07, solubility_mg_mL=20.0)

            results = {}
            for cname, state in conditions.items():
                r = run_prediction(predictor, drug, state)
                results[cname] = r

            fig = go.Figure()
            colors = ["#2196F3", "#F44336", "#FF9800", "#9C27B0", "#4CAF50"]
            for i, (cn, r) in enumerate(results.items()):
                fig.add_trace(go.Scatter(
                    x=r.time_h, y=r.cumulative_permeation,
                    mode="lines", name=cn,
                    line=dict(color=colors[i], width=3),
                ))

            fig.update_layout(
                height=450, xaxis_title="Time (h)",
                yaxis_title=f"{drug_name} Permeation (µg/cm²)",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart
            jss_vals = {cn: r.steady_state_flux for cn, r in results.items()}
            fig_bar = go.Figure(data=go.Bar(
                x=list(jss_vals.keys()), y=list(jss_vals.values()),
                marker_color=colors[:len(jss_vals)],
                text=[f"{v:.2f}" for v in jss_vals.values()],
                textposition="auto",
            ))
            fig_bar.update_layout(
                height=300, yaxis_title="Jss (µg/cm²/h)", plot_bgcolor="white",
                title=f"{drug_name}: Flux Across Conditions",
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: Model Validation
# ════════════════════════════════════════════
elif page == "📊 Model Validation":
    st.markdown('<div class="main-header">Model Validation — 106 Drug Benchmark</div>', unsafe_allow_html=True)

    predictor, drug_lib, db = load_models()

    # Load validation CSV if available
    val_path = "results/training/validation_results.csv"
    if os.path.exists(val_path):
        import pandas as pd
        df = pd.read_csv(val_path)

        col1, col2, col3 = st.columns(3)
        folds = df["fold_error"].values
        col1.metric("Drugs Tested", len(df))
        col2.metric("Within 3-fold", f"{(folds <= 3).sum()}/{len(df)} ({100*(folds<=3).mean():.0f}%)")
        col3.metric("Median Fold Error", f"{np.median(folds):.2f}×")

        # Scatter plot
        fig = px.scatter(
            df, x="kp_lit", y="kp_pred", color="fold_error",
            hover_name="drug", hover_data=["mw", "logp", "fold_error"],
            color_continuous_scale="RdYlGn_r",
            range_color=[1, 10],
            log_x=True, log_y=True,
            labels={"kp_lit": "Literature Kp (cm/h)", "kp_pred": "Predicted Kp (cm/h)",
                    "fold_error": "Fold Error"},
        )

        vals = list(df["kp_lit"]) + list(df["kp_pred"])
        vals = [v for v in vals if v > 0]
        lo, hi = min(vals) * 0.3, max(vals) * 3
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                  line=dict(dash="dash", color="black", width=1), name="1:1", showlegend=True))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo*3, hi*3], mode="lines",
                                  line=dict(dash="dot", color="red", width=1), name="3-fold", showlegend=True))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo/3, hi/3], mode="lines",
                                  line=dict(dash="dot", color="red", width=1), showlegend=False))

        fig.update_layout(height=500, plot_bgcolor="white")
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")),
                          selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        st.markdown('<div class="section-header">Error Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(df, x="fold_error", nbins=30, color_discrete_sequence=["#4facfe"])
        fig_hist.add_vline(x=3, line_dash="dash", line_color="green", annotation_text="3-fold")
        fig_hist.add_vline(x=10, line_dash="dash", line_color="orange", annotation_text="10-fold")
        fig_hist.update_layout(height=300, xaxis_title="Fold Error", yaxis_title="Count",
                               plot_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Searchable table
        st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
        df_display = df.copy()
        df_display["kp_pred"] = df_display["kp_pred"].apply(lambda x: f"{x:.2e}")
        df_display["kp_lit"] = df_display["kp_lit"].apply(lambda x: f"{x:.2e}")
        df_display["fold_error"] = df_display["fold_error"].apply(lambda x: f"{x:.2f}×")
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)
    else:
        st.warning("No validation results found. Run `scripts/run_training.py` first.")

    # Literature DB summary
    st.markdown('<div class="section-header">Literature Database</div>', unsafe_allow_html=True)
    st.info(f"**{db.n_records}** permeation records across **{len(db.drug_names)}** drugs\n\n"
            f"Sources: Flynn 1990, Potts & Guy 1992, HuskinDB 2020, SkinPiX 2024, Cheruvu et al. 2022")

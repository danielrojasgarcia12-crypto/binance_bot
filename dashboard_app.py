import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# =========================
# CARGA DE DATOS
# =========================
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"


@st.cache_data(show_spinner=True)
def load_all_trades() -> Optional[pd.DataFrame]:
    """
    Carga todos los CSV que empiecen por paper_trades_*.csv en reports/.
    Devuelve un DataFrame con todos los eventos (OPEN, CLOSE, etc).
    """
    if not REPORTS_DIR.exists():
        return None

    files = sorted(REPORTS_DIR.glob("paper_trades_*.csv"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Aseguramos columnas esperadas
            expected_cols = [
                "timestamp",
                "event",
                "symbol",
                "setup",
                "class",
                "entry",
                "sl",
                "tp",
                "exit_price",
                "outcome",
                "pnl_pct",
                "score",
            ]
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                # Saltamos archivos raros
                continue

            # Parseo de timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    full = pd.concat(dfs, ignore_index=True)
    # Limpieza básica
    full["pnl_pct"] = pd.to_numeric(full["pnl_pct"], errors="coerce")
    full["class"] = full["class"].fillna("")
    full["event"] = full["event"].fillna("")
    full["symbol"] = full["symbol"].fillna("")
    full["date"] = full["timestamp"].dt.date

    return full


# =========================
# MÉTRICAS
# =========================
def compute_metrics(df_closed: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Calcula métricas generales y por tipo (SCALP / SWING) para un DF de CLOSE.
    Devuelve:
      - metrics_global: dict con métricas agregadas
      - metrics_by_class: dict { "SCALP": {...}, "SWING": {...} }
    """
    metrics_global = {
        "trades": 0,
        "win_rate": np.nan,
        "avg_win": np.nan,
        "avg_loss": np.nan,
        "expectancy": np.nan,
    }
    metrics_by_class = {}

    if df_closed is None or df_closed.empty:
        return metrics_global, metrics_by_class

    def _compute(df_sub: pd.DataFrame) -> dict:
        out = {
            "trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "expectancy": np.nan,
        }
        if df_sub.empty:
            return out

        out["trades"] = len(df_sub)
        wins = df_sub["pnl_pct"] > 0
        losses = df_sub["pnl_pct"] < 0

        num_wins = wins.sum()
        num_trades = len(df_sub)
        out["win_rate"] = (num_wins / num_trades) * 100 if num_trades > 0 else np.nan

        if wins.any():
            out["avg_win"] = df_sub.loc[wins, "pnl_pct"].mean()
        if losses.any():
            out["avg_loss"] = df_sub.loc[losses, "pnl_pct"].mean()

        # Expectancy = p(win)*avg_win + (1-p(win))*avg_loss
        if num_trades > 0 and (wins.any() or losses.any()):
            p_win = out["win_rate"] / 100.0
            avg_win = out["avg_win"] if not np.isnan(out["avg_win"]) else 0.0
            avg_loss = out["avg_loss"] if not np.isnan(out["avg_loss"]) else 0.0
            out["expectancy"] = p_win * avg_win + (1 - p_win) * avg_loss

        return out

    metrics_global = _compute(df_closed)

    for cls in ["SCALP", "SWING"]:
        sub = df_closed[df_closed["class"].str.upper() == cls]
        metrics_by_class[cls] = _compute(sub)

    return metrics_global, metrics_by_class


# =========================
# UI
# =========================
st.set_page_config(
    page_title="Auditoría Bot Binance - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Dashboard de Auditoría - Sistema Binance (Paper Trading)")
st.caption("Monitoreo de resultados históricos del sistema EMA20_BOUNCE LONG sobre pares USDT.")

# Cargar datos
df_all = load_all_trades()

if df_all is None or df_all.empty:
    st.warning("No se encontraron archivos `paper_trades_*.csv` en la carpeta `reports/`.")
    st.stop()

# Nos quedamos solo con eventos de cierre
df_closed_all = df_all[df_all["event"] == "CLOSE"].copy()
df_closed_all = df_closed_all.dropna(subset=["timestamp"])

if df_closed_all.empty:
    st.warning("Todavía no hay trades cerrados (event = CLOSE) para analizar.")
    st.stop()

# =========================
# SIDEBAR - Filtros
# =========================
st.sidebar.header("⚙️ Filtros")

min_date = df_closed_all["date"].min()
max_date = df_closed_all["date"].max()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

classes_available = sorted(df_closed_all["class"].dropna().unique().tolist())
selected_classes = st.sidebar.multiselect(
    "Tipo de operación",
    options=classes_available,
    default=classes_available,
)

# Aplicar filtros
mask_date = (df_closed_all["date"] >= start_date) & (df_closed_all["date"] <= end_date)
mask_class = df_closed_all["class"].isin(selected_classes)

df_closed = df_closed_all[mask_date & mask_class].copy()

if df_closed.empty:
    st.warning("No hay trades cerrados para el rango de fechas y filtros seleccionados.")
    st.stop()

# =========================
# MÉTRICAS + DIARIA
# =========================
metrics_global, metrics_by_class = compute_metrics(df_closed)

# Serie diaria de % (suma de pnl_pct por día)
daily_pnl = (
    df_closed.groupby("date", as_index=False)["pnl_pct"]
    .sum()
    .rename(columns={"pnl_pct": "pnl_pct_day"})
    .sort_values("date")
)

# =========================
# KPIs PRINCIPALES
# =========================
st.subheader("📌 KPIs del Sistema (según filtros)")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Trades cerrados",
    f"{metrics_global['trades']}",
)

col2.metric(
    "Win rate",
    f"{metrics_global['win_rate']:.2f}%" if not np.isnan(metrics_global["win_rate"]) else "N/A",
)

col3.metric(
    "Ganancia media (wins)",
    f"{metrics_global['avg_win']:.2f}%" if not np.isnan(metrics_global["avg_win"]) else "N/A",
)

col4.metric(
    "Expectancy por trade",
    f"{metrics_global['expectancy']:.2f}%" if not np.isnan(metrics_global["expectancy"]) else "N/A",
)

st.markdown("---")

# =========================
# GRÁFICA % DIARIO
# =========================
st.subheader("📈 Evolución diaria del sistema (suma de % diario)")

chart_df = daily_pnl.set_index("date")
st.line_chart(chart_df["pnl_pct_day"])

st.caption(
    "La gráfica muestra la **suma de pnl_pct por día** (en porcentaje). "
    "No depende de un capital fijo, solo del desempeño relativo del sistema."
)

st.markdown("---")

# =========================
# MÉTRICAS POR TIPO
# =========================
st.subheader("🔍 Detalle por tipo de operación")

col_s, col_w = st.columns(2)

scalp = metrics_by_class.get("SCALP", {})
swing = metrics_by_class.get("SWING", {})

with col_s:
    st.markdown("**SCALP**")
    st.write(f"Trades: {scalp.get('trades', 0)}")
    st.write(
        f"Win rate: {scalp['win_rate']:.2f}%" if not np.isnan(scalp.get("win_rate", np.nan)) else "Win rate: N/A"
    )
    st.write(
        f"Ganancia media: {scalp['avg_win']:.2f}%" if not np.isnan(scalp.get("avg_win", np.nan)) else "Ganancia media: N/A"
    )
    st.write(
        f"Pérdida media: {scalp['avg_loss']:.2f}%" if not np.isnan(scalp.get("avg_loss", np.nan)) else "Pérdida media: N/A"
    )
    st.write(
        f"Expectancy: {scalp['expectancy']:.2f}%" if not np.isnan(scalp.get("expectancy", np.nan)) else "Expectancy: N/A"
    )

with col_w:
    st.markdown("**SWING**")
    st.write(f"Trades: {swing.get('trades', 0)}")
    st.write(
        f"Win rate: {swing['win_rate']:.2f}%" if not np.isnan(swing.get("win_rate", np.nan)) else "Win rate: N/A"
    )
    st.write(
        f"Ganancia media: {swing['avg_win']:.2f}%" if not np.isnan(swing.get("avg_win", np.nan)) else "Ganancia media: N/A"
    )
    st.write(
        f"Pérdida media: {swing['avg_loss']:.2f}%" if not np.isnan(swing.get("avg_loss", np.nan)) else "Pérdida media: N/A"
    )
    st.write(
        f"Expectancy: {swing['expectancy']:.2f}%" if not np.isnan(swing.get("expectancy", np.nan)) else "Expectancy: N/A"
    )

st.markdown("---")

# =========================
# TABLA DE ÚLTIMOS TRADES
# =========================
st.subheader("📋 Últimos trades cerrados")

df_last = df_closed.sort_values("timestamp", ascending=False).copy()
df_last_view = df_last[
    [
        "timestamp",
        "symbol",
        "class",
        "setup",
        "entry",
        "sl",
        "tp",
        "exit_price",
        "outcome",
        "pnl_pct",
        "score",
    ]
].head(50)

st.dataframe(df_last_view, use_container_width=True)

st.caption("Se muestran los últimos 50 trades cerrados según los filtros aplicados.")

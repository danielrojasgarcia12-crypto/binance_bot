# dashboard_app.py
# Dashboard de auditoría del bot + panel IA
#
# - Lee todos los paper_trades_YYYY-MM-DD.csv de la carpeta ./reports
# - Muestra:
#     * Serie diaria de % PnL (suma de pnl_pct por día)
#     * Métricas agregadas (win rate, expectancy, etc.)
#     * Filtros por fecha y tipo (SCALP / SWING)
# - Panel IA:
#     * Envía a un modelo de OpenAI un resumen de resultados + tu pregunta
#     * Devuelve recomendaciones en lenguaje natural
#
# Requisitos:
#   pip install streamlit pandas plotly openai python-dotenv

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# =========================
# CONFIG BÁSICA
# =========================
load_dotenv()

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# =========================
# IMPORT OPENAI (SDK NUEVO)
# =========================
AI_AVAILABLE = False
try:
    from openai import OpenAI

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        AI_AVAILABLE = True
except Exception:
    client = None
    AI_AVAILABLE = False


# =========================
# CARGA DE DATOS
# =========================
@st.cache_data
def load_trades() -> pd.DataFrame:
    if not REPORTS_DIR.exists():
        return pd.DataFrame()

    files = sorted(REPORTS_DIR.glob("paper_trades_*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Limpieza básica
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
    else:
        df["date"] = pd.NaT

    # Asegurar columnas clave
    for col in ["pnl_pct", "class", "symbol", "event", "outcome"]:
        if col not in df.columns:
            df[col] = None

    # Sólo trades CERRADOS
    df_closed = df[df["event"] == "CLOSE"].copy()
    df_closed["pnl_pct"] = pd.to_numeric(df_closed["pnl_pct"], errors="coerce")
    df_closed = df_closed.dropna(subset=["pnl_pct"])

    return df_closed


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for d, grp in df.groupby("date"):
        pnl = grp["pnl_pct"]

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        total_trades = len(pnl)
        num_wins = len(wins)
        num_losses = len(losses)
        wl_trades = num_wins + num_losses

        win_rate = (num_wins / wl_trades * 100) if wl_trades > 0 else 0.0
        avg_win = wins.mean() if num_wins > 0 else 0.0
        avg_loss = losses.mean() if num_losses > 0 else 0.0

        # Expectancy diaria teórica (por trade)
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        # PnL diario en % (suma simple de % por trade)
        daily_pnl_pct = pnl.sum()

        rows.append(
            {
                "date": d,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expectancy": expectancy,
                "daily_pnl_pct": daily_pnl_pct,
            }
        )

    daily = pd.DataFrame(rows).sort_values("date")
    # Serie acumulada en %
    daily["cum_pnl_pct"] = daily["daily_pnl_pct"].cumsum()
    return daily


# =========================
# PANEL IA
# =========================
def build_ai_context(df_all: pd.DataFrame, df_daily: pd.DataFrame, df_filtered: pd.DataFrame) -> str:
    """
    Construye un contexto compacto para darle a la IA:
    - Resumen global (últimos N trades)
    - Resumen diario (últimos M días)
    - Resumen filtrado (lo que el usuario está viendo)
    """
    parts = []

    # Global
    if not df_all.empty:
        total_trades = len(df_all)
        wins = (df_all["pnl_pct"] > 0).sum()
        losses = (df_all["pnl_pct"] < 0).sum()
        wl = wins + losses
        win_rate = (wins / wl * 100) if wl > 0 else 0.0
        avg_win = df_all[df_all["pnl_pct"] > 0]["pnl_pct"].mean() or 0.0
        avg_loss = df_all[df_all["pnl_pct"] < 0]["pnl_pct"].mean() or 0.0
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        parts.append(
            f"RESUMEN GLOBAL:\n"
            f"- Trades cerrados: {total_trades}\n"
            f"- Win rate: {win_rate:.2f}%\n"
            f"- Ganancia media: {avg_win:.2f}%\n"
            f"- Pérdida media: {avg_loss:.2f}%\n"
            f"- Expectancy estimada: {expectancy:.2f}% por trade\n"
        )

    # Últimos días
    if not df_daily.empty:
        last_daily = df_daily.tail(10).copy()
        parts.append("RESUMEN DIARIO (últimos días):")
        for _, r in last_daily.iterrows():
            parts.append(
                f"- {r['date']}: PnL diario {r['daily_pnl_pct']:.2f}%, "
                f"trades={int(r['total_trades'])}, win_rate={r['win_rate']:.1f}%, "
                f"expectancy={r['expectancy']:.2f}%"
            )

    # Filtro actual
    if not df_filtered.empty:
        parts.append("\nRESUMEN FILTRO ACTUAL (lo que ve el usuario):")
        total_trades = len(df_filtered)
        wins = (df_filtered["pnl_pct"] > 0).sum()
        losses = (df_filtered["pnl_pct"] < 0).sum()
        wl = wins + losses
        win_rate = (wins / wl * 100) if wl > 0 else 0.0
        avg_win = df_filtered[df_filtered["pnl_pct"] > 0]["pnl_pct"].mean() or 0.0
        avg_loss = df_filtered[df_filtered["pnl_pct"] < 0]["pnl_pct"].mean() or 0.0
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        parts.append(
            f"- Trades en el filtro: {total_trades}\n"
            f"- Win rate filtro: {win_rate:.2f}%\n"
            f"- Ganancia media filtro: {avg_win:.2f}%\n"
            f"- Pérdida media filtro: {avg_loss:.2f}%\n"
            f"- Expectancy filtro: {expectancy:.2f}% por trade\n"
        )

    return "\n".join(parts)


def ask_ai(question: str, context: str) -> str:
    if not AI_AVAILABLE:
        return (
            "La IA no está disponible.\n\n"
            "Revisa que tengas la variable de entorno OPENAI_API_KEY configurada "
            "y que el paquete 'openai' esté instalado."
        )

    prompt_system = (
        "Eres un analista cuantitativo experto en trading algorítmico.\n"
        "El usuario tiene un sistema de trading que ejecuta trades (SCALP y SWING) "
        "en criptomonedas usando señales técnicas (por ejemplo, EMA20, ATR, etc.).\n"
        "Tu tarea es analizar los resultados históricos (porcentajes de PnL por trade y por día) "
        "y dar recomendaciones prácticas y claras.\n"
        "No hables de capital en dinero, solo de porcentajes, calidad del sistema, "
        "riesgo, filtrado de señales y posibles mejoras.\n"
    )

    full_user = (
        "Estos son los resultados de su sistema (histórico y filtro actual):\n\n"
        f"{context}\n\n"
        "Pregunta del usuario:\n"
        f"{question}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": full_user},
            ],
            temperature=0.25,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al llamar a la IA: {e}"


# =========================
# UI STREAMLIT
# =========================
st.set_page_config(
    page_title="Auditoría Bot Cripto (Paper)",
    layout="wide",
)

st.title("📈 Auditoría del Sistema de Trading (Paper)")

df_all = load_trades()

if df_all.empty:
    st.warning("No se encontraron trades cerrados en la carpeta `reports/`.")
    st.stop()

df_daily_all = compute_daily_stats(df_all)

# --- SIDEBAR: Filtros ---
st.sidebar.header("Filtros")

min_date = df_all["date"].min()
max_date = df_all["date"].max()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

classes = ["SCALP", "SWING"]
selected_classes = st.sidebar.multiselect(
    "Tipo de trade",
    options=classes,
    default=classes,
)

symbols = sorted(df_all["symbol"].dropna().unique().tolist())
selected_symbols = st.sidebar.multiselect(
    "Símbolos",
    options=symbols,
    default=[],
    help="Si lo dejas vacío, usa todos.",
)

# Filtrado
df_filtered = df_all.copy()
df_filtered = df_filtered[
    (df_filtered["date"] >= start_date)
    & (df_filtered["date"] <= end_date)
]

if selected_classes:
    df_filtered = df_filtered[df_filtered["class"].isin(selected_classes)]

if selected_symbols:
    df_filtered = df_filtered[df_filtered["symbol"].isin(selected_symbols)]

df_daily_filtered = compute_daily_stats(df_filtered)

# =========================
# LAYOUT: 2 COLUMNAS
# =========================
col_left, col_right = st.columns([2, 1])

# -------------------------
# IZQUIERDA: MÉTRICAS + GRÁFICAS
# -------------------------
with col_left:
    st.subheader("Resumen de rendimiento (por % diario, sin capital)")

    if df_daily_filtered.empty:
        st.info("No hay trades cerrados en el rango / filtros seleccionados.")
    else:
        # Métricas globales del filtro
        total_trades = len(df_filtered)
        wins = (df_filtered["pnl_pct"] > 0).sum()
        losses = (df_filtered["pnl_pct"] < 0).sum()
        wl = wins + losses
        win_rate = (wins / wl * 100) if wl > 0 else 0.0
        avg_win = df_filtered[df_filtered["pnl_pct"] > 0]["pnl_pct"].mean() or 0.0
        avg_loss = df_filtered[df_filtered["pnl_pct"] < 0]["pnl_pct"].mean() or 0.0
        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Trades cerrados", total_trades)
        col_m2.metric("Win rate", f"{win_rate:.2f}%")
        col_m3.metric("Ganancia media", f"{avg_win:.2f}%")
        col_m4.metric("Pérdida media", f"{avg_loss:.2f}%")

        st.metric("Expectancy (por trade)", f"{expectancy:.2f}%")

        # Gráfica 1: PnL diario (%)
        fig_daily = px.bar(
            df_daily_filtered,
            x="date",
            y="daily_pnl_pct",
            title="PnL diario (suma de % por día)",
            labels={"date": "Fecha", "daily_pnl_pct": "PnL diario (%)"},
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Gráfica 2: PnL acumulado (%)
        fig_cum = px.line(
            df_daily_filtered,
            x="date",
            y="cum_pnl_pct",
            title="PnL acumulado (en % desde el inicio del rango)",
            labels={"date": "Fecha", "cum_pnl_pct": "PnL acumulado (%)"},
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Tabla de trades filtrados
        with st.expander("Ver tabla de trades filtrados"):
            show_cols = [
                "timestamp",
                "symbol",
                "class",
                "pnl_pct",
                "outcome",
                "setup",
                "source_file",
            ]
            cols_present = [c for c in show_cols if c in df_filtered.columns]
            st.dataframe(df_filtered[cols_present].sort_values("timestamp", ascending=False))

# -------------------------
# DERECHA: PANEL IA
# -------------------------
with col_right:
    st.subheader("🤖 Asistente IA para el sistema")

    if not AI_AVAILABLE:
        st.info(
            "Para activar la IA, configura la variable de entorno "
            "`OPENAI_API_KEY` con tu clave de OpenAI y reinicia el servidor.\n\n"
            "Ejemplo en Windows (PowerShell):\n"
            "```powershell\n"
            "$env:OPENAI_API_KEY = \"TU_API_KEY\"\n"
            "streamlit run dashboard_app.py\n"
            "```"
        )
    else:
        st.success("IA conectada correctamente. Puedes hacer preguntas sobre tu sistema.")

        default_question = (
            "¿Qué opinas del rendimiento reciente del sistema con estos resultados? "
            "¿Qué mejorarías primero en cuanto a filtros o gestión de riesgo?"
        )

        question = st.text_area(
            "Escribe tu pregunta sobre el sistema de trading:",
            value=default_question,
            height=150,
        )

        if st.button("👉 Analizar con IA"):
            with st.spinner("Analizando resultados con IA..."):
                context = build_ai_context(df_all, df_daily_all, df_filtered)
                answer = ask_ai(question, context)
            st.markdown("### Respuesta de la IA")
            st.write(answer)

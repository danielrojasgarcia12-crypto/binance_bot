# alerts_runner.py
# ETAPA 9+10 — Paper execution realista + gestión de riesgo (LONG) + reportes
# Mejoras de robustez:
# - Timeouts y reintentos (Binance + Telegram)
# - Watchdog de ciclo (si se pasa del budget, aborta y reintenta en el siguiente ciclo)
# - Mejor manejo de Ctrl+C (stop flag)
# - Evita “ciclos colgados” que impiden escribir reportes
# Phase C:
# - Filtros extra por tipo (SCALP muy estricto, SWING algo más exigente)
# - Filtro de contexto de mercado en H1 (tendencia + volumen + ATR)

import os
import sys
import time
import json
import math
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import requests
from dotenv import load_dotenv
from binance.spot import Spot

# =========================
# ROOT / PATH
# =========================
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports del proyecto
from market.scanner_1h import run_scanner_1h  # ranking (no lo usamos todavía, pero se deja)
from main import scan_15m_for_symbol, CORE    # setups 15m + core list
from utils.indicators import ema              # EMA base (la envolvemos tolerante)

# =========================
# STOP FLAG (Ctrl+C robusto)
# =========================
STOP = {"flag": False}


def _handle_sigint(signum, frame):
    STOP["flag"] = True


signal.signal(signal.SIGINT, _handle_sigint)

# =========================
# ENV
# =========================
load_dotenv()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# =========================
# CONFIG PRINCIPAL
# =========================
CYCLE_MINUTES = 15
MODE = "MIXTO"  # CORE | MIXTO | HIGH

# Universo
MIN_VOL_USDT = 15_000_000  # liquidez mínima absoluta para considerar el símbolo
EXCLUDE_SYMBOLS = {
    "USDCUSDT",
    "FDUSDUSDT",
    "USD1USDT",
    "BFUSDUSDT",
    "TUSDUSDT",
    "USDPUSDT",
    "BUSDUSDT",
}

# Señales / filtros base
MIN_SCORE_TO_ALERT = 55           # score mínimo para considerar una señal
PREALERT_EMA20_DISTANCE_PCT = 0.25
PREALERT_MAX = 3

# --- Phase C: filtros extra por tipo de trade ---
# (NO tocamos MIN_ATR_PCT_TO_TRADE global)

# SWING: más filtrado pero sigue activo
SWING_MIN_SCORE = 60                 # algo más alto que 55
SWING_MIN_VOL_USDT = 10_000_000

# SCALP: ultra-estricto (lo mejor de lo mejor)
SCALP_MIN_SCORE = 75
SCALP_MIN_VOL_USDT = 20_000_000
SCALP_MIN_ATR_PCT = 0.90             # requiere volatilidad fuerte para scalpear

# Filtro de tendencia en H1 (solo LONG)
H1_FAST_EMA = 20
H1_SLOW_EMA = 50

# Paper + gestión
MAX_ALERTS_PER_CYCLE = 5             # máximo de confirmadas por ciclo
TIME_BUDGET_SECONDS = (CYCLE_MINUTES * 60) - 25  # watchdog

# ATR / Confirmaciones
MIN_ATR_PCT_TO_TRADE = 0.10          # tu ATR mínimo global existente (no se toca)

# Trailing
TRAIL_ACTIVATE_AT_TP = True
TRAIL_SL_FACTOR = 0.6                # trailing SL a partir de fracción del recorrido

# Red
BINANCE_TIMEOUT = 10                 # segundos
TELEGRAM_TIMEOUT = 10                # segundos
RETRIES = 3
BACKOFF_BASE = 1.5                   # 1.5, 2.25, 3.37...

# =========================
# DIRECTORIOS / ARCHIVOS
# =========================
STATE_DIR = ROOT / "state"
REPORTS_DIR = ROOT / "reports"
ANALYSIS_DIR = ROOT / "analysis"
STATE_FILE = STATE_DIR / "alerts_state.json"
LOG_FILE = REPORTS_DIR / "alerts.log"

STATE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

# =========================
# HTTP SESSION (keep-alive)
# =========================
SESSION = requests.Session()

# =========================
# HELPERS
# =========================


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{now_ts()}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # no frenar el bot por errores de log
        pass

def is_weekend() -> bool:
    """
    Devuelve True si hoy es sábado o domingo.
    weekday(): Lunes=0 ... Domingo=6
    """
    return datetime.now().weekday() >= 5


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def ensure_daily_report() -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    name = f"paper_trades_{datetime.now().strftime('%Y-%m-%d')}.csv"
    path = REPORTS_DIR / name
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "timestamp,event,symbol,setup,class,entry,sl,tp,exit_price,outcome,pnl_pct,score\n"
            )
    log(f"📄 Reporte diario activo: {path}")
    return path


def write_trade_event(
    report_path: Path,
    event: str,
    trade: Dict[str, Any],
    exit_price: Optional[float] = None,
    outcome: Optional[str] = None,
    pnl_pct: Optional[float] = None,
):
    row = [
        now_ts(),
        event,
        trade.get("symbol", ""),
        trade.get("setup", ""),
        trade.get("class", ""),
        trade.get("entry", ""),
        trade.get("sl", ""),
        trade.get("tp", ""),
        "" if exit_price is None else exit_price,
        "" if outcome is None else outcome,
        "" if pnl_pct is None else pnl_pct,
        trade.get("score", ""),
    ]
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")
    except Exception:
        pass


def retry_call(fn, *args, **kwargs):
    """Reintentos con backoff para llamadas de red."""
    for i in range(RETRIES):
        if STOP["flag"]:
            raise KeyboardInterrupt()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = i == RETRIES - 1
            if last:
                raise
            sleep_s = BACKOFF_BASE ** i
            time.sleep(sleep_s)


def send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    def _post():
        return SESSION.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
    try:
        retry_call(_post)
    except Exception as e:
        log(f"⚠️ Telegram error: {e}")


def make_client() -> Spot:
    # binance-connector Spot soporta timeout=...
    return Spot(timeout=BINANCE_TIMEOUT)


def ema_tolerant(values: List[float], period: int) -> Optional[float]:
    """Evita ValueError si hay pocos datos."""
    if values is None or len(values) < period:
        return None
    try:
        return float(ema(values, period))
    except Exception:
        return None


# =========================
# MARKET DATA (Binance)
# =========================
def fetch_exchange_symbols(client: Spot) -> List[str]:
    info = retry_call(client.exchange_info)
    symbols: List[str] = []
    for s in info.get("symbols", []):
        if s.get("status") == "TRADING":
            sym = s.get("symbol", "")
            if sym.endswith("USDT"):
                symbols.append(sym)
    return symbols


def fetch_24h_stats_map(client: Spot) -> Dict[str, Dict[str, Any]]:
    stats = retry_call(client.ticker_24hr)
    m: Dict[str, Dict[str, Any]] = {}
    for it in stats:
        sym = it.get("symbol")
        if sym:
            m[sym] = it
    return m


def fetch_15m_ohlc(client: Spot, symbol: str, limit: int = 60) -> List[Dict[str, float]]:
    kl = retry_call(client.klines, symbol, "15m", limit=limit)
    out: List[Dict[str, float]] = []
    for k in kl:
        # [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
        out.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
            }
        )
    return out


def fetch_1h_ohlc(client: Spot, symbol: str, limit: int = 80) -> List[Dict[str, float]]:
    """OHLC de 1 hora para filtros de contexto (tendencia)."""
    kl = retry_call(client.klines, symbol, "1h", limit=limit)
    out: List[Dict[str, float]] = []
    for k in kl:
        out.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
            }
        )
    return out


# =========================
# SCORING / CLASIFICACION
# =========================
def log10_safe(x: float) -> float:
    return math.log10(max(1.0, float(x)))


def compute_score(vol_usdt_24h: float, atr_pct: float) -> int:
    """
    Score 0..100: mezcla de liquidez (log) + volatilidad “sana”.
    """
    v = log10_safe(vol_usdt_24h)  # 10M=7, 1B=9, etc.
    vol_component = min(60, max(0, (v - 6.0) * 15))       # 10^6 a 10^10 escala
    atr_component = min(40, max(0, (atr_pct / 2.0) * 40))  # 2% ATR => 40
    return int(round(min(100, vol_component + atr_component)))


def classify_trade(atr_pct: float, score: int) -> str:
    # heurística simple: bajo ATR => swing; alto ATR => scalp
    if atr_pct >= 0.8:
        return "SCALP"
    return "SWING"


def context_filters_ok(
    client: Spot,
    symbol: str,
    vol_24h: float,
    trade_class: str,
    atr_pct: float,
) -> bool:
    """
    Filtro de contexto de mercado (Phase C):
    - Solo long en tendencia alcista en H1 (EMA20 > EMA50 y precio > EMA20)
    - Volumen mínimo según tipo (SCALP más exigente)
    - ATR extra alto para SCALP (sin tocar el ATR base global).
    """

    # ---- Tendencia en H1 (común para SWING y SCALP) ----
    try:
        ohlc_1h = fetch_1h_ohlc(client, symbol, limit=80)
    except Exception:
        # si no podemos obtener H1, preferimos NO operar
        return False

    closes_1h = [c["close"] for c in ohlc_1h]
    ema_fast = ema_tolerant(closes_1h, H1_FAST_EMA)
    ema_slow = ema_tolerant(closes_1h, H1_SLOW_EMA)
    if ema_fast is None or ema_slow is None:
        return False

    price_1h = closes_1h[-1]
    uptrend = price_1h > ema_fast > ema_slow
    if not uptrend:
        # solo queremos longs a favor de tendencia en H1
        return False

    # ---- Filtros específicos por tipo de trade ----
    if trade_class == "SCALP":
        if vol_24h < SCALP_MIN_VOL_USDT:
            return False
        # ATR para scalping mucho más alto que el mínimo global
        if atr_pct < SCALP_MIN_ATR_PCT:
            return False
        if atr_pct > 5.0:
            # evitamos locuras muy volátiles tipo shitcoin
            return False
        return True

    # SWING (por defecto)
    if vol_24h < SWING_MIN_VOL_USDT:
        return False
    # no subo el ATR mínimo para swing: uso el mismo atr_pct que ya pasó MIN_ATR_PCT_TO_TRADE
    if atr_pct > 3.0:
        # swing excesivamente volátil tampoco nos interesa
        return False

    return True


# =========================
# CONFIRMACIONES (LONG)
# =========================
def check_confirmations(client: Spot, symbol: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Confirmación básica:
    - ATR% >= MIN_ATR_PCT_TO_TRADE
    - precio > EMA20
    - última vela verde (close > open)
    """
    ctx: Dict[str, Any] = {}
    ohlc = fetch_15m_ohlc(client, symbol, limit=60)
    closes = [c["close"] for c in ohlc]
    opens = [c["open"] for c in ohlc]

    e20 = ema_tolerant(closes, 20)
    if e20 is None:
        ctx["reason"] = "ema"
        return False, [], ctx

    price = closes[-1]
    last_green = closes[-1] > opens[-1]

    # ATR% aproximado: (media rango verdadero / precio) * 100
    trs: List[float] = []
    prev_close = closes[0]
    for c in ohlc[1:]:
        h, l, cl = c["high"], c["low"], c["close"]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = cl
    atr = sum(trs[-14:]) / min(14, len(trs)) if trs else 0.0
    atr_pct = (atr / price) * 100.0 if price else 0.0

    ctx["ema20"] = e20
    ctx["price"] = price
    ctx["atr_pct"] = atr_pct

    notes: List[str] = []
    if price > e20:
        notes.append("Precio > EMA20")
    else:
        ctx["reason"] = "ema"
        return False, notes, ctx

    if last_green:
        notes.append("Última vela verde")
    else:
        ctx["reason"] = "candle"
        return False, notes, ctx

    if atr_pct >= MIN_ATR_PCT_TO_TRADE:
        notes.append(f"ATR OK ({atr_pct:.2f}%)")
    else:
        ctx["reason"] = "atr"
        return False, notes, ctx

    return True, notes, ctx


# =========================
# SL/TP ejecutable
# =========================
def sl_tp_executable(entry: float, sl: float, tp: float) -> Tuple[bool, str, float, float]:
    if entry <= 0:
        return False, "entry<=0", 0.0, 0.0
    risk = (entry - sl) / entry * 100.0
    reward = (tp - entry) / entry * 100.0
    if risk <= 0 or reward <= 0:
        return False, "risk/reward<=0", risk, reward
    return True, "ok", risk, reward


# =========================
# MENSAJES (simple)
# =========================
def fmt_entry_long(
    sym: str,
    cls: str,
    score: int,
    vol_24h: float,
    entry: float,
    sl: float,
    tp: float,
    atr_pct: float,
    setup_name: str,
    confirm_notes: List[str],
) -> str:
    return (
        "🟢 ENTRADA CONFIRMADA (COMPRA)\n\n"
        f"Moneda: {sym}\n"
        f"Tipo: {cls} | Calidad: {score}/100\n"
        f"Setup: {setup_name}\n"
        f"Liquidez (24h): {vol_24h:,.0f} USDT\n\n"
        f"Entrada: {entry:.8f}\n"
        f"Stop Loss: {sl:.8f}\n"
        f"Objetivo:  {tp:.8f}\n"
        f"ATR (mov. típico): {atr_pct:.2f}%\n\n"
        "✅ Confirmaciones:\n- "
        + "\n- ".join(confirm_notes)
        + "\n\n"
        "👉 Nota: Este es el ÚNICO mensaje que sugiere operar."
    )


def fmt_summary(
    total_usdt: int,
    operables: int,
    elapsed: int,
    top_symbols: List[str],
    diag: Dict[str, Any],
) -> str:
    return (
        "📊 RESUMEN DEL MERCADO (últimos 15 minutos)\n\n"
        f"Mercado analizado: {total_usdt} monedas\n"
        f"Monedas con buena liquidez: {operables}\n"
        f"⏱️ Tiempo de análisis: {elapsed}s\n\n"
        "Estado de señales:\n"
        f"🔎 Setups detectados (crudos): {diag.get('raw_setups', 0)}\n"
        f"🟡 Pasan Score mínimo: {diag.get('pass_score', 0)}\n"
        f"🟢 Entradas confirmadas: {diag.get('confirmed', 0)}\n"
        f"📌 Trades abiertos (paper): {diag.get('open_trades', 0)}\n\n"
        "Mejores monedas por calidad hoy:\n"
        + (" • ".join(top_symbols) if top_symbols else "(sin datos)")
        + "\n\n"
        "👉 Recomendación: Si hay pocas entradas, es intencional: priorizamos calidad."
    )


def fmt_prealert(sym: str, price: float, ema20: float, dist_pct: float) -> str:
    return (
        "👀 MONEDA EN VIGILANCIA\n\n"
        f"Moneda: {sym}\n"
        f"Precio actual: {price:.8f}\n"
        f"Zona clave (EMA20): {ema20:.8f}\n"
        f"Cercanía: {dist_pct:.2f}%\n\n"
        "⛔ No es entrada. Solo observar."
    )


def fmt_trade_closed(
    sym: str, reason: str, entry: float, exit_price: float, pnl: float
) -> str:
    sign = "+" if pnl >= 0 else ""
    return (
        "🏁 TRADE CERRADO (PAPER)\n\n"
        f"Moneda: {sym}\n"
        f"Motivo: {reason}\n"
        f"Entrada: {entry:.8f}\n"
        f"Salida:  {exit_price:.8f}\n"
        f"Resultado: {sign}{pnl:.2f}%"
    )


# =========================
# MAIN LOOP HELPERS
# =========================
def build_universe(
    all_usdt: List[str], stats_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    operables: List[str] = []
    for sym in all_usdt:
        if sym in EXCLUDE_SYMBOLS:
            continue
        vol = float(stats_map.get(sym, {}).get("quoteVolume", 0) or 0)
        if vol >= MIN_VOL_USDT:
            operables.append(sym)

    # top por volumen (para pre-alertas)
    operables_by_vol = sorted(
        operables,
        key=lambda s: float(stats_map.get(s, {}).get("quoteVolume", 0) or 0),
        reverse=True,
    )
    return operables, operables_by_vol


def select_top_symbols(
    stats_map: Dict[str, Dict[str, Any]], operables: List[str]
) -> List[str]:
    # top 3 por volumen (simple y útil)
    top = sorted(
        operables,
        key=lambda s: float(stats_map.get(s, {}).get("quoteVolume", 0) or 0),
        reverse=True,
    )
    return top[:3]


# =========================
# MAIN LOOP
# =========================
def main_loop():
    report_path = ensure_daily_report()
    client = make_client()

    send_telegram(
        "🤖 SISTEMA DE MERCADO INICIADO\n\n"
        "Estoy analizando TODO el mercado USDT.\n"
        "Te avisaré solo cuando haya oportunidades claras.\n\n"
        "📌 Si no llegan alertas, es porque el mercado\n"
        "no ofrece entradas seguras en este momento."
    )

    log("🚀 ETAPA 9+10 iniciada — Paper realista + gestión de riesgo")
    state = load_state()  # trades abiertos + info trailing

    while not STOP["flag"]:
        start_cycle = time.time()
        diag: Dict[str, Any] = {
            "raw_setups": 0,
            "pass_score": 0,
            "confirmed": 0,
            "open_trades": 0,
            "fail_exec": 0,
            "discard_context": 0,
        }

        log("🚀 ETAPA 9+10 iniciada — Paper realista + gestión de riesgo")
        state = load_state()  # trades abiertos + TTL

        while not STOP["flag"]:

            # 🚫 No operar fines de semana
            if is_weekend():
                log("🧘 Fin de semana — el bot no opera sábados ni domingos. Reviso de nuevo en 60 minutos...")
                # Sleep interrumpible 60 minutos
                for _ in range(60 * 60):
                    if STOP["flag"]:
                        raise KeyboardInterrupt()
                    time.sleep(1)
                continue  # vuelve al while y revisa otra vez el día

            start_cycle = time.time()
            diag = {
                "raw_setups": 0,
                "pass_score": 0,
                "confirmed": 0,
                "open_trades": 0,
                "fail_exec": 0,
            }
            ...


        try:
            log("⏱️ Ciclo de análisis iniciado")
            report_path = ensure_daily_report()  # si cambió el día, escribe nuevo

            # Watchdog rápido: si algo raro pasa antes de arrancar, aborta ciclo
            if (time.time() - start_cycle) > TIME_BUDGET_SECONDS:
                raise TimeoutError(
                    "Ciclo excedió TIME_BUDGET_SECONDS antes de empezar (condición anómala)."
                )

            # =========================
            # Universo USDT operable
            # =========================
            all_usdt = fetch_exchange_symbols(client)
            stats_map = fetch_24h_stats_map(client)
            operables, operables_by_vol = build_universe(all_usdt, stats_map)
            top_symbols = select_top_symbols(stats_map, operables)

            # =========================
            # Gestión de trades abiertos (paper)
            # =========================
            closed_keys: List[str] = []
            for key, trade in list(state.items()):
                if STOP["flag"]:
                    raise KeyboardInterrupt()

                sym = trade.get("symbol")
                entry = float(trade.get("entry", 0) or 0)
                if not sym or entry <= 0:
                    closed_keys.append(key)
                    continue

                # precio actual (último close 15m)
                ohlc = fetch_15m_ohlc(client, sym, limit=2)
                price = float(ohlc[-1]["close"])

                tp = float(trade.get("tp", 0) or 0)
                sl = float(trade.get("sl", 0) or 0)

                # Trailing: si activado y toca TP, subimos SL
                if (
                    TRAIL_ACTIVATE_AT_TP
                    and (not trade.get("trailing"))
                    and tp > 0
                    and price >= tp
                ):
                    trade["trailing"] = True
                    new_sl = max(sl, entry * 1.0005)  # +0.05%
                    trade["sl"] = float(new_sl)
                    write_trade_event(report_path, "TP_HIT_TRAIL_ON", trade)
                    send_telegram(
                        f"🔁 TRAILING ACTIVADO (PAPER)\n\n{sym}\nTP tocado. SL sube para proteger ganancia."
                    )

                # Si trailing activo, vamos subiendo SL con el máximo
                if trade.get("trailing"):
                    prev_max = float(trade.get("max_price", entry) or entry)
                    trade["max_price"] = max(prev_max, price)

                    maxp = float(trade["max_price"])
                    trail_sl = entry + (maxp - entry) * TRAIL_SL_FACTOR
                    trade["sl"] = max(float(trade["sl"]), float(trail_sl))

                # Salidas: SL (normal o trailing)
                if price <= float(trade.get("sl", 0) or 0):
                    pnl = ((price - entry) / entry) * 100.0
                    trade["exit_price"] = price
                    trade["outcome"] = "SL" if not trade.get("trailing") else "TRAIL_SL"
                    write_trade_event(
                        report_path,
                        "CLOSE",
                        trade,
                        exit_price=price,
                        outcome=trade["outcome"],
                        pnl_pct=round(pnl, 4),
                    )
                    send_telegram(
                        fmt_trade_closed(
                            sym,
                            "Stop Loss"
                            if trade["outcome"] == "SL"
                            else "Salida por Trailing",
                            entry,
                            price,
                            pnl,
                        )
                    )
                    closed_keys.append(key)

                if (time.time() - start_cycle) > TIME_BUDGET_SECONDS:
                    break

            for k in closed_keys:
                state.pop(k, None)

            # =========================
            # Señales nuevas (market scan)
            # =========================
            alerts_sent = 0
            for sym in operables:
                if STOP["flag"]:
                    raise KeyboardInterrupt()

                # watchdog
                if (time.time() - start_cycle) > TIME_BUDGET_SECONDS:
                    raise TimeoutError(
                        "Watchdog: ciclo excedió TIME_BUDGET_SECONDS durante el escaneo."
                    )

                vol_24h = float(stats_map.get(sym, {}).get("quoteVolume", 0) or 0)

                try:
                    setups = scan_15m_for_symbol(client, sym)
                except Exception:
                    continue

                for s in setups or []:
                    if s.get("side") != "LONG":
                        continue

                    diag["raw_setups"] += 1

                    atr_pct = float(s.get("atr_pct", 0) or 0)
                    score = compute_score(vol_24h, atr_pct)
                    if score < MIN_SCORE_TO_ALERT:
                        continue
                    diag["pass_score"] += 1

                    ok, notes, ctx = check_confirmations(client, sym)
                    if not ok:
                        continue

                    atr_from_ctx = float(ctx.get("atr_pct", atr_pct) or atr_pct)

                    # Clasificación base
                    cls = classify_trade(atr_from_ctx, score)

                    # ---- Filtros extra por tipo (Phase C) ----
                    if cls == "SCALP":
                        if score < SCALP_MIN_SCORE:
                            diag["discard_context"] += 1
                            continue
                        if not context_filters_ok(
                            client=client,
                            symbol=sym,
                            vol_24h=vol_24h,
                            trade_class="SCALP",
                            atr_pct=atr_from_ctx,
                        ):
                            diag["discard_context"] += 1
                            continue
                    elif cls == "SWING":
                        if score < SWING_MIN_SCORE:
                            diag["discard_context"] += 1
                            continue
                        if not context_filters_ok(
                            client=client,
                            symbol=sym,
                            vol_24h=vol_24h,
                            trade_class="SWING",
                            atr_pct=atr_from_ctx,
                        ):
                            diag["discard_context"] += 1
                            continue

                    # A partir de aquí ya pasó todos los filtros
                    entry = float(s["entry"])
                    sl = float(s["sl"])
                    tp = float(s["tp"])

                    exec_ok, _, _, _ = sl_tp_executable(entry, sl, tp)
                    if not exec_ok:
                        diag["fail_exec"] += 1
                        continue

                    key = f"{sym}:{s.get('setup', 'SETUP')}:LONG:{cls}"
                    if key in state:
                        continue

                    trade = {
                        "symbol": sym,
                        "setup": s.get("setup", "SETUP"),
                        "class": cls,
                        "score": score,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "opened_at": int(time.time()),
                        "max_price": entry,
                        "trailing": False,
                    }
                    state[key] = trade

                    write_trade_event(report_path, "OPEN", trade)
                    send_telegram(
                        fmt_entry_long(
                            sym=sym,
                            cls=cls,
                            score=score,
                            vol_24h=vol_24h,
                            entry=entry,
                            sl=sl,
                            tp=tp,
                            atr_pct=atr_from_ctx,
                            setup_name=str(s.get("setup", "SETUP")),
                            confirm_notes=notes,
                        )
                    )

                    alerts_sent += 1
                    diag["confirmed"] += 1
                    if alerts_sent >= MAX_ALERTS_PER_CYCLE:
                        break

                if alerts_sent >= MAX_ALERTS_PER_CYCLE:
                    break

            diag["open_trades"] = len(state)
            save_state(state)

            # =========================
            # Pre-alertas (cerca EMA20)
            # =========================
            prealerts: List[Tuple[float, str, float, float]] = []
            for sym in operables_by_vol:
                if STOP["flag"]:
                    raise KeyboardInterrupt()
                if (time.time() - start_cycle) > TIME_BUDGET_SECONDS:
                    break
                try:
                    ohlc = fetch_15m_ohlc(client, sym, limit=60)
                    closes = [c["close"] for c in ohlc]
                    e20 = ema_tolerant(closes, 20)
                    if e20 is None:
                        continue
                    price = closes[-1]
                    dist_pct = abs((price - e20) / e20) * 100.0
                    if dist_pct <= PREALERT_EMA20_DISTANCE_PCT:
                        prealerts.append((dist_pct, sym, price, e20))
                except Exception:
                    continue
                if len(prealerts) >= PREALERT_MAX:
                    break

            elapsed = int(time.time() - start_cycle)

            send_telegram(
                fmt_summary(
                    total_usdt=len(all_usdt),
                    operables=len(operables),
                    elapsed=elapsed,
                    top_symbols=top_symbols,
                    diag=diag,
                )
            )

            for dist, sym, price, e20 in sorted(prealerts, key=lambda x: x[0])[
                :PREALERT_MAX
            ]:
                send_telegram(fmt_prealert(sym, price, e20, dist))

            log(
                f"🟢 Ciclo finalizado. Confirmadas: {diag['confirmed']} | Crudos: {diag['raw_setups']} | "
                f"FailExec: {diag['fail_exec']} | DescContexto: {diag['discard_context']}"
            )
            log(f"😴 Durmiendo {CYCLE_MINUTES} minutos...\n")

            # Sleep interrumpible
            for _ in range(CYCLE_MINUTES * 60):
                if STOP["flag"]:
                    raise KeyboardInterrupt()
                time.sleep(1)

        except KeyboardInterrupt:
            log("🛑 Detenido por el usuario.")
            break
        except TimeoutError as e:
            log(f"⏳ Watchdog: {e}")
            time.sleep(30)
        except Exception as e:
            log(f"⚠️ Error del ciclo: {type(e).__name__}: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main_loop()

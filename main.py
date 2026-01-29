# main.py
# MARKET INTELLIGENCE SYSTEM (MIS) - ETAPA 4
# Consola interactiva para:
# - Ver ranking 1H (Swing / Acción)
# - Filtrar CORE vs HIGH RISK
# - Ver setups 15M por símbolo o por Top candidatos
#
# Requisitos:
# - market/scanner_1h.py debe tener run_scanner_1h()
# - utils/indicators.py debe existir (ema, atr, last_n_high_low, etc.)
#
# Para detener en cualquier momento: Ctrl + C en la terminal

from strategy_setups import setup_ema20_bounce
import sys
from pathlib import Path

# Fix de rutas (Windows / VS Code)
ROOT = Path(__file__).resolve().parent  # binance_bot/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional

from binance.spot import Spot
from utils.indicators import ema, atr, last_n_high_low

from market.scanner_1h import run_scanner_1h

# =========================
# CONFIGURACIÓN MIS
# =========================
CORE = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"
}

EXCLUDE_SYMBOLS = {"USDCUSDT", "FDUSDUSDT", "USD1USDT", "BFUSDUSDT", "TUSDUSDT", "USDPUSDT", "BUSDUSDT"}

# 15M setup params (ETAPA 3.1)
KLINES_LIMIT_15M = 140  # moderado para rapidez
EMA_FAST_15M = 20
EMA_SLOW_15M = 50
ATR_PERIOD = 14

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0

BREAKOUT_LOOKBACK = 40
BREAKOUT_MARGIN_PCT = 0.05
RETEST_WINDOW = 6
RETEST_TOL_PCT = 0.20

PULLBACK_ZONE_TOL_PCT = 0.45
REJECTION_BODY_MIN_PCT = 0.12
REJECTION_WICK_RATIO = 1.3

DT_LOOKBACK = 60
DT_TOL_PCT = 0.25

MIN_ATR_PCT_15M = 0.20
MAX_ATR_PCT_15M = 5.00
MIN_PRICE = 0.00001


# =========================
# UTILIDADES
# =========================
def safe_call(fn, *args, retries: int = 4, base_sleep: float = 1.5, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = base_sleep * (2 ** i)
            print(f"⚠️  Red inestable ({type(e).__name__}). Reintento {i+1}/{retries} en {wait:.0f}s...")
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = base_sleep * (2 ** i)
            print(f"⚠️  Error temporal ({type(e).__name__}). Reintento {i+1}/{retries} en {wait:.0f}s...")
            time.sleep(wait)
    raise last_err


def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"


def print_table(title: str, rows: List[dict], limit: int = 20):
    print(f"\n=== {title} ===")
    if not rows:
        print("Sin resultados.")
        return
    for r in rows[:limit]:
        sym = r["symbol"]
        st = r["state"]
        atrp = r["atr_pct"]
        volm = r["quoteVolume"] / 1_000_000
        sc = r["score"]
        tag = "CORE" if sym in CORE else "HIGH_RISK"
        print(f"{sym:<12} {tag:<10} {st:<26} ATR%:{atrp:<5.2f} Vol:{volm:>7.1f}M Score:{sc:.2f}")


# =========================
# 15M: DATOS + SETUPS
# =========================
def fetch_klines(client: Spot, symbol: str, interval: str, limit: int):
    kl = safe_call(client.klines, symbol=symbol, interval=interval, limit=limit)
    opens  = [float(k[1]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]
    return opens, highs, lows, closes


def atr_pct(highs, lows, closes):
    a = atr(highs, lows, closes, ATR_PERIOD)
    return (a / closes[-1]) * 100, a


def suggest_sl_tp(entry, atr_val, side: str):
    if side == "LONG":
        sl = entry - SL_ATR_MULT * atr_val
        tp = entry + TP_ATR_MULT * atr_val
    else:
        sl = entry + SL_ATR_MULT * atr_val
        tp = entry - TP_ATR_MULT * atr_val
    return sl, tp


def detect_pullback_rejection(opens, highs, lows, closes) -> Optional[dict]:
    ema20 = ema(closes, EMA_FAST_15M)
    ema50 = ema(closes, EMA_SLOW_15M)

    # Tendencia local alcista: EMA20 > EMA50
    if not (ema20 > ema50):
        return None

    price = closes[-1]
    zone_low = min(ema20, ema50) * (1 - PULLBACK_ZONE_TOL_PCT / 100)
    zone_high = max(ema20, ema50) * (1 + PULLBACK_ZONE_TOL_PCT / 100)

    if not (zone_low <= price <= zone_high):
        return None

    o = opens[-1]
    l = lows[-1]
    c = closes[-1]

    body = abs(c - o)
    wick_down = min(o, c) - l
    body_pct = (body / max(c, 1e-9)) * 100

    # Rechazo alcista: cierre > apertura + wick inferior dominante
    if c > o and wick_down > body * REJECTION_WICK_RATIO and body_pct >= REJECTION_BODY_MIN_PCT:
        return {"type": "PULLBACK_REJECTION", "side": "LONG", "entry": c}

    return None


def detect_breakout_retest(highs, lows, closes) -> Optional[dict]:
    max_h, _ = last_n_high_low(highs, lows, BREAKOUT_LOOKBACK)
    margin = max_h * (BREAKOUT_MARGIN_PCT / 100)

    breakout_idx = None
    start_scan = max(0, len(closes) - RETEST_WINDOW)
    for i in range(start_scan, len(closes)):
        if closes[i] > (max_h + margin):
            breakout_idx = i
            break

    if breakout_idx is None:
        return None

    level = max_h
    tol = level * (RETEST_TOL_PCT / 100)

    for j in range(breakout_idx, len(closes)):
        if abs(lows[j] - level) <= tol:
            if closes[-1] > level:
                return {"type": "BREAKOUT_RETEST", "side": "LONG", "entry": closes[-1]}

    return None


def detect_double_top_bottom_basic(highs, lows, closes) -> Optional[dict]:
    max_h, min_l = last_n_high_low(highs, lows, DT_LOOKBACK)
    mid = (max_h + min_l) / 2
    tol = mid * (DT_TOL_PCT / 100)
    price = closes[-1]

    if abs(highs[-1] - max_h) <= tol and price < mid:
        return {"type": "DOUBLE_TOP_BASIC", "side": "SHORT", "entry": price}
    if abs(lows[-1] - min_l) <= tol and price > mid:
        return {"type": "DOUBLE_BOTTOM_BASIC", "side": "LONG", "entry": price}
    return None


def scan_15m_for_symbol(client, symbol: str):
    """
    Scanner 15m: ahora usa setup base EMA20_BOUNCE.
    Devuelve lista de setups.
    """
    try:
        kl = client.klines(symbol, "15m", limit=60)
    except Exception:
        return []

    ohlc = []
    try:
        for k in kl:
            ohlc.append({
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
            })
    except Exception:
        return []

    # Setup base
    setups = []
    setups += setup_ema20_bounce(ohlc)

    return setups

def scan_15m_for_universe(universe: List[str]) -> List[dict]:
    client = Spot()
    all_setups = []
    for sym in universe:
        try:
            all_setups.extend(scan_15m_for_symbol(client, sym))
        except Exception:
            continue
    # ordenar por ATR% (más calmado primero)
    all_setups.sort(key=lambda x: x["atr_pct"])
    return all_setups


# =========================
# MIS: MENÚ
# =========================
def build_universe(rank_1h: dict, mode: str) -> Dict[str, List[dict]]:
    swing = rank_1h["swing"]
    action = rank_1h["action"]

    if mode == "CORE":
        swing = [r for r in swing if r["symbol"] in CORE]
        action = [r for r in action if r["symbol"] in CORE]
    elif mode == "HIGH":
        swing = [r for r in swing if r["symbol"] not in CORE]
        action = [r for r in action if r["symbol"] not in CORE]
    return {"swing": swing, "action": action}


def menu():
    last_rank = None
    last_mode = "MIXTO"
    last_time = None

    while True:
        print("\n==============================")
        print(" MARKET INTELLIGENCE SYSTEM")
        print("==============================")
        print("1) Actualizar ranking 1H (Top Swing + Acción)")
        print("2) Ver ranking actual (MIXTO / CORE / HIGH RISK)")
        print("3) Cambiar modo de filtro (MIXTO / CORE / HIGH)")
        print("4) Ver setups 15M de UNA moneda (ej. BTCUSDT)")
        print("5) Ver setups 15M del Top (Swing+Acción) según modo")
        print("6) Salir")
        choice = input("Seleccione una opción: ").strip()

        if choice == "1":
            print("\n⏳ Ejecutando Scanner 1H...")
            last_rank = run_scanner_1h(20, 10)
            last_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ Ranking 1H actualizado: {last_time}")

        elif choice == "2":
            if not last_rank:
                print("Primero ejecuta (1) para generar el ranking 1H.")
                continue

            filtered = build_universe(last_rank, last_mode)
            print(f"\nModo actual: {last_mode} | Ranking actualizado: {last_time}")
            print_table("TOP SWING (1H)", filtered["swing"], 20)
            print_table("TOP ACCIÓN (1H)", filtered["action"], 10)

        elif choice == "3":
            print("\nElige modo:")
            print(" - MIXTO  (todo)")
            print(" - CORE   (BTC/ETH/BNB/SOL/XRP/ADA)")
            print(" - HIGH   (alto riesgo / fuera de CORE)")
            m = input("Modo: ").strip().upper()
            if m in {"MIXTO", "CORE", "HIGH"}:
                last_mode = m
                print(f"✅ Modo cambiado a: {last_mode}")
            else:
                print("Modo inválido. Usa: MIXTO, CORE o HIGH.")

        elif choice == "4":
            sym = input("Símbolo (ej. BTCUSDT): ").strip().upper()
            if not sym.endswith("USDT"):
                print("Símbolo inválido. Ejemplo válido: BTCUSDT")
                continue

            print(f"\n⏳ Analizando 15M para {sym}...")
            try:
                client = Spot()
                setups = scan_15m_for_symbol(client, sym)
            except Exception as e:
                print(f"Error analizando {sym}: {type(e).__name__}: {e}")
                continue

            if not setups:
                print("No se detectaron setups claros (15M) para este símbolo.")
                continue

            print("\n=== SETUPS DETECTADOS (15M) ===")
            for r in setups:
                print(
                    f"{r['symbol']:<12} {r['setup']:<18} {r['side']:<5} "
                    f"Entry:{fmt_price(r['entry']):>10} "
                    f"SL:{fmt_price(r['sl']):>10} "
                    f"TP:{fmt_price(r['tp']):>10} "
                    f"ATR%:{r['atr_pct']:<5.2f}"
                )

        elif choice == "5":
            if not last_rank:
                print("Primero ejecuta (1) para generar el ranking 1H.")
                continue

            filtered = build_universe(last_rank, last_mode)
            universe_syms = [r["symbol"] for r in filtered["swing"]] + [r["symbol"] for r in filtered["action"]]
            universe_syms = [s for s in universe_syms if s not in EXCLUDE_SYMBOLS]

            if not universe_syms:
                print("No hay símbolos en el universo actual (con el modo seleccionado).")
                continue

            print(f"\n⏳ Analizando 15M para universo Top (modo {last_mode})... (esto puede tardar un poco)")
            setups = scan_15m_for_universe(universe_syms)

            if not setups:
                print("No se detectaron setups claros en este momento para el universo seleccionado.")
                continue

            print("\n=== SETUPS DETECTADOS (15M) ===")
            for r in setups[:25]:
                print(
                    f"{r['symbol']:<12} {r['setup']:<18} {r['side']:<5} "
                    f"Entry:{fmt_price(r['entry']):>10} "
                    f"SL:{fmt_price(r['sl']):>10} "
                    f"TP:{fmt_price(r['tp']):>10} "
                    f"ATR%:{r['atr_pct']:<5.2f}"
                )

        elif choice == "6":
            print("Saliendo. Listo.")
            break

        else:
            print("Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    menu()

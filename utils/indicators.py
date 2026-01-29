# utils/indicators.py
from __future__ import annotations

from typing import List, Tuple
import math


def ema(values: List[float], period: int) -> float:
    """
    EMA simple (devuelve el último valor).
    Requiere al menos 'period' valores.
    """
    if len(values) < period:
        raise ValueError(f"EMA necesita al menos {period} valores.")

    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period  # arranque con SMA

    for v in values[period:]:
        ema_val = (v - ema_val) * k + ema_val

    return ema_val


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """
    ATR basado en True Range promedio (simple).
    Requiere al menos period+1 velas.
    """
    if len(closes) < period + 1:
        raise ValueError(f"ATR necesita al menos {period + 1} velas.")

    trs = []
    for i in range(1, len(closes)):
        tr = true_range(highs[i], lows[i], closes[i - 1])
        trs.append(tr)

    return sum(trs[-period:]) / period


def last_n_high_low(highs: List[float], lows: List[float], n: int = 50) -> Tuple[float, float]:
    """
    Devuelve (max_high, min_low) de las últimas n velas.
    """
    if len(highs) < n or len(lows) < n:
        raise ValueError(f"Necesito al menos {n} velas para rango.")

    max_high = max(highs[-n:])
    min_low = min(lows[-n:])
    return max_high, min_low


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def log10_safe(x: float) -> float:
    return math.log10(max(x, 1.0))

#!/usr/bin/env python
# analyze_paper_results.py
#
# Analiza resultados de los CSV generados por alerts_runner (paper_trades_YYYY-MM-DD.csv)
# Modos:
#   - Sin argumentos: analiza el último archivo disponible.
#   - --date=YYYY-MM-DD: analiza solo ese día.
#   - --from=YYYY-MM-DD --to=YYYY-MM-DD: analiza un rango de días (multi-día).
#
# Métricas:
#   - Trades cerrados, win rate, ganancia media, pérdida media, expectancy.
#   - Separado por SCALP y SWING.
#
# IMPORTANTE: solo considera filas con event == "CLOSE".

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple, Dict

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"


@dataclass
class TradeClose:
    ts: datetime
    symbol: str
    setup: str
    trade_class: str  # SCALP | SWING | otro
    entry: float
    exit_price: float
    outcome: str
    pnl_pct: float
    score: float
    day: date


@dataclass
class Stats:
    n_trades: int
    n_wins: int
    n_losses: int
    avg_win: float
    avg_loss: float
    win_rate: float
    expectancy: float


def parse_date_str(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def find_report_files_for_range(d_from: Optional[date], d_to: Optional[date]) -> List[Path]:
    """
    Busca todos los archivos paper_trades_YYYY-MM-DD.csv dentro de REPORTS_DIR.
    Si d_from y d_to son None, devuelve solo el último por fecha.
    Si solo hay d_from, y d_to es None, analiza solo ese día.
    Si hay ambos, filtra por rango inclusivo.
    """
    files = []
    for p in REPORTS_DIR.glob("paper_trades_*.csv"):
        try:
            # espera formato paper_trades_YYYY-MM-DD.csv
            name = p.stem  # paper_trades_YYYY-MM-DD
            parts = name.split("_")
            d_str = parts[-1]
            d = parse_date_str(d_str)
        except Exception:
            continue
        files.append((d, p))

    if not files:
        return []

    files.sort(key=lambda x: x[0])  # por fecha

    if d_from is None and d_to is None:
        # último archivo
        return [files[-1][1]]

    if d_from is not None and d_to is None:
        # solo un día específico
        return [p for (d, p) in files if d == d_from]

    # rango completo
    assert d_from is not None and d_to is not None
    return [p for (d, p) in files if d_from <= d <= d_to]


def load_trades_from_file(path: Path) -> List[TradeClose]:
    out: List[TradeClose] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get("event") != "CLOSE":
                    continue
                ts_str = row.get("timestamp", "")
                # form: "2026-01-15 14:01:44"
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                d = ts.date()
                symbol = row.get("symbol", "")
                setup = row.get("setup", "")
                trade_class = row.get("class", "")
                entry = float(row.get("entry") or 0.0)
                exit_price = float(row.get("exit_price") or 0.0)
                outcome = row.get("outcome", "")
                pnl_pct = float(row.get("pnl_pct") or 0.0)
                score = float(row.get("score") or 0.0)
                out.append(
                    TradeClose(
                        ts=ts,
                        symbol=symbol,
                        setup=setup,
                        trade_class=trade_class,
                        entry=entry,
                        exit_price=exit_price,
                        outcome=outcome,
                        pnl_pct=pnl_pct,
                        score=score,
                        day=d,
                    )
                )
            except Exception:
                # si hay una fila corrupta, la saltamos
                continue
    return out


def compute_stats(trades: List[TradeClose]) -> Stats:
    if not trades:
        return Stats(
            n_trades=0,
            n_wins=0,
            n_losses=0,
            avg_win=0.0,
            avg_loss=0.0,
            win_rate=0.0,
            expectancy=0.0,
        )

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct < 0]

    n_trades = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)

    avg_win = sum(t.pnl_pct for t in wins) / n_wins if n_wins > 0 else 0.0
    avg_loss = sum(t.pnl_pct for t in losses) / n_losses if n_losses > 0 else 0.0

    win_rate = (n_wins / n_trades) * 100.0

    # Expectancy = p(win)*avg_win + p(loss)*avg_loss
    p_win = n_wins / n_trades
    p_loss = n_losses / n_trades
    expectancy = p_win * avg_win + p_loss * avg_loss

    return Stats(
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_rate=win_rate,
        expectancy=expectancy,
    )


def print_stats(title: str, stats: Stats):
    print()
    print(title)
    print("-" * len(title))
    if stats.n_trades == 0:
        print("Sin trades cerrados en este conjunto.")
        return

    print(f"Trades cerrados: {stats.n_trades}")
    print(f"Win rate: {stats.win_rate:.2f}%")
    print(f"Ganancia media: {stats.avg_win:+.2f}%")
    print(f"Pérdida media: {stats.avg_loss:+.2f}%")
    print(f"Expectancy: {stats.expectancy:+.2f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analizador de resultados de paper trading.")
    parser.add_argument("--date", type=str, default=None,
                        help="Fecha específica a analizar, formato YYYY-MM-DD.")
    parser.add_argument("--from", dest="from_date", type=str, default=None,
                        help="Fecha inicio (inclusive) para análisis multi-día, formato YYYY-MM-DD.")
    parser.add_argument("--to", dest="to_date", type=str, default=None,
                        help="Fecha fin (inclusive) para análisis multi-día, formato YYYY-MM-DD.")

    args = parser.parse_args(argv)

    # Determinar qué archivos analizar
    d_from: Optional[date] = None
    d_to: Optional[date] = None

    if args.date:
        # Si se pasa --date, ignoramos from/to y analizamos solo ese día
        try:
            d_from = parse_date_str(args.date)
            d_to = d_from
        except Exception:
            print(f"Fecha inválida en --date: {args.date}")
            sys.exit(1)
        files = find_report_files_for_range(d_from, d_to)
        if not files:
            print(f"No se encontró archivo para la fecha específica: {args.date}")
            sys.exit(0)
        mode_desc = f"Analizando fecha específica: {files[0].name}"
    elif args.from_date or args.to_date:
        # Rango multi-día
        if not args.from_date or not args.to_date:
            print("Debe proporcionar ambos argumentos: --from y --to (formato YYYY-MM-DD).")
            sys.exit(1)
        try:
            d_from = parse_date_str(args.from_date)
            d_to = parse_date_str(args.to_date)
        except Exception:
            print("Formato de fecha inválido en --from o --to. Use YYYY-MM-DD.")
            sys.exit(1)
        if d_from > d_to:
            print("La fecha --from no puede ser mayor que --to.")
            sys.exit(1)
        files = find_report_files_for_range(d_from, d_to)
        if not files:
            print(f"No se encontraron archivos entre {args.from_date} y {args.to_date}.")
            sys.exit(0)
        mode_desc = f"Analizando rango: {args.from_date} a {args.to_date} ({len(files)} archivo(s))"
    else:
        # Último archivo disponible
        files = find_report_files_for_range(None, None)
        if not files:
            print("No se encontraron archivos paper_trades_YYYY-MM-DD.csv en 'reports/'.")
            sys.exit(0)
        mode_desc = f"Analizando último archivo: {files[-1].name}"

    # Cargar trades de todos los archivos seleccionados
    all_trades: List[TradeClose] = []
    for path in files:
        all_trades.extend(load_trades_from_file(path))

    print(f"{mode_desc}\n")

    if not all_trades:
        print("⚠️ Aún no hay trades cerrados en los archivos seleccionados.")
        return

    # Estadísticas generales
    stats_all = compute_stats(all_trades)

    # Separar por clase
    scalps = [t for t in all_trades if t.trade_class.upper() == "SCALP"]
    swings = [t for t in all_trades if t.trade_class.upper() == "SWING"]

    stats_scalp = compute_stats(scalps)
    stats_swing = compute_stats(swings)

    print_stats("RESULTADOS GENERALES", stats_all)
    print_stats("SCALP", stats_scalp)
    print_stats("SWING", stats_swing)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Headless CVaR-Portfolio-Optimierung mit cuFOLIO (cuOpt + cuML).

Reproduziert notebooks/cvar_basic.ipynb ohne Jupyter. Fokus: einfach lauffaehig
auf einem (dual) RTX A6000 Setup. Kein CPU-Vergleich, keine Extras.

Ablauf: Daten -> Log-Returns -> KDE-Szenarien (GPU) -> Mean-CVaR
        -> cuOpt-LP-Solve (GPU) -> Backtest -> speichern.

Im cuFOLIO-Repo-Root ablegen und so ausfuehren:
    uv run python run_cvar.py --check          # erst pruefen: GPUs + cuOpt/cuML sichtbar?
    uv run python run_cvar.py                   # voller Lauf (Defaults)
    uv run python run_cvar.py --gpu-id 1        # auf GPU 1 pinnen
    uv run python run_cvar.py --config my.yaml  # alle Parameter aus einer YAML laden

Die --config-Option laedt eine YAML mit allen Parametern (siehe config.example.yaml).
Fehlende Schluessel werden mit den Defaults aufgefuellt. Das ist die Basis fuer die
Gradio-Oberflaeche (app.py), die pro Lauf eine YAML schreibt und dieses Skript aufruft.

Ursprung:
    Basiert auf dem NVIDIA-AI-Blueprint cuFOLIO, konkret auf dem Beispiel-Notebook
    notebooks/cvar_basic.ipynb. Dieses Notebook wurde hier in ein headless
    ausfuehrbares Skript ueberfuehrt.
    Quelle: https://github.com/NVIDIA-AI-Blueprints/cuFOLIO (Apache-2.0)

Entstehung:
    Erstellt mit Claude Opus 4.8 als Coding-Assistent.

Details und vollstaendige Schritt-fuer-Schritt-Anleitung im Blogpost:
    https://ai-box.eu/news/cufolio-ohne-jupyter-gpu-portfolio-optimierung-per-skript-auf-dual-rtx-a6000/2657/
"""

import argparse
import glob
import json
import os
import subprocess


# Alle einstellbaren Parameter an einem Ort. Die Defaults entsprechen exakt
# dem urspruenglichen run_cvar.py, damit ein Lauf ohne Config identisch bleibt.
DEFAULTS = {
    # Daten & Laufzeit
    "dataset": "sp500",                 # sp500, sp100, dow30, global_titans
    "num_scen": 10000,                  # Anzahl KDE-Szenarien
    "seed": None,                       # Zufalls-Seed (None = zufaellig, Zahl = reproduzierbar)
    "regime_start": "2021-01-01",       # Optimierungszeitraum Start
    "regime_end": "2024-01-01",         # Optimierungszeitraum Ende
    "test_start": "2023-09-01",         # Backtest-Zeitraum Start
    "test_end": "2024-07-01",           # Backtest-Zeitraum Ende
    "results_dir": "results/run_cvar",  # Ausgabeverzeichnis
    "no_backtest": False,               # Backtest ueberspringen
    "gpu_id": None,                     # None = beide sichtbar, sonst 0 oder 1
    # Mean-CVaR-Parameter
    "w_min_others": -0.3,               # untere Gewichts-Grenze (alle ausser Overrides)
    "w_max_others": 0.4,                # obere Gewichts-Grenze (alle ausser Overrides)
    "ticker_overrides": {"NVDA": [0.1, 0.6]},  # pro Titel [min, max]
    "c_min": 0.0,                       # Cash-Untergrenze
    "c_max": 0.2,                       # Cash-Obergrenze
    "L_tar": 1.6,                       # Leverage-Ziel
    "T_tar": None,                      # Turnover-Ziel (None = aus)
    "cvar_limit": None,                 # harte CVaR-Grenze (None = unbeschraenkt)
    "cardinality": None,                # max. Anzahl Titel (None = aus)
    "risk_aversion": 1,                 # Risikoaversion
    "confidence": 0.95,                 # CVaR-Konfidenz (alpha)
    # Solver
    "solver_method": "PDLP",            # cuOpt-Loesungsverfahren
    "time_limit": 15,                   # Zeitlimit in Sekunden
    "optimality": 1e-4,                 # Optimalitaets-Toleranz
}


def gpu_check():
    """Schneller Smoke-Test: sind die GPUs und der GPU-Stack sichtbar?"""
    print("== GPU-Check ==")
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("  nvidia-smi nicht verfuegbar:", exc)

    for mod in ("cuml", "cuopt", "cvxpy", "cufolio"):
        try:
            m = __import__(mod)
            print(f"  {mod:8s} OK  {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:8s} Import-Problem -> {exc}")
    print("Wenn alle Zeilen OK zeigen, ist das Setup bereit.")


def _set_seed(seed):
    """Setzt die gaengigen Zufallsquellen. Macht die KDE-Szenarien reproduzierbar.

    Hinweis: Das deckt numpy, cupy (GPU) und Pythons random ab. Falls cuFOLIO intern
    eine eigene, nicht erfasste Quelle nutzt, ist absolute Determinismus nicht garantiert.
    In der Praxis liefern zwei Laeufe mit gleichem Seed aber dasselbe Portfolio.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cupy
        cupy.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    print(f"[info] Zufalls-Seed gesetzt: {seed}")


def _build_cvar_params(cfg, CvarParameters):
    """Baut die CvarParameters aus dem Config-Dict (inkl. Ticker-Overrides)."""
    w_min = {"others": cfg["w_min_others"]}
    w_max = {"others": cfg["w_max_others"]}
    for ticker, bounds in (cfg.get("ticker_overrides") or {}).items():
        lo, hi = bounds
        w_min[ticker] = lo
        w_max[ticker] = hi
    return CvarParameters(
        w_min=w_min,
        w_max=w_max,
        c_min=cfg["c_min"], c_max=cfg["c_max"],
        L_tar=cfg["L_tar"], T_tar=cfg["T_tar"],
        cvar_limit=cfg["cvar_limit"],
        cardinality=cfg["cardinality"],
        risk_aversion=cfg["risk_aversion"],
        confidence=cfg["confidence"],
    )


def _write_run_summary(results_dir, cfg, gpu_results, portfolio_path, backtest_result):
    """Schreibt run_summary.json: kompakte, maschinenlesbare Ergebnisse fuer die UI."""
    summary = {"seed": cfg.get("seed"), "solve": {}, "positions": [],
               "cash": None, "backtest": {}, "plot": None}

    # Solve-Kennzahlen (gpu_results ist i. d. R. eine pandas-Series)
    try:
        d = gpu_results.to_dict()
        for k, v in d.items():
            try:
                summary["solve"][str(k)] = float(v)
            except (TypeError, ValueError):
                summary["solve"][str(k)] = str(v)
    except Exception:  # noqa: BLE001
        summary["solve"] = {"raw": str(gpu_results)}

    # Positionen aus der gespeicherten portfolio.json (robust gegen API-Details)
    try:
        with open(portfolio_path) as fh:
            pf = json.load(fh)
        weights = pf.get("weights", [])
        tickers = pf.get("tickers", [])
        summary["cash"] = pf.get("cash")
        for tk, w in zip(tickers, weights):
            if abs(w) > 1e-4:
                summary["positions"].append({
                    "ticker": tk,
                    "direction": "Long" if w > 0 else "Short",
                    "weight": w,
                })
        summary["positions"].sort(key=lambda x: -x["weight"])
    except Exception as exc:  # noqa: BLE001
        summary["positions_error"] = str(exc)

    # Backtest-Kennzahlen (Sortino, Max Drawdown) aus dem DataFrame ziehen
    if backtest_result is not None:
        try:
            for name in backtest_result.index:
                row = backtest_result.loc[name]
                entry = {}
                for col in row.index:
                    label = str(col).lower()
                    if "sortino" in label:
                        entry["sortino"] = float(row[col])
                    elif "drawdown" in label:
                        entry["max_drawdown"] = float(row[col])
                summary["backtest"][str(name)] = entry
        except Exception as exc:  # noqa: BLE001
            summary["backtest_error"] = str(exc)

    # Pfad zur Backtest-Grafik suchen
    plots = glob.glob(os.path.join(results_dir, "combined_*.png"))
    if plots:
        summary["plot"] = os.path.basename(plots[0])

    with open(os.path.join(results_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def run_pipeline(cfg):
    """Fuehrt die komplette Pipeline aus. cfg ist ein Dict (fehlende Keys -> DEFAULTS)."""
    cfg = {**DEFAULTS, **(cfg or {})}

    # GPU-Pinning muss vor dem Import der CUDA-Bibliotheken passieren
    if cfg["gpu_id"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])
        print(f"[info] CUDA_VISIBLE_DEVICES={cfg['gpu_id']}")

    import cvxpy as cp
    from cufolio import cvar_optimizer, cvar_utils, utils
    from cufolio.cvar_parameters import CvarParameters
    from cufolio.settings import (
        KDESettings,
        ReturnsComputeSettings,
        ScenarioGenerationSettings,
    )

    # Seed setzen, bevor irgendetwas Zufaelliges passiert (KDE-Sampling)
    if cfg.get("seed") is not None:
        _set_seed(cfg["seed"])

    # --- 1. Daten ---
    data_path = f"data/stock_data/{cfg['dataset']}.csv"
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"[1/6] Lade Daten nach {data_path} ...")
        utils.download_data(data_path)
    else:
        print(f"[1/6] Verwende vorhandene Daten: {data_path}")

    # --- 2. Log-Returns fuer den Optimierungszeitraum ---
    print("[2/6] Berechne Returns ...")
    regime_dict = {"name": "recent", "range": (cfg["regime_start"], cfg["regime_end"])}
    returns_compute_settings = ReturnsComputeSettings(return_type="LOG", freq=1)
    returns_dict = utils.calculate_returns(data_path, regime_dict, returns_compute_settings)

    # --- 3. KDE-Szenarien auf der GPU (cuML) ---
    print(f"[3/6] Erzeuge {cfg['num_scen']} KDE-Szenarien (GPU) ...")
    scenario_generation_settings = ScenarioGenerationSettings(
        num_scen=cfg["num_scen"],
        fit_type="kde",
        kde_settings=KDESettings(bandwidth=0.01, kernel="gaussian", device="GPU"),
        verbose=False,
    )
    returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)

    # --- 4. Mean-CVaR-Parameter ---
    cvar_params = _build_cvar_params(cfg, CvarParameters)

    # --- 5. Problem aufbauen und auf der GPU loesen (cuOpt) ---
    print(f"[4/6] Loese Mean-CVaR auf der GPU (cuOpt {cfg['solver_method']}) ...")
    problem = cvar_optimizer.CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
    gpu_solver_settings = {
        "solver": cp.CUOPT,
        "verbose": False,
        "solver_method": cfg["solver_method"],
        "time_limit": cfg["time_limit"],
        "optimality": cfg["optimality"],
    }
    gpu_results, gpu_portfolio = problem.solve_optimization_problem(solver_settings=gpu_solver_settings)
    print(gpu_results)

    # --- 6. Ergebnis speichern + Backtest ---
    os.makedirs(cfg["results_dir"], exist_ok=True)
    portfolio_path = os.path.join(cfg["results_dir"], "portfolio.json")
    gpu_portfolio.save_portfolio(portfolio_path)
    print("[5/6] Optimiertes Portfolio:")
    gpu_portfolio.print_clean(min_percentage=1)

    backtest_result = None
    if not cfg["no_backtest"]:
        print("[6/6] Backtest gegen Equal-Weight ...")
        from cufolio import backtest

        test_regime_dict = {"name": "test_recent", "range": (cfg["test_start"], cfg["test_end"])}
        test_returns_dict = utils.calculate_returns(data_path, test_regime_dict, returns_compute_settings)
        backtester = backtest.portfolio_backtester(
            gpu_portfolio, test_returns_dict, 0.0, "historical", benchmark_portfolios=None
        )
        backtest_result, _ = backtester.backtest_against_benchmarks(
            plot_returns=False, cut_off_date=regime_dict["range"][1]
        )
        print(backtest_result)
        utils.portfolio_plot_with_backtest(
            portfolio=gpu_portfolio,
            backtester=backtester,
            cut_off_date=regime_dict["range"][1],
            backtest_plot_title="Backtest Results",
            save_plot=True,
            results_dir=cfg["results_dir"],
        )

    summary = _write_run_summary(cfg["results_dir"], cfg, gpu_results, portfolio_path, backtest_result)
    print(f"\nFertig. Ergebnisse in: {cfg['results_dir']}")
    return summary


def _cfg_from_args(args):
    """Baut ein Config-Dict aus den CLI-Argumenten (CVaR-Parameter bleiben Default)."""
    return {
        "dataset": args.dataset,
        "num_scen": args.num_scen,
        "seed": args.seed,
        "regime_start": args.regime_range[0],
        "regime_end": args.regime_range[1],
        "test_start": args.test_range[0],
        "test_end": args.test_range[1],
        "results_dir": args.results_dir,
        "no_backtest": args.no_backtest,
        "gpu_id": args.gpu_id,
    }


def parse_args():
    p = argparse.ArgumentParser(description="cuFOLIO CVaR ohne Notebook")
    p.add_argument("--check", action="store_true", help="Nur Smoke-Test: GPUs + Pakete pruefen, dann Ende")
    p.add_argument("--config", default=None, help="Pfad zu einer YAML mit allen Parametern")
    p.add_argument("--dataset", default="sp500", help="sp500, sp100, dow30, global_titans")
    p.add_argument("--num-scen", type=int, default=10000, help="Anzahl KDE-Szenarien")
    p.add_argument("--seed", type=int, default=None, help="Zufalls-Seed (fuer reproduzierbare Laeufe)")
    p.add_argument("--regime-range", nargs=2, default=["2021-01-01", "2024-01-01"],
                   metavar=("START", "END"), help="Optimierungszeitraum")
    p.add_argument("--test-range", nargs=2, default=["2023-09-01", "2024-07-01"],
                   metavar=("START", "END"), help="Backtest-Zeitraum")
    p.add_argument("--results-dir", default="results/run_cvar", help="Ausgabeverzeichnis")
    p.add_argument("--no-backtest", action="store_true", help="Backtest ueberspringen")
    p.add_argument("--gpu-id", type=int, default=None, help="Prozess auf eine bestimmte GPU pinnen")
    return p.parse_args()


def main():
    args = parse_args()

    if args.check:
        gpu_check()
        return

    if args.config:
        import yaml  # lazy: nur noetig, wenn eine Config geladen wird
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = _cfg_from_args(args)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()

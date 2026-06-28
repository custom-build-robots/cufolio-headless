#!/usr/bin/env python3
"""Headless CVaR-Portfolio-Optimierung mit cuFOLIO (cuOpt + cuML).

Reproduziert notebooks/cvar_basic.ipynb ohne Jupyter. Fokus: einfach lauffaehig
auf einem (dual) RTX A6000 Setup. Kein CPU-Vergleich, keine Extras.

Ablauf: Daten -> Log-Returns -> KDE-Szenarien (GPU) -> Mean-CVaR
        -> cuOpt-LP-Solve (GPU) -> Backtest -> speichern.

Im cuFOLIO-Repo-Root ablegen und so ausfuehren:
    uv run python run_cvar.py --check     # erst pruefen: GPUs + cuOpt/cuML sichtbar?
    uv run python run_cvar.py             # voller Lauf
    uv run python run_cvar.py --gpu-id 1  # auf GPU 1 pinnen

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
import os
import subprocess

import matplotlib

# Headless: kein Display, Figuren werden nur gespeichert
matplotlib.use("Agg")


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


def parse_args():
    p = argparse.ArgumentParser(description="cuFOLIO CVaR ohne Notebook")
    p.add_argument("--check", action="store_true", help="Nur Smoke-Test: GPUs + Pakete pruefen, dann Ende")
    p.add_argument("--dataset", default="sp500", help="sp500, sp100, dow30, global_titans")
    p.add_argument("--num-scen", type=int, default=10000, help="Anzahl KDE-Szenarien")
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

    # GPU-Pinning (fuer Durchsatz: je ein Prozess pro Karte)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[info] CUDA_VISIBLE_DEVICES={args.gpu_id}")

    # Importe erst hier, damit --check auch bei kaputtem GPU-Stack noch diagnostiziert
    import cvxpy as cp
    from cufolio import cvar_optimizer, cvar_utils, utils
    from cufolio.cvar_parameters import CvarParameters
    from cufolio.settings import (
        KDESettings,
        ReturnsComputeSettings,
        ScenarioGenerationSettings,
    )

    # --- 1. Daten (laedt SP500 per yfinance, falls nicht vorhanden) ---
    data_path = f"data/stock_data/{args.dataset}.csv"
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"[1/6] Lade Daten nach {data_path} ...")
        utils.download_data(data_path)
    else:
        print(f"[1/6] Verwende vorhandene Daten: {data_path}")

    # --- 2. Log-Returns fuer den Optimierungszeitraum ---
    print("[2/6] Berechne Returns ...")
    regime_dict = {"name": "recent", "range": tuple(args.regime_range)}
    returns_compute_settings = ReturnsComputeSettings(return_type="LOG", freq=1)
    returns_dict = utils.calculate_returns(data_path, regime_dict, returns_compute_settings)

    # --- 3. KDE-Szenarien auf der GPU (cuML) ---
    print(f"[3/6] Erzeuge {args.num_scen} KDE-Szenarien (GPU) ...")
    scenario_generation_settings = ScenarioGenerationSettings(
        num_scen=args.num_scen,
        fit_type="kde",
        kde_settings=KDESettings(bandwidth=0.01, kernel="gaussian", device="GPU"),
        verbose=False,
    )
    returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)

    # --- 4. Mean-CVaR-Parameter (wie im Beispiel-Notebook) ---
    cvar_params = CvarParameters(
        w_min={"NVDA": 0.1, "others": -0.3},
        w_max={"NVDA": 0.6, "others": 0.4},
        c_min=0.0, c_max=0.2,        # Cash-Grenzen
        L_tar=1.6, T_tar=None,       # Leverage / Turnover
        cvar_limit=None,             # harte CVaR-Grenze (None = unbeschraenkt)
        cardinality=None,            # max. Anzahl Titel (None = aus)
        risk_aversion=1,
        confidence=0.95,             # CVaR-Konfidenz (alpha)
    )

    # --- 5. Problem aufbauen und auf der GPU loesen (cuOpt PDLP) ---
    print("[4/6] Loese Mean-CVaR auf der GPU (cuOpt PDLP) ...")
    problem = cvar_optimizer.CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
    gpu_solver_settings = {
        "solver": cp.CUOPT,
        "verbose": False,
        "solver_method": "PDLP",
        "time_limit": 15,
        "optimality": 1e-4,
    }
    gpu_results, gpu_portfolio = problem.solve_optimization_problem(solver_settings=gpu_solver_settings)
    print(gpu_results)

    # --- 6. Ergebnis speichern + Backtest ---
    os.makedirs(args.results_dir, exist_ok=True)
    gpu_portfolio.save_portfolio(os.path.join(args.results_dir, "portfolio.json"))
    print("[5/6] Optimiertes Portfolio:")
    gpu_portfolio.print_clean(min_percentage=1)

    if not args.no_backtest:
        print("[6/6] Backtest gegen Equal-Weight ...")
        from cufolio import backtest

        test_regime_dict = {"name": "test_recent", "range": tuple(args.test_range)}
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
            results_dir=args.results_dir,
        )

    print(f"\nFertig. Ergebnisse in: {args.results_dir}")


if __name__ == "__main__":
    main()

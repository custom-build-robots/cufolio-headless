#!/usr/bin/env python3
"""Gradio-Oberflaeche fuer run_cvar.py (cuFOLIO Mean-CVaR), mehrsprachig.

Reiter "Einzellauf": Parameter setzen, Config als YAML speichern/laden, fester Seed,
Vergleich zweier Laeufe (A vs B).
Reiter "Sweep": einen Parameter ueber mehrere Werte durchspielen, optional parallel
ueber zwei GPUs. Jeder Job laeuft als eigener Prozess (run_cvar.py --config job.yaml).

Alle sichtbaren Texte und Tooltips liegen in language.yml (de / en).

Im cuFOLIO-Repo-Root ablegen (neben run_cvar.py und language.yml) und starten:
    uv run --with gradio --with pyyaml python app.py

Im Browser dann: http://<server-ip>:7860
Laeuft die App im Docker-Container, starte den Container mit -p 7860:7860.

Erstellt mit Claude Opus 4.8 als Coding-Assistent.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import gradio as gr
import pandas as pd
import yaml

CONFIG_DIR = "configs"
SNAPSHOT_DIR = "snapshots"
SWEEP_DIR = os.path.join("results", "sweep")
LANG_FILE = "language.yml"
DEFAULT_LANG = "de"
GPU_CHOICES = ["Auto", "GPU 0", "GPU 1"]
GPU_MODE_CHOICES = ["1 GPU", "2 GPUs"]
# Sinnvoll sweepbare Parameter (Config-Schluessel). Bedeutung steht in den Tooltips.
SWEEP_PARAMS = ["confidence", "risk_aversion", "num_scen", "L_tar", "c_max", "c_min",
                "w_max_others", "w_min_others", "time_limit", "optimality", "seed"]
INT_PARAMS = {"num_scen", "time_limit", "seed", "cardinality"}
# Sinnvolle Beispiel-Wertelisten je Parameter. Werden beim Wechsel automatisch eingetragen.
SWEEP_DEFAULTS = {
    "confidence": "0.90, 0.95, 0.99",
    "risk_aversion": "0.5, 1, 2, 5",
    "num_scen": "10000, 20000, 40000, 80000",
    "L_tar": "1.0, 1.3, 1.6, 2.0",
    "c_max": "0.0, 0.1, 0.2, 0.3",
    "c_min": "0.0, 0.05, 0.1",
    "w_max_others": "0.2, 0.3, 0.4, 0.6",
    "w_min_others": "-0.5, -0.3, -0.1, 0.0",
    "time_limit": "5, 15, 30",
    "optimality": "0.001, 0.0001, 0.00001",
    "seed": "1, 2, 3, 4, 5",
}
# Sinnvolle Beispiel-Werte je Parameter. Werden beim Wechsel des Parameters
# automatisch in das Werte-Feld uebernommen, damit dort keine unpassenden Werte
# stehen bleiben (z. B. 0.95 fuer num_scen).
SWEEP_DEFAULTS = {
    "confidence": "0.90, 0.95, 0.99",
    "risk_aversion": "0.5, 1, 2, 5",
    "num_scen": "10000, 20000, 40000, 80000",
    "L_tar": "1.0, 1.3, 1.6, 2.0",
    "c_max": "0.0, 0.1, 0.2, 0.3",
    "c_min": "0.0, 0.05, 0.1",
    "w_max_others": "0.2, 0.3, 0.4, 0.5",
    "w_min_others": "-0.1, -0.2, -0.3, -0.4",
    "time_limit": "5, 15, 30",
    "optimality": "0.001, 0.0001, 0.00001",
    "seed": "1, 2, 3, 4, 5",
}

with open(LANG_FILE, encoding="utf-8") as _fh:
    LANG = yaml.safe_load(_fh)

NAME_TO_CODE = {LANG[c]["language_name"]: c for c in LANG}
LANG_NAMES = [LANG[c]["language_name"] for c in LANG]
DEF = LANG[DEFAULT_LANG]


def _code(lang_name):
    return NAME_TO_CODE.get(lang_name, DEFAULT_LANG)


# ----------------------------- Hilfsfunktionen -----------------------------

def _gpu_choice_to_id(choice):
    return {"Auto": None, "GPU 0": 0, "GPU 1": 1}.get(choice, None)


def _gpu_id_to_choice(gpu_id):
    return {None: "Auto", 0: "GPU 0", 1: "GPU 1"}.get(gpu_id, "Auto")


def _gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=15)
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:  # noqa: BLE001
        return 1


def _parse_optional(value, cast):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    return cast(s)


def _overrides_text_to_dict(text):
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 3:
            tk, lo, hi = parts
            try:
                out[tk.upper()] = [float(lo), float(hi)]
            except ValueError:
                continue
    return out


def _overrides_dict_to_text(d):
    return "\n".join(f"{tk} {b[0]} {b[1]}" for tk, b in (d or {}).items())


def _fmtval(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def make(cls, info=None, **kw):
    """Erzeugt eine Komponente. Faengt aeltere Gradio-Versionen ohne 'info' ab."""
    if info is not None:
        try:
            return cls(info=info, **kw)
        except TypeError:
            return cls(**kw)
    return cls(**kw)


# Reihenfolge der Eingabe-Komponenten. build_cfg und cfg_to_ui_values nutzen sie exakt.
def build_cfg(dataset, num_scen, regime_start, regime_end, test_start, test_end,
              gpu_choice, no_backtest, results_dir,
              w_min_others, w_max_others, overrides_text,
              c_min, c_max, l_tar, t_tar, cvar_limit, cardinality,
              risk_aversion, confidence,
              solver_method, time_limit, optimality, seed):
    return {
        "dataset": dataset,
        "num_scen": int(num_scen),
        "regime_start": regime_start,
        "regime_end": regime_end,
        "test_start": test_start,
        "test_end": test_end,
        "results_dir": results_dir,
        "no_backtest": bool(no_backtest),
        "gpu_id": _gpu_choice_to_id(gpu_choice),
        "w_min_others": float(w_min_others),
        "w_max_others": float(w_max_others),
        "ticker_overrides": _overrides_text_to_dict(overrides_text),
        "c_min": float(c_min),
        "c_max": float(c_max),
        "L_tar": _parse_optional(l_tar, float),
        "T_tar": _parse_optional(t_tar, float),
        "cvar_limit": _parse_optional(cvar_limit, float),
        "cardinality": _parse_optional(cardinality, int),
        "risk_aversion": float(risk_aversion),
        "confidence": float(confidence),
        "solver_method": solver_method,
        "time_limit": int(time_limit),
        "optimality": float(optimality),
        "seed": _parse_optional(seed, int),
    }


def cfg_to_ui_values(cfg):
    return [
        cfg.get("dataset", "sp500"),
        cfg.get("num_scen", 10000),
        cfg.get("regime_start", "2021-01-01"),
        cfg.get("regime_end", "2024-01-01"),
        cfg.get("test_start", "2023-09-01"),
        cfg.get("test_end", "2024-07-01"),
        _gpu_id_to_choice(cfg.get("gpu_id")),
        bool(cfg.get("no_backtest", False)),
        cfg.get("results_dir", "results/run_cvar"),
        cfg.get("w_min_others", -0.3),
        cfg.get("w_max_others", 0.4),
        _overrides_dict_to_text(cfg.get("ticker_overrides", {"NVDA": [0.1, 0.6]})),
        cfg.get("c_min", 0.0),
        cfg.get("c_max", 0.2),
        "" if cfg.get("L_tar") is None else cfg.get("L_tar"),
        "" if cfg.get("T_tar") is None else cfg.get("T_tar"),
        "" if cfg.get("cvar_limit") is None else cfg.get("cvar_limit"),
        "" if cfg.get("cardinality") is None else cfg.get("cardinality"),
        cfg.get("risk_aversion", 1),
        cfg.get("confidence", 0.95),
        cfg.get("solver_method", "PDLP"),
        cfg.get("time_limit", 15),
        cfg.get("optimality", 1e-4),
        "" if cfg.get("seed") is None else cfg.get("seed"),
    ]


def list_configs():
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith((".yaml", ".yml")))


def _optimal_key(summary):
    for k in summary.get("backtest", {}):
        if "optimal" in k.lower() or "cuopt" in k.lower():
            return k
    return None


def _parse_job(results_dir):
    """Liest Solve-Zeit, Sortino, Drawdown und Plot-Pfad aus run_summary.json eines Jobs."""
    sp = os.path.join(results_dir, "run_summary.json")
    if not os.path.exists(sp):
        return None, "", "", None
    try:
        with open(sp) as fh:
            s = json.load(fh)
    except Exception:  # noqa: BLE001
        return None, "", "", None
    solve = ""
    for k, v in s.get("solve", {}).items():
        if "solve" in k.lower() and "time" in k.lower():
            try:
                solve = f"{float(v):.3f} s"
            except (TypeError, ValueError):
                pass
    sortino, drawdown = "", ""
    k = _optimal_key(s)
    if k:
        b = s["backtest"][k]
        if b.get("sortino") is not None:
            sortino = f"{b['sortino']:.2f}"
        if b.get("max_drawdown") is not None:
            drawdown = f"{b['max_drawdown'] * 100:.2f} %"
    plot = None
    if s.get("plot"):
        p = os.path.join(results_dir, s["plot"])
        if os.path.exists(p):
            plot = p
    return solve, sortino, drawdown, plot


# ------------------------------ Aktionen: Einzellauf ------------------------------

def save_config(name, lang_name, *ui_values):
    L = LANG[_code(lang_name)]
    cfg = build_cfg(*ui_values)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    safe = "".join(c for c in (name or "").strip() if c.isalnum() or c in ("-", "_")) or "config"
    path = os.path.join(CONFIG_DIR, f"{safe}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return gr.update(choices=list_configs(), value=f"{safe}.yaml"), L["msg"]["saved"].format(path=path)


def load_config(filename, lang_name):
    L = LANG[_code(lang_name)]
    if not filename:
        return cfg_to_ui_values({}) + [L["msg"]["no_config_sel"]]
    path = os.path.join(CONFIG_DIR, filename)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg_to_ui_values(cfg) + [L["msg"]["loaded"].format(path=path)]


def gpu_check_action():
    try:
        out = subprocess.run([sys.executable, "run_cvar.py", "--check"],
                             capture_output=True, text=True, timeout=120)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"GPU-Check: {exc}"


def run_action(lang_name, *ui_values):
    L = LANG[_code(lang_name)]
    T, M = L["tbl"], L["msg"]
    cfg = build_cfg(*ui_values)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    active_path = os.path.join(CONFIG_DIR, "_active.yaml")
    with open(active_path, "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    log = M["run_start"]
    yield log, None, None, None

    proc = subprocess.Popen([sys.executable, "run_cvar.py", "--config", active_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log += line
        yield log, None, None, None
    proc.wait()

    positions_df, backtest_df, plot_path = None, None, None
    summary_path = os.path.join(cfg["results_dir"], "run_summary.json")
    try:
        with open(summary_path) as fh:
            summary = json.load(fh)
        rows = [[p["ticker"], p["direction"], f'{p["weight"] * 100:.2f} %']
                for p in summary.get("positions", [])]
        if summary.get("cash") is not None:
            rows.append([T["cash_name"], T["cash_dir"], f'{summary["cash"] * 100:.2f} %'])
        if rows:
            positions_df = pd.DataFrame(rows, columns=[T["asset"], T["direction"], T["weight"]])
        bt_rows = []
        for name, vals in summary.get("backtest", {}).items():
            s = vals.get("sortino")
            dd = vals.get("max_drawdown")
            bt_rows.append([name, "" if s is None else f"{s:.2f}",
                            "" if dd is None else f"{dd * 100:.2f} %"])
        if bt_rows:
            backtest_df = pd.DataFrame(bt_rows, columns=[T["portfolio"], T["sortino"], T["drawdown"]])
        if summary.get("plot"):
            candidate = os.path.join(cfg["results_dir"], summary["plot"])
            if os.path.exists(candidate):
                plot_path = candidate
    except FileNotFoundError:
        log += "\n[i] " + M["no_summary"] + "\n"
    except Exception as exc:  # noqa: BLE001
        log += "\n[i] " + M["read_error"].format(error=exc) + "\n"

    log += M["run_done"]
    yield log, positions_df, backtest_df, plot_path


# ------------------------------ Aktionen: Vergleich ------------------------------

def snapshot(slot, results_dir, lang_name):
    L = LANG[_code(lang_name)]
    M = L["msg"]
    src_summary = os.path.join(results_dir, "run_summary.json")
    if not os.path.exists(src_summary):
        return M["no_result"].format(dir=results_dir)
    dst = os.path.join(SNAPSHOT_DIR, slot)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(src_summary, os.path.join(dst, "run_summary.json"))
    with open(src_summary) as fh:
        summary = json.load(fh)
    if summary.get("plot"):
        plot_src = os.path.join(results_dir, summary["plot"])
        if os.path.exists(plot_src):
            shutil.copy(plot_src, os.path.join(dst, summary["plot"]))
    active = os.path.join(CONFIG_DIR, "_active.yaml")
    if os.path.exists(active):
        shutil.copy(active, os.path.join(dst, "config.yaml"))
    seed = summary.get("seed")
    seed_txt = M["seed_none"] if seed is None else M["seed_set"].format(seed=seed)
    return M["snap_saved"].format(slot=slot, seed=seed_txt)


def _load_snapshot(slot):
    base = os.path.join(SNAPSHOT_DIR, slot)
    summary, cfg, img = None, None, None
    sp = os.path.join(base, "run_summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            summary = json.load(fh)
        if summary.get("plot"):
            ip = os.path.join(base, summary["plot"])
            if os.path.exists(ip):
                img = ip
    cp = os.path.join(base, "config.yaml")
    if os.path.exists(cp):
        with open(cp) as fh:
            cfg = yaml.safe_load(fh)
    return summary, cfg, img


def _positions_map(summary):
    m = {p["ticker"]: p["weight"] for p in summary.get("positions", [])}
    if summary.get("cash") is not None:
        m["Cash"] = summary["cash"]
    return m


def compare_runs(lang_name):
    L = LANG[_code(lang_name)]
    T, M = L["tbl"], L["msg"]
    sa, ca, ia = _load_snapshot("A")
    sb, cb, ib = _load_snapshot("B")
    if sa is None or sb is None:
        return None, None, None, None, None, M["need_ab"]

    ma, mb = _positions_map(sa), _positions_map(sb)
    tickers = sorted(set(ma) | set(mb), key=lambda t: -max(abs(ma.get(t, 0)), abs(mb.get(t, 0))))
    prows = []
    for t in tickers:
        wa, wb = ma.get(t, 0.0), mb.get(t, 0.0)
        prows.append([t, f"{wa * 100:.2f} %", f"{wb * 100:.2f} %", f"{(wb - wa) * 100:+.2f} %"])
    pos_df = pd.DataFrame(prows, columns=[T["asset"], T["weight_a"], T["weight_b"], T["diff"]])

    ka, kb = _optimal_key(sa), _optimal_key(sb)
    bt_rows = []
    if ka and kb:
        va, vb = sa["backtest"][ka], sb["backtest"][kb]

        def fmt(x, pct=False):
            if x is None:
                return ""
            return f"{x * 100:.2f} %" if pct else f"{x:.2f}"

        bt_rows.append([T["sortino"], fmt(va.get("sortino")), fmt(vb.get("sortino"))])
        bt_rows.append([T["drawdown"], fmt(va.get("max_drawdown"), True), fmt(vb.get("max_drawdown"), True)])
    bt_df = pd.DataFrame(bt_rows, columns=[T["metric"], T["run_a"], T["run_b"]]) if bt_rows else None

    diff_rows = []
    if ca and cb:
        for k in sorted(set(ca) | set(cb)):
            if ca.get(k) != cb.get(k):
                diff_rows.append([k, str(ca.get(k)), str(cb.get(k))])
    if not diff_rows:
        diff_rows = [[M["no_diff"], "", ""]]
    cfg_df = pd.DataFrame(diff_rows, columns=[T["parameter"], T["run_a"], T["run_b"]])

    return pos_df, bt_df, cfg_df, ia, ib, M["compare_done"]


# ------------------------------ Aktionen: Sweep ------------------------------

def run_sweep(lang_name, gpu_mode, sweep_param, values_text, *ui_values):
    """Variiert einen Parameter ueber mehrere Werte und verteilt die Jobs auf 1 oder 2 GPUs."""
    L = LANG[_code(lang_name)]
    SW = L["sweep"]
    CO, ST, MS = SW["cols"], SW["status"], SW["msg"]
    cols = [CO["scenario"], CO["value"], CO["gpu"], CO["status"],
            CO["solve_time"], CO["sortino"], CO["drawdown"]]

    raw = [x.strip() for x in (values_text or "").replace(";", ",").split(",")]
    raw = [x for x in raw if x]
    if not raw:
        yield None, None, None, MS["no_values"]
        return

    def cast(v):
        try:
            return int(v) if sweep_param in INT_PARAMS else float(v)
        except ValueError:
            return v

    values = [cast(v) for v in raw]
    base = build_cfg(*ui_values)

    want2 = str(gpu_mode).strip().startswith("2")
    n_gpu = _gpu_count()
    gpus = [0, 1] if (want2 and n_gpu >= 2) else [0]
    note = MS["one_gpu_fallback"] if (want2 and n_gpu < 2) else ""

    os.makedirs(SWEEP_DIR, exist_ok=True)
    jobs = []
    for v in values:
        safe = f"{sweep_param}_{_fmtval(v)}".replace("/", "-").replace(" ", "")
        rdir = os.path.join(SWEEP_DIR, safe)
        cfg = dict(base)
        cfg[sweep_param] = v
        cfg["results_dir"] = rdir
        jobs.append({"name": f"{sweep_param}={_fmtval(v)}", "value": _fmtval(v),
                     "gpu": "-", "status": "queued", "solve": "", "sortino": "",
                     "drawdown": "", "plot": None, "cfg": cfg, "rdir": rdir})

    lock = threading.Lock()
    q = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                j["status"] = "running"
                j["gpu"] = str(gpu)
            cfg = dict(j["cfg"])
            cfg["gpu_id"] = gpu
            yml = os.path.join(SWEEP_DIR, f"_job_{j['value']}_{gpu}.yaml".replace("/", "-").replace(" ", ""))
            ok = False
            try:
                with open(yml, "w") as fh:
                    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
                out = subprocess.run([sys.executable, "run_cvar.py", "--config", yml],
                                     capture_output=True, text=True)
                ok = (out.returncode == 0)
            except Exception:  # noqa: BLE001
                ok = False
            solve, sortino, drawdown, plot = _parse_job(j["rdir"])
            with lock:
                if ok and solve is not None:
                    j.update(status="done", solve=solve, sortino=sortino,
                             drawdown=drawdown, plot=plot)
                else:
                    j["status"] = "error"

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()

    def snapshot_df():
        with lock:
            rows = [[j["name"], j["value"], j["gpu"], ST.get(j["status"], j["status"]),
                     j["solve"], j["sortino"], j["drawdown"]] for j in jobs]
        return pd.DataFrame(rows, columns=cols)

    while any(t.is_alive() for t in threads):
        yield snapshot_df(), None, None, (MS["running"] + (" " + note if note else ""))
        time.sleep(0.5)
    for t in threads:
        t.join()

    df = snapshot_df()
    gallery = [(j["plot"], j["name"]) for j in jobs if j["plot"]]
    csv_path = os.path.join(SWEEP_DIR, "sweep_results.csv")
    try:
        df.to_csv(csv_path, index=False)
    except Exception:  # noqa: BLE001
        csv_path = None
    yield df, (gallery or None), csv_path, (MS["done"] + (" " + note if note else ""))


# ------------------------------ Oberflaeche ------------------------------

I18N = []  # Registry fuer den Sprachwechsel: (Komponente, fn(L) -> update-kwargs)


def md(key):
    return lambda L: {"value": L[key]}


def fld(key):
    return lambda L: {"label": L["fields"][key]["label"], "info": L["fields"][key].get("info")}


def btn(key):
    return lambda L: {"value": L["btn"][key]}


def outl(key):
    return lambda L: {"label": L["out"][key]}


def smd(key):
    return lambda L: {"value": L["sweep"][key]}


def sfld(key):
    return lambda L: {"label": L["sweep"][key]["label"], "info": L["sweep"][key].get("info")}


def slbl(key):
    return lambda L: {"label": L["sweep"][key]}


def tab(key):
    return lambda L: {"label": L["tabs"][key]}


with gr.Blocks(title="cuFOLIO") as demo:
    language = gr.Radio(LANG_NAMES, value=LANG[DEFAULT_LANG]["language_name"], label=DEF["lang_label"])
    I18N.append((language, lambda L: {"label": L["lang_label"]}))
    m_title = gr.Markdown(DEF["title"]); I18N.append((m_title, md("title")))
    m_sub = gr.Markdown(DEF["subtitle"]); I18N.append((m_sub, md("subtitle")))

    with gr.Tabs():
        with gr.Tab(DEF["tabs"]["single"]) as tab_single:
            I18N.append((tab_single, tab("single")))
            with gr.Row():
                with gr.Column(scale=1):
                    s1 = gr.Markdown(DEF["sec_data"]); I18N.append((s1, md("sec_data")))
                    dataset = make(gr.Dropdown, choices=["sp500", "sp100", "dow30", "global_titans"],
                                   value="sp500", label=DEF["fields"]["dataset"]["label"],
                                   info=DEF["fields"]["dataset"]["info"])
                    I18N.append((dataset, fld("dataset")))
                    with gr.Row():
                        num_scen = make(gr.Number, value=10000, precision=0,
                                        label=DEF["fields"]["num_scen"]["label"],
                                        info=DEF["fields"]["num_scen"]["info"])
                        I18N.append((num_scen, fld("num_scen")))
                        seed = make(gr.Textbox, value="42", label=DEF["fields"]["seed"]["label"],
                                    info=DEF["fields"]["seed"]["info"])
                        I18N.append((seed, fld("seed")))
                    with gr.Row():
                        regime_start = make(gr.Textbox, value="2021-01-01",
                                            label=DEF["fields"]["regime_start"]["label"],
                                            info=DEF["fields"]["regime_start"]["info"])
                        I18N.append((regime_start, fld("regime_start")))
                        regime_end = make(gr.Textbox, value="2024-01-01",
                                          label=DEF["fields"]["regime_end"]["label"],
                                          info=DEF["fields"]["regime_end"]["info"])
                        I18N.append((regime_end, fld("regime_end")))
                    with gr.Row():
                        test_start = make(gr.Textbox, value="2023-09-01",
                                          label=DEF["fields"]["test_start"]["label"],
                                          info=DEF["fields"]["test_start"]["info"])
                        I18N.append((test_start, fld("test_start")))
                        test_end = make(gr.Textbox, value="2024-07-01",
                                        label=DEF["fields"]["test_end"]["label"],
                                        info=DEF["fields"]["test_end"]["info"])
                        I18N.append((test_end, fld("test_end")))
                    gpu_choice = make(gr.Radio, choices=GPU_CHOICES, value="Auto",
                                      label=DEF["fields"]["gpu"]["label"],
                                      info=DEF["fields"]["gpu"]["info"])
                    I18N.append((gpu_choice, fld("gpu")))
                    no_backtest = make(gr.Checkbox, value=False,
                                       label=DEF["fields"]["no_backtest"]["label"],
                                       info=DEF["fields"]["no_backtest"]["info"])
                    I18N.append((no_backtest, fld("no_backtest")))
                    results_dir = make(gr.Textbox, value="results/run_cvar",
                                       label=DEF["fields"]["results_dir"]["label"],
                                       info=DEF["fields"]["results_dir"]["info"])
                    I18N.append((results_dir, fld("results_dir")))

                    s2 = gr.Markdown(DEF["sec_cvar"]); I18N.append((s2, md("sec_cvar")))
                    with gr.Row():
                        w_min_others = make(gr.Number, value=-0.3,
                                            label=DEF["fields"]["w_min_others"]["label"],
                                            info=DEF["fields"]["w_min_others"]["info"])
                        I18N.append((w_min_others, fld("w_min_others")))
                        w_max_others = make(gr.Number, value=0.4,
                                            label=DEF["fields"]["w_max_others"]["label"],
                                            info=DEF["fields"]["w_max_others"]["info"])
                        I18N.append((w_max_others, fld("w_max_others")))
                    overrides_text = make(gr.Textbox, value="NVDA 0.1 0.6", lines=3,
                                          label=DEF["fields"]["overrides_text"]["label"],
                                          info=DEF["fields"]["overrides_text"]["info"])
                    I18N.append((overrides_text, fld("overrides_text")))
                    with gr.Row():
                        c_min = make(gr.Number, value=0.0, label=DEF["fields"]["c_min"]["label"],
                                     info=DEF["fields"]["c_min"]["info"])
                        I18N.append((c_min, fld("c_min")))
                        c_max = make(gr.Number, value=0.2, label=DEF["fields"]["c_max"]["label"],
                                     info=DEF["fields"]["c_max"]["info"])
                        I18N.append((c_max, fld("c_max")))
                    with gr.Row():
                        l_tar = make(gr.Textbox, value="1.6", label=DEF["fields"]["l_tar"]["label"],
                                     info=DEF["fields"]["l_tar"]["info"])
                        I18N.append((l_tar, fld("l_tar")))
                        t_tar = make(gr.Textbox, value="", label=DEF["fields"]["t_tar"]["label"],
                                     info=DEF["fields"]["t_tar"]["info"])
                        I18N.append((t_tar, fld("t_tar")))
                    with gr.Row():
                        cvar_limit = make(gr.Textbox, value="", label=DEF["fields"]["cvar_limit"]["label"],
                                          info=DEF["fields"]["cvar_limit"]["info"])
                        I18N.append((cvar_limit, fld("cvar_limit")))
                        cardinality = make(gr.Textbox, value="", label=DEF["fields"]["cardinality"]["label"],
                                           info=DEF["fields"]["cardinality"]["info"])
                        I18N.append((cardinality, fld("cardinality")))
                    with gr.Row():
                        risk_aversion = make(gr.Number, value=1, label=DEF["fields"]["risk_aversion"]["label"],
                                             info=DEF["fields"]["risk_aversion"]["info"])
                        I18N.append((risk_aversion, fld("risk_aversion")))
                        confidence = make(gr.Number, value=0.95, label=DEF["fields"]["confidence"]["label"],
                                          info=DEF["fields"]["confidence"]["info"])
                        I18N.append((confidence, fld("confidence")))

                    s3 = gr.Markdown(DEF["sec_solver"]); I18N.append((s3, md("sec_solver")))
                    with gr.Row():
                        solver_method = make(gr.Dropdown, choices=["PDLP"], value="PDLP",
                                             label=DEF["fields"]["solver_method"]["label"],
                                             info=DEF["fields"]["solver_method"]["info"])
                        I18N.append((solver_method, fld("solver_method")))
                        time_limit = make(gr.Number, value=15, precision=0,
                                          label=DEF["fields"]["time_limit"]["label"],
                                          info=DEF["fields"]["time_limit"]["info"])
                        I18N.append((time_limit, fld("time_limit")))
                        optimality = make(gr.Number, value=1e-4,
                                          label=DEF["fields"]["optimality"]["label"],
                                          info=DEF["fields"]["optimality"]["info"])
                        I18N.append((optimality, fld("optimality")))

                with gr.Column(scale=1):
                    s4 = gr.Markdown(DEF["sec_config"]); I18N.append((s4, md("sec_config")))
                    with gr.Row():
                        config_dropdown = make(gr.Dropdown, choices=list_configs(),
                                               label=DEF["fields"]["config_dropdown"]["label"],
                                               info=DEF["fields"]["config_dropdown"]["info"],
                                               interactive=True)
                        I18N.append((config_dropdown, fld("config_dropdown")))
                        load_btn = gr.Button(DEF["btn"]["load"]); I18N.append((load_btn, btn("load")))
                    with gr.Row():
                        config_name = make(gr.Textbox, value="mein-setup",
                                           label=DEF["fields"]["config_name"]["label"],
                                           info=DEF["fields"]["config_name"]["info"])
                        I18N.append((config_name, fld("config_name")))
                        save_btn = gr.Button(DEF["btn"]["save"]); I18N.append((save_btn, btn("save")))
                    config_status = gr.Markdown("")

                    s5 = gr.Markdown(DEF["sec_run"]); I18N.append((s5, md("sec_run")))
                    with gr.Row():
                        check_btn = gr.Button(DEF["btn"]["gpu_check"]); I18N.append((check_btn, btn("gpu_check")))
                        run_btn = gr.Button(DEF["btn"]["run"], variant="primary"); I18N.append((run_btn, btn("run")))
                    log_box = gr.Textbox(label=DEF["out"]["log"], lines=16, max_lines=16)
                    I18N.append((log_box, outl("log")))

                    s6 = gr.Markdown(DEF["sec_results"]); I18N.append((s6, md("sec_results")))
                    positions_table = gr.Dataframe(label=DEF["out"]["positions"], interactive=False)
                    I18N.append((positions_table, outl("positions")))
                    backtest_table = gr.Dataframe(label=DEF["out"]["backtest"], interactive=False)
                    I18N.append((backtest_table, outl("backtest")))
                    plot_image = gr.Image(label=DEF["out"]["plot"], type="filepath")
                    I18N.append((plot_image, outl("plot")))

            cmp_h = gr.Markdown(DEF["cmp_header"]); I18N.append((cmp_h, md("cmp_header")))
            cmp_i = gr.Markdown(DEF["cmp_intro"]); I18N.append((cmp_i, md("cmp_intro")))
            with gr.Row():
                snap_a_btn = gr.Button(DEF["btn"]["snap_a"]); I18N.append((snap_a_btn, btn("snap_a")))
                snap_b_btn = gr.Button(DEF["btn"]["snap_b"]); I18N.append((snap_b_btn, btn("snap_b")))
                compare_btn = gr.Button(DEF["btn"]["compare"], variant="primary")
                I18N.append((compare_btn, btn("compare")))
            compare_status = gr.Markdown("")
            cmp_config = gr.Dataframe(label=DEF["out"]["cmp_config"], interactive=False)
            I18N.append((cmp_config, outl("cmp_config")))
            with gr.Row():
                cmp_positions = gr.Dataframe(label=DEF["out"]["cmp_positions"], interactive=False)
                I18N.append((cmp_positions, outl("cmp_positions")))
                cmp_backtest = gr.Dataframe(label=DEF["out"]["cmp_backtest"], interactive=False)
                I18N.append((cmp_backtest, outl("cmp_backtest")))
            with gr.Row():
                cmp_img_a = gr.Image(label=DEF["out"]["cmp_img_a"], type="filepath")
                I18N.append((cmp_img_a, outl("cmp_img_a")))
                cmp_img_b = gr.Image(label=DEF["out"]["cmp_img_b"], type="filepath")
                I18N.append((cmp_img_b, outl("cmp_img_b")))

        with gr.Tab(DEF["tabs"]["sweep"]) as tab_sweep:
            I18N.append((tab_sweep, tab("sweep")))
            sw_h = gr.Markdown(DEF["sweep"]["header"]); I18N.append((sw_h, smd("header")))
            sw_i = gr.Markdown(DEF["sweep"]["intro"]); I18N.append((sw_i, smd("intro")))
            with gr.Row():
                sweep_param = make(gr.Dropdown, choices=SWEEP_PARAMS, value="confidence",
                                   label=DEF["sweep"]["param"]["label"],
                                   info=DEF["sweep"]["param"]["info"])
                I18N.append((sweep_param, sfld("param")))
                sweep_values = make(gr.Textbox, value="0.90, 0.95, 0.99",
                                    label=DEF["sweep"]["values"]["label"],
                                    info=DEF["sweep"]["values"]["info"])
                I18N.append((sweep_values, sfld("values")))
                sweep_gpu = make(gr.Radio, choices=GPU_MODE_CHOICES, value="1 GPU",
                                 label=DEF["sweep"]["gpu_mode"]["label"],
                                 info=DEF["sweep"]["gpu_mode"]["info"])
                I18N.append((sweep_gpu, sfld("gpu_mode")))
            sweep_run = gr.Button(DEF["sweep"]["run_btn"], variant="primary")
            I18N.append((sweep_run, slbl("run_btn")))
            sweep_msg = gr.Markdown("")
            sweep_status = gr.Dataframe(label=DEF["sweep"]["status_label"], interactive=False)
            I18N.append((sweep_status, slbl("status_label")))
            sweep_gallery = gr.Gallery(label=DEF["sweep"]["gallery_label"], columns=3, height="auto")
            I18N.append((sweep_gallery, slbl("gallery_label")))
            sweep_csv = gr.File(label=DEF["sweep"]["csv_label"])
            I18N.append((sweep_csv, slbl("csv_label")))

    # Eingabe-Komponenten in exakt der Reihenfolge von build_cfg / cfg_to_ui_values
    config_inputs = [
        dataset, num_scen, regime_start, regime_end, test_start, test_end,
        gpu_choice, no_backtest, results_dir,
        w_min_others, w_max_others, overrides_text,
        c_min, c_max, l_tar, t_tar, cvar_limit, cardinality,
        risk_aversion, confidence,
        solver_method, time_limit, optimality, seed,
    ]

    def set_language(lang_name):
        L = LANG[_code(lang_name)]
        return [gr.update(**fn(L)) for _, fn in I18N]

    language.change(set_language, inputs=[language], outputs=[c for c, _ in I18N])

    save_btn.click(save_config, inputs=[config_name, language] + config_inputs,
                   outputs=[config_dropdown, config_status])
    load_btn.click(load_config, inputs=[config_dropdown, language],
                   outputs=config_inputs + [config_status])
    check_btn.click(gpu_check_action, inputs=None, outputs=[log_box])
    run_btn.click(run_action, inputs=[language] + config_inputs,
                  outputs=[log_box, positions_table, backtest_table, plot_image])

    snap_a_btn.click(lambda rd, ln: snapshot("A", rd, ln), inputs=[results_dir, language],
                     outputs=[compare_status])
    snap_b_btn.click(lambda rd, ln: snapshot("B", rd, ln), inputs=[results_dir, language],
                     outputs=[compare_status])
    compare_btn.click(compare_runs, inputs=[language],
                      outputs=[cmp_positions, cmp_backtest, cmp_config, cmp_img_a, cmp_img_b, compare_status])

    sweep_param.change(lambda p: SWEEP_DEFAULTS.get(p, ""),
                       inputs=[sweep_param], outputs=[sweep_values])
    sweep_run.click(run_sweep, inputs=[language, sweep_gpu, sweep_param, sweep_values] + config_inputs,
                    outputs=[sweep_status, sweep_gallery, sweep_csv, sweep_msg])

    # Beim Wechsel des Sweep-Parameters eine passende Beispiel-Werteliste eintragen
    sweep_param.change(lambda p: SWEEP_DEFAULTS.get(p, ""),
                       inputs=[sweep_param], outputs=[sweep_values])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

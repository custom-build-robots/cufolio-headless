# cuFOLIO ohne Jupyter

Ein kleines, headless ausführbares Python-Skript für die CVaR-Portfolio-Optimierung mit [NVIDIAs cuFOLIO-Blueprint](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO). Es ersetzt das offizielle Beispiel-Notebook `notebooks/cvar_basic.ipynb` durch ein einzelnes Skript, das sich versionieren, per `systemd` oder Cron starten und sauber automatisieren lässt.

Getestet auf einem Homelab-Server mit zwei NVIDIA RTX A6000 (Ampere). cuFOLIO empfiehlt offiziell einen H100, aber für den reinen Durchlauf reicht Ampere problemlos.

> **Das hier ist erst der Anfang.** Das Repo enthält bewusst nur das Skript und das Nötigste zum Loslegen. Die ausführliche Schritt-für-Schritt-Anleitung mit allen Befehlen, Stolpersteinen und den echten Messwerten findest Du im Blogpost:
>
> **➡️ [GPU-Portfolio-Optimierung per Skript auf Dual RTX A6000](https://ai-box.eu/news/gpu-portfolio-optimierung-per-skript-auf-dual-rtx-a6000/2657/)**

## Was das Skript macht

Es bildet die sechs Stufen des Notebooks 1:1 ab, nur ohne Jupyter:

```
Daten (yfinance) -> Log-Returns -> KDE-Szenarien (GPU) -> Mean-CVaR -> cuOpt-Solve (GPU) -> Backtest
```

Am Ende liegen ein optimiertes Portfolio (`portfolio.json`) und eine Backtest-Grafik im Ergebnisordner.

## Schnellstart

Das Skript braucht das cuFOLIO-Repo und seine GPU-Umgebung. Es gehört in den Repo-Root, weil es `cufolio` importiert.

```bash
# GPU-Container mit CUDA 13 starten (Projektordner wird gemountet)
docker run --gpus all -it --rm -v "$PWD":/workspace/host --ipc=host nvcr.io/nvidia/pytorch:25.10-py3

# im Container: in den gemounteten Ordner, cuFOLIO klonen, Umgebung aufsetzen
cd /workspace/host
git clone https://github.com/NVIDIA-AI-Blueprints/cuFOLIO.git
cd cuFOLIO
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync --extra cuda13

# run_cvar.py aus diesem Repo in den cuFOLIO-Ordner legen, dann:
uv run python run_cvar.py --check   # Smoke-Test: GPUs + Stack sichtbar?
uv run python run_cvar.py           # voller Lauf
```

Die genauen Hintergründe zu jedem Schritt stehen im Blogpost.

## Optionen

```
--check               Nur prüfen: GPUs und cuOpt/cuML/cufolio sichtbar?
--dataset NAME        sp500 (Standard), sp100, dow30, global_titans
--num-scen N          Anzahl KDE-Szenarien (Standard 10000)
--regime-range A B    Optimierungszeitraum (Standard 2021-01-01 2024-01-01)
--test-range A B      Backtest-Zeitraum (Standard 2023-09-01 2024-07-01)
--results-dir PFAD    Ausgabeverzeichnis (Standard results/run_cvar)
--no-backtest         Backtest überspringen
--gpu-id N            Prozess auf eine bestimmte GPU pinnen
```

## Herkunft und Credits

- Basiert auf dem NVIDIA-AI-Blueprint **cuFOLIO**, konkret auf `notebooks/cvar_basic.ipynb`. Quelle: [github.com/NVIDIA-AI-Blueprints/cuFOLIO](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO) (Apache-2.0).
- Die Überführung in ein headless Skript ist mit **Claude Opus 4.8** als Coding-Assistent entstanden.

## Lizenz

cuFOLIO selbst steht unter Apache-2.0. Dieses Skript ist eine abgeleitete Hilfsdatei und wird im selben Geist bereitgestellt. Lege bei Bedarf eine eigene `LICENSE` in dieses Repo.

# SussyPoker

A machine-learning system that detects **coordinated play (collusion) between poker players**, such as chip dumping, soft play and coordinated isolation. For every alert it also points to the specific hands an investigator should review.

Built on the Kaggle competition [Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker): 2M synthetic six-max No-Limit Hold'em hands and 12k players.

> 🚧 Work in progress. Live demo, results and write-up coming soon.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
kaggle competitions download detect-suspicious-value-transfers-in-poker -p data/raw
pytest
```

The competition data is not included in this repository. Download it with the Kaggle CLI after accepting the competition rules.

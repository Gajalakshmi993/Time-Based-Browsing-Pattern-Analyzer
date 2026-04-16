# Time-Based Browsing Pattern Analyzer

Analyzes your browser history to surface productivity insights, behavior clusters,
RAM correlations, and anomalous sessions using unsupervised ML and deep learning.

---

## Quick Start

```bash
# 1. Clone / unzip project
cd browsing_analyzer

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Start RAM logger BEFORE your browsing session
python -m src.collect.ram_logger --minutes 60 &

# 5. Run full pipeline
python main.py --days 5

# 6. View report
cat reports/final_report.md

# 7. (Optional) Launch dashboard
streamlit run dashboard/app.py
```

---

## Configuration

All tunable parameters live in **`config/config.yaml`**:

| Section | Key settings |
|---|---|
| `collection` | browser type, time window days, RAM poll interval |
| `preprocessing` | timezone, deduplication, query-string removal |
| `sessionization` | inactivity gap (default 15 min) |
| `clustering` | algorithm (kmeans/gmm/dbscan), n_clusters |
| `deep_learning` | mode (autoencoder/lstm), epochs, hidden sizes |
| `recommendations` | thresholds for social ratio, late-night hours, RAM |

---

## Project Structure

```
browsing_analyzer/
├── config/
│   └── config.yaml              ← single source of truth for all params
├── data/
│   ├── raw/
│   │   ├── browsing_history.csv
│   │   ├── ram_log.csv
│   │   └── domain_category_map.csv   ← extend with your domains
│   └── processed/               ← auto-generated artifacts
├── src/
│   ├── config_loader.py
│   ├── collect/
│   │   ├── history_extractor.py  [Module 1a]
│   │   └── ram_logger.py         [Module 1b]
│   ├── prep/
│   │   ├── preprocessor.py       [Module 2]
│   │   └── sessionizer.py        [Module 3]
│   ├── analytics/
│   │   ├── ram_correlation.py    [Module 4]
│   │   └── report_generator.py   [Module 7b]
│   ├── models/
│   │   ├── clustering.py         [Module 5]
│   │   ├── autoencoder.py        [Module 6a]
│   │   └── lstm_predictor.py     [Module 6b]
│   └── recommend/
│       └── engine.py             [Module 7a]
├── dashboard/
│   └── app.py                    ← Streamlit dashboard
├── reports/
│   └── final_report.md           ← auto-generated
├── main.py                       ← full pipeline runner
└── requirements.txt
```

---

## CLI Options

```bash
python main.py --days 3            # analyze last 3 days
python main.py --skip-collect      # skip extraction (reuse existing CSVs)
python main.py --skip-ram          # no RAM log available
python main.py --dl lstm           # use LSTM instead of autoencoder
python main.py --dl none           # skip DL entirely
python main.py --browser edge      # use Edge instead of Chrome
```

---

## Privacy

- Only domain names are stored, never full URLs or query strings
- Raw browsing DB is never modified — a temporary copy is used
- No data leaves your machine
- Set `remove_query_strings: true` in config (default) for maximum privacy

---

## Extending the Domain Map

Add rows to `data/raw/domain_category_map.csv`:

```csv
domain,category
yoursite.com,work
anothersite.com,learning
```

Categories used by the system: `social_media`, `video`, `news`, `shopping`,
`learning`, `work`, `search`, `email`, `messaging`, `finance`, `health`, `unknown`

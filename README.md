# ESCO Occupation Classifier

An explainable application for matching a free-text Spanish description to ESCO
occupations. It returns the occupation name (`preferredLabel_esp`), code,
hierarchy, description, and an evidence sentence with matching words highlighted.

## Approach

This is an **information retrieval / semantic classification** problem. The application provides two modes:

- **Lexical (offline):** BM25 over the preferred label, alternative labels, and
  description, plus query-term coverage and phrase-level scoring. This prevents
  secondary context from hiding a very specific professional task. Each synonym
  is indexed separately so occupations with many alternative names are not
  penalized. It does not require an API or model downloads.
- **Hybrid:** combines BM25 (35%) with multilingual embeddings (65%) using
  `paraphrase-multilingual-MiniLM-L12-v2`. This works better for expressions
  that share few exact words.

## Requirements

- Python 3.10 or higher.
- The `data/data_oc.csv` file, included in this repository.

## Installation and Running

Clone the repository and enter its folder:

```bash
git clone https://github.com/dmanzanoa/occupation_categorization.git
cd occupation_categorization
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

In Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the basic dependencies and start the interface:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will show the local address in the terminal, usually
`http://localhost:8501`.

### Semantic Mode

The basic mode runs fully offline with BM25. To enable hybrid retrieval with
multilingual embeddings:

```bash
python -m pip install -r requirements-semantic.txt
streamlit run app.py
```

The first time you enable the **Use semantic model** toggle, the model is
downloaded. After that, it remains in the local cache.

## Command-Line Interface

You can also query the system without Streamlit:

```bash
python cli.py "Instalo sistemas eléctricos y reparo averías en edificios"
python cli.py --semantic "Ayudo a pacientes con ejercicios de rehabilitación"
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Structure

```text
.
├── app.py                    # Streamlit interface
├── cli.py                    # Command-line interface
├── data/data_oc.csv          # ESCO occupation catalog
├── src/esco_matcher/         # Loading, normalization, and ranking
└── tests/                    # Regression tests
```

## Limitations

This project retrieves candidates for occupational matching. Scores are useful
for ranking results, but they are not calibrated probabilities. When several
occupations are similar, the top candidates should be presented and additional
information should be requested instead of assuming the first result is
definitive.

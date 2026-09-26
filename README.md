# 🤖 AutoDS AI Studio
https://autods-ai-studio.streamlit.app/

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-453%20passed-brightgreen.svg)](https://pytest.org)

**AutoDS AI Studio** is a production-grade, AI-powered autonomous data science and analytics platform built with **Streamlit**, **Pandas**, **Scikit-Learn**, **OpenRouter LLMs**, and **Python**.

It features a strict **deterministic-first architecture** with an intelligent AI reasoning layer for end-to-end data ingestion, single-cell record unrolling, generalized date standardization, explicit type conversion, semantic consistency repair, automated EDA, AutoML modeling, model explainability, executive reporting, and dataset-aware conversational AI.

---

## 🌟 Core Architecture Philosophy

```
                         ┌───────────────────────────┐
                         │      Uploaded Dataset     │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Data Quality Analyzer   │
                         └─────────────┬─────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ Date Standardization │   │ Dtype Auto-Detection │   │ Quality Inspection & │
 │  Engine (DD-MM-YY)   │   │(Dates, Currencies, %)│   │ Multi-Signal Struct. │
 └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Recursive JSON Sanitizer │
                         │ (ISO Dates, Safe Primit.) │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │ Python Execution Engine   │
                         │(Deterministic Operations) │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Post-Repair Validation   │
                         │ (Catastrophic Loss Check) │
                         └─────────────┬─────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 ┌──────────────────────┐                               ┌──────────────────────┐
 │ 🧠 AI Data Scientist │                               │ 🧪 AI Chatbot        │
 │  • Profiling         │                               │  • Dataset-Aware Q&A │
 │  • EDA & Visuals     │                               │  • Semantic Insights │
 │  • ⚙️ AutoML Modeling│                               └──────────────────────┘
 │  • Explainability    │
 │  • Executive Reports │
 └──────────────────────┘
```

> **Safety & Integrity Guarantee:**
> The AI never directly mutates DataFrames. The AI reasons on compact, sanitized metadata profiles. Python executes all deterministic transformations, and the post-repair validator inspects results before committing to state, ensuring zero silent data loss.

---

## 🚀 Key Features

### 1. Robust Multi-Format Data Ingestion
- Ingests **CSV**, **Excel (`.xlsx`, `.xls`)**, and **Parquet (`.parquet`)**.
- Automatic archiving of raw files to `data/raw/` with timestamps.
- Defensive in-memory state preservation — raw datasets are never overwritten.

### 2. Comprehensive Generalized Date Standardization Engine
- Value-and-distribution-based semantic date detection evaluating date-likeness **before** categorical classification.
- Recognizes all common international, numeric, textual, and ISO date formats (e.g. `YYYY-MM-DD`, `DD/MM/YYYY`, `MM/DD/YYYY`, `DD-MMM-YYYY`, `MMMM DD, YYYY`, dates with timestamps, two-digit years, and dotted separators).
- Formats dates to standard **`DD-MM-YY`** display while preserving native `datetime64[ns]` datatypes internally.
- Evaluates column-wide evidence to resolve ambiguous conventions (DMY vs MDY); unresolved ambiguous or invalid values become `pd.NaT` with full lineage auditing without dropping rows or columns.

### 3. Explicit User Type Conversion Engine
- **Convert to Numeric**: Converts genuine numbers, numeric strings, magnitude suffixes (`50K`, `1.5M`), currency/percentages, and natural-language written numbers (`"ten"`, `"twenty five thousand"`) to `Int64` or `Float64`. Unparseable text cells become `pd.NA` with lineage logging (never inventing arbitrary category integer codes).
- **Convert to Date**: Overrides automatic pandas classification to convert `category`, `object`, or string columns directly to `datetime64[ns]`.

### 4. Natural-Language Number Normalization & Written Parsing
- Dynamically parses numbers written as text (`"one hundred"`, `"five hundred"`, `"twenty five thousand"`, `"two million"`) and magnitude abbreviations (`"50K"`, `"1.5M"`, `"$500K"`) into clean numeric representations.

### 5. Single-Cell Embedded Record Extraction
- Autonomous multi-signal detection and unrolling of complex datasets where entire multi-field records are embedded inside individual cells/strings.

### 6. Centralized Recursive JSON Sanitizer Layer
- Centralized sanitization utility (`core/utils/json_sanitizer.py`) guaranteeing 100% JSON-safe payloads before transmission to LLMs or `json.dumps()`.
- Converts `datetime.datetime`, `pandas.Timestamp`, `np.datetime64` to ISO-8601 strings.
- Converts `NaN`, `Inf`, `-Inf`, `pd.NA`, `pd.NaT` safely to `null`.
- Unrolls NumPy integers, floats, booleans, and arrays into native Python primitives.

### 7. Generalized Consistency & Normalization Engine
- Domain-agnostic string normalization, case harmonization, and whitespace cleaning.
- Algorithmic fuzzy clustering (Levenshtein distance, Soundex phonetic matching, Token Sorting) to detect and group semantic inconsistencies and typos without hardcoded lookups.
- Interactive user preview with confidence scoring and audit trails.

### 8. Multi-Signal Table Structure Detection
- Identifies 10 table structure orientations: `NORMAL_TABLE`, `WIDE_TABLE`, `LONG_TABLE`, `PIVOT_TABLE`, `CROSSTAB`, `MULTI_HEADER`, `TIME_SERIES`, `SURVEY`, `TRANSACTIONAL`, `UNKNOWN`.
- Multi-signal evidence scoring using column name patterns, dtype uniformity, category distributions, and header regularity.
- AI structural reasoning for ambiguous schemas.

### 9. 🛠️ Autonomous Clean & Repair Engine
- **1-Click Autonomous Repair**: Deduplicates rows, imputes missing values (median/mode/constant), caps outliers (IQR), standardizes dtypes, and formats dates.
- **Granular Manual Tools**: Categorical Normalization, Basic Cleaning, Missing Value Imputation, Type Conversion & Outliers, Reshape/Structural, Rename Column, Remove Rows & Columns, **Replace Values**, and Reset.
- **🔄 Replace Values Tool**: Literal string search and replacement with empty replacement support for cell removal, strict row/column count preservation, and audit logging.
- **Column Renaming Tab**: Interactive column renaming with automatic snake_case suggestions, collision detection, and instant validation.

### 10. 📈 Automated Visual Exploratory Data Analysis (EDA)
- Interactive Plotly visualizations for correlation heatmaps, missing value matrices, univariate feature distributions, and bivariate scatter/boxplots.
- AI EDA Analyst providing natural-language interpretations of statistical patterns.

### 11. ⚙️ AutoML Modeling, Leak-Free Pipelines & Explainability
- Auto-detects task type: **Binary Classification**, **Multiclass Classification**, or **Regression**.
- Value-based datetime detection and leak-free temporal feature extraction (`year`, `month`, `day`, `dayofweek`, `hour`, `days_elapsed` relative to training set max date).
- Excludes high-cardinality non-temporal identifier columns (e.g. `customer_id`, UUIDs) while retaining high-cardinality date features.
- Benchmarks multiple Scikit-Learn models (Random Forest, Gradient Boosting, HistGradientBoosting, Logistic Regression, Ridge, Decision Trees) against Dummy Baselines.
- Stratified 5-Fold Cross-Validation for classification and 5-Fold CV for regression.
- Displays real-time leaderboard with evaluation metrics (Accuracy, F1, Precision, Recall, ROC-AUC, RMSE, MAE, R², CV Mean ± Std).
- Stores exact fitted Scikit-Learn `Pipeline` (`best_pipeline`) connected directly to Explainability and prediction workflows.
- Model explainability providing feature importance rankings mapped back to clean human-readable feature names, key drivers, and diagnostic insights with non-causation disclaimers.

### 12. 🧠 AI Data Scientist Mission Control & Executive Reports
- 6-stage unified analysis pipeline: Profile → EDA → AutoML → Explainability → AI Insights → Executive Report.
- One-click compilation of comprehensive Markdown reports ready for download.

### 13. 🧪 Dataset-Aware AI Chatbot
- Powered by OpenRouter dual-model strategy (Primary: `nvidia/nemotron-3-super-120b-a12b:free`, Fallback: `minimax/minimax-m2.7:free`).
- Answers questions about dataset dimensions, missingness, distributions, and insights grounded strictly in uploaded data.

---

## 📁 Project Structure

```
AutoDS-AI-Studio/
├── app.py                         # Streamlit application entry point & navigation
├── requirements.txt               # Production Python dependencies
├── README.md                      # Comprehensive project documentation
├── .env.example                   # Environment variables template
├── .gitignore                     # Git exclusion rules
├── pyrefly.toml                   # Type checking & linting configuration
│
├── core/                          # Application foundation & core services
│   ├── config.py                  # Application configuration & directory paths
│   ├── constants.py               # UI constants, icons & colors
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── logging.py                 # Structured Loguru logging
│   ├── state.py                   # Streamlit session state management
│   ├── types.py                   # Shared type definitions
│   ├── llm/                       # OpenRouter LLM orchestration
│   │   ├── client.py              # OpenRouter client wrapper
│   │   ├── config.py              # LLM configuration loader
│   │   ├── exceptions.py          # LLM exception classes
│   │   └── service.py             # Primary/Fallback LLM orchestration
│   ├── schemas/                   # Shared schema definitions
│   │   ├── cleaning.py            # Cleaning actions & plan schemas
│   │   ├── dataset_profile.py     # Dataset profiling & issue schemas
│   │   ├── eda.py                 # EDA statistics & correlation schemas
│   │   └── ml.py                  # ML plan, explainability & insights schemas
│   └── utils/                     # Core system utilities
│       └── json_sanitizer.py      # Centralized recursive JSON sanitization
│
├── models/                        # Pydantic data schemas
│   ├── ai.py                      # AI plan and action schemas
│   ├── dataset.py                 # Dataset metadata & profile schemas
│   ├── inspection.py              # Quality report & statistics schemas
│   ├── ml.py                      # AutoML task, model & metric schemas
│   ├── repair.py                  # Repair action, taxonomy & validation schemas
│   └── structure.py               # Structure classification & evidence schemas
│
├── services/                      # Business logic & domain engines
│   ├── ai/                        # AI client, prompt templates, planners & validators
│   ├── cleaning/                  # AI cleaning planner service
│   ├── eda/                       # Statistical EDA & AI analyzer service
│   ├── explainability/            # Model diagnostics & feature importance service
│   ├── insights/                  # Cross-stage executive insights synthesizer
│   ├── inspector/                 # Quality inspector & column metrics service
│   ├── ml/                        # AutoML task detection, training & benchmarking
│   ├── pipeline/                  # Autonomous 1-click clean pipeline
│   ├── profiler/                  # Dataset profiler & AI profile interpreter
│   ├── repair/                    # Consistency, date standardization & repair engines
│   ├── reports/                   # Executive Markdown report generation service
│   ├── structure/                 # Multi-signal table structure detection service
│   └── upload/                    # File ingestion, format validation & archiving
│
├── ui/                            # Streamlit dark SaaS user interface
│   ├── theme.py                   # Dark SaaS design system & styling
│   ├── upload.py                  # File upload & ingestion page
│   ├── profiler.py                # Dataset profiler & AI schema overview
│   ├── inspector.py               # Dataset quality health & inspection page
│   ├── structure.py               # Table structure detection & AI reasoning
│   ├── repair.py                  # Autonomous repair, granular tools & column renaming
│   ├── eda.py                     # Visual EDA & Plotly correlation dashboard
│   ├── agent_dashboard.py         # AI Data Scientist unified mission control
│   ├── modeling.py                # AutoML training, leaderboard & explainability
│   └── llm_test.py                # Dataset-aware AI Chatbot interactive page
│
├── utils/                         # Reusable helper utilities
│   ├── dataframe.py               # Safe DataFrame copy & column type helpers
│   ├── files.py                   # File path & extension utilities
│   ├── formatting.py              # Number, byte, and currency formatting
│   ├── number_parser.py           # Natural language number & magnitude parser
│   └── json_sanitizer.py          # Re-export for centralized JSON sanitization
│
└── tests/                         # Automated Unit & Integration Test Suite (453 Tests)
    ├── test_ai.py                 # AI planner, structure reasoning & repair plan tests
    ├── test_cleaning.py           # Cleaning execution tests
    ├── test_column_management.py  # Column renaming & standardization tests
    ├── test_date_engine_advanced.py # Advanced date detection & normalization tests
    ├── test_date_engine_comprehensive.py # Comprehensive date parsing tests
    ├── test_date_output_format.py # Date DD-MM-YY formatting tests
    ├── test_eda_analyst.py        # EDA analysis & statistics tests
    ├── test_enhanced_repair.py    # Repair operations & validation tests
    ├── test_explicit_numeric_conversion.py # Explicit numeric conversion tests
    ├── test_generalized_consistency_engine.py # Fuzzy string clustering tests
    ├── test_generalized_date_engine.py        # Date ambiguity & convention tests
    ├── test_generalized_engine.py             # Generalized repair pipeline tests
    ├── test_generalized_repair.py             # Autonomous repair tests
    ├── test_generalized_semantic_engine.py    # Semantic normalization tests
    ├── test_inspector.py          # Quality inspector & metrics tests
    ├── test_json_sanitizer.py     # JSON sanitizer & type serialization tests
    ├── test_llm.py                # OpenRouter LLM orchestration tests
    ├── test_ml.py                 # AutoML training & evaluation tests
    ├── test_ml_ai_planner.py      # ML AI planner tests
    ├── test_ml_integration_audit.py # End-to-end ML integration, fitted pipeline, state invalidation & error boundary tests
    ├── test_ml_pipeline_robustness.py # Robust ML pipeline & datetime feature scenario benchmarks
    ├── test_natural_language_number_normalization.py # Natural language number tests
    ├── test_pipeline.py           # 1-click pipeline integration tests
    ├── test_profiler.py           # Dataset profiler tests
    ├── test_profiler_ai.py        # AI profiler interpreter tests
    ├── test_profiler_validator.py # Profile validation tests
    ├── test_repair.py             # Granular repair tool tests
    ├── test_replace_values.py     # Literal replace values tool tests
    ├── test_row_column_removal.py # Specific row & column removal tests
    ├── test_structure.py          # Structure detection tests
    ├── test_upload.py             # Upload service tests
    └── test_utils.py              # Utility helper tests
```

---

## 🛠️ Installation & Quickstart

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AutoDS-AI-Studio.git
cd AutoDS-AI-Studio
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and add your OpenRouter API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_PRIMARY_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_FALLBACK_MODEL=minimax/minimax-m2.7:free
```

*(Note: The platform includes complete deterministic fallbacks, allowing core profiling, cleaning, EDA, and AutoML to operate even without an API key).*

### 6. Launch the App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running the Test Suite

Run all **453 automated tests** using pytest:

```bash
pytest -v
```

To run a specific test suite (e.g., Date Engine or Explicit Numeric Conversion):
```bash
pytest tests/test_date_engine_advanced.py -v
pytest tests/test_explicit_numeric_conversion.py -v
```

---



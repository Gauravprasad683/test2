# Elite Athletics Injury Risk Intelligence & Clinical Load Management System

A Python & Streamlit application for sports science and sports medicine teams. Integrates machine learning predictive modeling (`scikit-learn`), LLM protocol-grounded retrieval-augmented generation (`Groq Llama-3.3-70b-versatile` / `Llama-3.1-8b-instant`), and human-in-the-loop clinical governance.

---

## 🚀 Quick Start

### 1. Launch the Application
Run the Streamlit app directly from the project directory:
```bash
streamlit run app.py
```
*(Or via python: `py -m streamlit run app.py`)*

### 2. Configure Credentials (`credentials.json`)
The application requires `credentials.json` in the project root with your Groq API key:
```json
{
  "groq_api_key": "gsk_your_groq_api_key_here"
}
```
*Note: If `credentials.json` is missing or the key is not set, the app displays a clean `st.error()` message and allows you to enter and save your key directly in the UI or toggle offline deterministic mode.*

---

## 📁 Project Structure

```
├── app.py                         # Primary Streamlit application entry point
├── player_health_data.json        # 30-player squad roster with longitudinal telemetry
├── credentials.json               # Groq API credentials
├── medical-protocol.md            # Clinical sports medicine thresholds & load rules
├── briefing-template.md           # Clinical briefing structure for LLM RAG engine
├── synthetic_data.py              # Synthetic telemetry generator, 80/20 split & ML pipeline
├── groq_rag.py                    # Groq LLM RAG engine with rate limit exponential backoff
├── test_app_logic.py              # Test suite for ML, RAG, and guardrail validation
├── guardrails/
│   ├── guardrails_policy.md       # Comprehensive ISO/IEC 42001 Guardrails documentation
│   └── validator.py               # Programmatic input sanitization & clinical bounds
├── models/
│   ├── injury_risk_model.joblib   # Trained RandomForestClassifier
│   └── model_metadata.json        # Accuracy, ROC-AUC, confusion matrix & cost analysis
└── data/
    ├── synthetic_player_telemetry.csv
    └── test_eval_sample.csv
```

---

## 🎯 Acceptance Criteria & Demo Scenarios

During presentation, use the **Quick Showcase Demos** in the sidebar to review the 3 primary test cases:

1. **Demo 1: Low-Risk Baseline (`PLY-001` Marcus Vance)**
   - Normal workload (`ACWR = 1.05`), optimal sleep (`8.5h`), composite wellness `8.8/10`.
   - ML Risk Class: **Low Risk** (~0.0% probability).
   - Medical Status: **Cleared**. Full team training without restriction.

2. **Demo 2: High-Risk Flagged (`PLY-002` Julian Alvarez-Cruz)**
   - Acute ACWR spike (`1.62 > 1.50`), sleep deficit (`5.8h`), soreness `8.2/10`, HSR spike `+34.5%`.
   - ML Risk Class: **High Risk** (~89.3% probability).
   - Medical Status: **Rest Mandated**. 48-hour off-pitch recovery protocol.

3. **Demo 3: Edge Case / Conflicting Protocol (`PLY-003` Dominik Szabo)**
   - ACWR (`1.28`) in safe zone (< 1.30), but severe sleep deprivation (`5.2h`) and muscle soreness (`7.6/10`) with recent calf strain 42 days ago.
   - Demonstrates the **Human Medical Checkpoint**: Medical staff overrides AI to assign **Modified Training** (restricting acceleration drills).

---

## 🛡️ Clinical Guardrails Architecture

Documented under [`guardrails/guardrails_policy.md`](file:///c:/Users/ASUS/Downloads/player/guardrails/guardrails_policy.md):
- **Tier 1:** Telemetry range clamping and regex prompt injection defense.
- **Tier 2:** Clinical diagnostic scope guardrail (prohibits definitive diagnoses or unauthorized drug prescriptions).
- **Tier 3:** Grounding & hallucination suppression (fresh reads of `medical-protocol.md`).
- **Tier 4:** Groq rate-limiting exponential backoff (`2s -> 4s -> 8s`) and deterministic rule fallback.
- **Tier 5:** Human medical checkpoint sign-off gate with immutable logging.


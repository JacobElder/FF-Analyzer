# 🏈 FF-Analyzer: Agentic Fantasy Football Assistant

## Project Overview
This project is an AI-powered Fantasy Football assistant designed to help manage rosters, evaluate waiver wire players, and suggest FAAB bids.

## Core Components
- **`src/agent.py`**: The multimodal core using the Gemini API to parse screenshots and process natural language.
- **`src/state_manager.py`**: Manages the local `roster.json` file.
- **`src/wrangler.py`**: Fetches NFL data via `nfl_data_py`.
- **`src/evaluator.py`**: Logic for waiver evaluation and FAAB bidding.
- **`app.py`**: Streamlit interface.

## 🛠️ Getting Started
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Setup environment**: API key is already in `.env`.
3. **Run the dashboard**: `streamlit run app.py`

## 🧠 Agent Guidelines
- **Multimodal Tasks**: When analyzing screenshots, focus on accurately identifying names and positions.
- **State Integrity**: Always ensure updates to `roster.json` maintain the valid JSON schema.
- **FAAB Logic**: Base bid recommendations on the `value_score` and current remaining budget.

## 📈 Future Enhancements
- Integration with live league APIs (Sleeper, Yahoo, ESPN).
- More advanced `nflfastR` metrics via `rpy2`.
- Real-time injury and news tracking.

### Product Requirements Document (PRD)

**Project Name:** Fantasy-AI-Manager

**Description:** A Gemini-powered agentic application and dashboard designed to act as an automated Fantasy Football co-manager. It evaluates player data, tracks roster state, and provides actionable, FAAB-optimized waiver and starting lineup recommendations.

**1. Core Features & Requirements**

- **Dynamic State Management:** * Maintains a local `roster.json` file.
  - Capable of updating the JSON via natural language commands (e.g., "Drop my kicker and add Jake Moody") or by parsing uploaded screenshots of a fantasy app roster.
- **Multimodal Waiver Evaluation:**
  - Accepts screenshots of the available waiver wire or text lists.
  - Parses the names and cross-references them with advanced statistical models.
- **FAAB / Waiver Optimizer:**
  - A configurable toggle for FAAB vs. Standard Waiver Wire.
  - Calculates recommended FAAB bids based on remaining budget, relative player value (Value Over Replacement Player), and upcoming matchup difficulty.
- **Data Visualization Dashboard:**
  - A lightweight UI to display insights, such as target share trends, expected vs. actual fantasy points, and historical FAAB spending efficiency.

**2. Architecture & Tech Stack**

- **LLM / Orchestration:** Gemini API (via `.env` key).
- **Data Source:** `nflfastR` (R package) for fetching deep, weekly NFL stats.
- **Backend / Logic:** Python (using a bridge like `rpy2` to execute the `nflfastR` scripts, or keeping the data pipeline entirely in R and wrapping it in an API).
- **Agentic Skills:**
  - *Skill 1: Roster Vision & State:* Parses screenshots to update `roster.json`.
  - *Skill 2: NFL Data Wrangler:* Executes queries to pull the latest weekly data.
  - *Skill 3: Fantasy Evaluator:* Applies custom league scoring rules (e.g., PPR, half-PPR) to the raw data and calculates FAAB recommendations.
- **UI/Dashboard:** Streamlit (Python) or Shiny (R) for rapid visual deployment.

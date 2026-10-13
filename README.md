# 🏈 Agentic Fantasy Football Assistant (FF-Analyzer)

An AI-powered assistant that uses the Gemini API (multimodal) to help you dominate your fantasy football leagues. Manage multiple leagues, parse roster screenshots, and get data-driven FAAB/waiver advice.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup API Key**:
   Ensure your Gemini API key is in the `.env` file (already configured during setup).
3. **Run the Dashboard**:
   From the project root, execute:
   ```bash
   streamlit run app.py
   ```

---

## 📖 User Guide

### 1. Syncing Your Leagues
The assistant supports multiple ways to bring in your data depending on the platform:

#### 📡 Sleeper (API Sync)
1. Open the **➕ Add League** expander in the sidebar.
2. Select the **Sleeper API** tab.
3. Enter your **Sleeper Username** and the **League ID** (from the Sleeper URL) or the exact **League Name**.
4. Click **Sync from Sleeper**. The app will automatically fetch your roster and remaining FAAB budget.

#### 🟣 Yahoo Fantasy (Multimodal Sync)
Since Yahoo requires a complex OAuth2 setup, the most efficient way to sync is via Gemini's vision:
1. Take a screenshot of your Yahoo Roster page.
2. In the **Screenshot Parsing** section of the dashboard, upload the image.
3. Click **Parse Roster with Gemini**. The agent will extract your team into the current league state.

### 2. Managing Multiple Leagues
- **Switching**: Use the "Select League" dropdown in the sidebar.
- **Manual Creation**: Use the **Manual** tab under **➕ Add League** to start a league from scratch.

### 2. Updating Your Roster
You can update your roster in two ways:
- **Via Screenshot (Recommended)**: 
  1. Go to the **Screenshot Parsing** section.
  2. Upload a clear image of your fantasy app's roster page.
  3. Click **Parse Roster with Gemini**. The agent will extract player names and positions automatically.
- **Via Manual Edit**: 
  You can directly edit the JSON file in the `leagues/` directory for fine-grained control over player stats and names.

### 3. Getting Waiver Wire Advice
- **Single Player Evaluation**: In the **Fantasy Evaluator & FAAB** section, type the name of the player you are considering picking up.
- **Evaluation Logic**: The assistant cross-references the player against your current roster's needs (e.g., if you are weak at RB, it prioritizes RBs).

### 4. FAAB vs. Normal Waiver Mode
- **Toggle FAAB**: Use the **💰 FAAB Mode** toggle in the sidebar.
- **FAAB Mode (ON)**: The assistant calculates a specific dollar amount bid based on your remaining budget and the player's "Value Score."
- **Normal Mode (OFF)**: The assistant provides a priority ranking (1-5) instead of a monetary bid.

### 5. Fantasy League Settings
- **Budget**: Your remaining FAAB budget is tracked per league. You can see it in the Evaluator section.
- **Scoring/Settings**: Currently, the agent uses standard logic. To customize, you can prompt the agent via the code or update the `roster.json` structure to include specific scoring formats (PPR, Half-PPR).

### 6. Asking for Advice
- Use the **Value Score** slider to tell the agent how much "hype" or "expert consensus" a player has.
- Click **Generate Recommendation** to get a detailed breakdown of whether to add the player and how much to spend.

---

## 🛠️ Project Structure

- `app.py`: The visual dashboard and user interface.
- `leagues/`: Directory containing JSON files for each of your fantasy leagues.
- `src/agent.py`: The brain of the operation. Uses Gemini to "see" screenshots and "think" about roster moves.
- `src/evaluator.py`: The math engine for FAAB bids and player scoring.
- `src/wrangler.py`: Handles fetching real NFL stats to back up the agent's advice.

## 🤖 Agentic Features
- **Multimodal Vision**: Gemini doesn't just read text; it understands the layout of fantasy apps (Sleeper, ESPN, Yahoo) from screenshots.
- **Natural Language Reasoning**: You can extend the `agent.py` to handle complex queries like "Should I drop my backup TE for a flyer on this rookie WR?"

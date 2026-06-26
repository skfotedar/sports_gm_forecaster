readme_content = """# NBA Agentic GM Forecaster

This repository contains an Agentic AI workflow acting as an advanced sports forecasting tool. It is designed to help an Executive General Manager navigate complex roster transactions by strictly adhering to the Collective Bargaining Agreement (CBA). Built in Python, the application orchestrates a Tree of Thought architecture using LangGraph, integrating live NBA data, RAG-based rule retrieval, and deterministic physics engines to evaluate parallel trade paths.

## System Architecture

The workflow is broken down into modular graph nodes that handle reasoning, data extraction, and evaluation:

* **Data Layer (`nba_extractor.py`)**: Connects to the live `nba_api` to pull current team payrolls, active rosters, and telemetry metrics (e.g., dynamically fetching live profiles for franchises like the New York Knicks).
* **RAG Component (`main.py`)**: Uses an in-memory vector store (`OpenAIEmbeddings`) to retrieve and inject dynamic CBA constraints (like the Hard Cap and First Apron) into the agent's context window.
* **Agentic Generator (Tree of Thoughts)**: An LLM-powered generator (`gm_recommender`) that spins up three parallel, highly creative trade scenarios using Pydantic structured outputs.
* **Physics Engine (`cba_validator.py`)**: A deterministic rule validator functioning as a BFS evaluator. It mathematically scores proposed trades (1.0 = Approved, 0.5 = High-Risk, 0.0 = Illegal) based on incoming/outgoing salary matching and luxury tax aprons, aggressively pruning non-compliant paths.
* **Executive Decision Node**: Synthesizes the surviving compliant paths and programmatically selects the optimal strategic move, returning a succinct justification.

## File Structure

* `main.py`: The entry point and LangGraph orchestrator compiling the state machine.
* `src/data_layer/nba_extractor.py`: NBA API integration and live data fetching.
* `src/engine/cba_validator.py`: The deterministic logic engine enforcing salary cap rules.
* `src/models/compliance.py`: Dataclasses defining trade proposals and compliance results.
* `src/graph/state.py`: TypedDict defining the payload state passed through the LangGraph workflow.

## Prerequisites & Installation

1. **Python Environment**: Ensure Python 3.10+ is installed.
2. **Dependencies**: I do not have my OpenAI API key, but you can install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
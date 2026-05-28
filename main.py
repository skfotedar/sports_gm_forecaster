# main.py
import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import the architectural building blocks we just wrote
from src.data_layer.nba_extractor import NBAMVPExtractor


# Load API credentials
load_dotenv()

def run_strawman_pipeline():
    # --- 1. RUN PRE-COMPUTATION PIPELINES (Layers 1, 2, 3) ---
    extractor = NBAMVPExtractor()
    data = extractor.fetch_basic_team_profile(team_abbr="DET")
    print(data)


if __name__ == "__main__":
    #print variables from the .json file
    print("\n--- VERIFYING CONFIGURATION VALUES ---")
    with open("config/cba_rules.json", "r") as f:
        rules = json.load(f)

    # Print a specific value from the JSON
    print(f"Target Salary Cap: ${rules.get('salary_cap'):,}")
    print(f"First Apron Limit: ${rules.get('first_apron'):,}")
    run_strawman_pipeline()
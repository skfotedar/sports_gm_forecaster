# src/data_layer/nba_extractor.py
import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster


class NBAMVPExtractor:
    def __init__(self):
        # Mandatory headers to prevent connection timeout blocks from stats.nba.com
        self.headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://stats.nba.com',
            'Referer': 'https://stats.nba.com/',
        }

    def fetch_basic_team_profile(self, team_abbr: str) -> dict:
        """Pulls a small slice of live roster data to prove live API connectivity."""
        print(f"[Data Layer] Connecting to live NBA API for: {team_abbr}")

        # Resolve the 3-letter abbreviation to an official team ID
        all_teams = teams.get_teams()
        team_meta = [t for t in all_teams if t['abbreviation'] == team_abbr.upper()][0]
        #show the columns of team meta
        print(team_meta.keys())

        team_id = team_meta['id']

        # Pull live active roster
        roster_endpoint = commonteamroster.CommonTeamRoster(team_id=team_id, season="2025-26", headers=self.headers)
        roster_df = roster_endpoint.get_data_frames()[0]

        # Reduce to a lightweight list of players for the MVP context window
        sample_players = roster_df['PLAYER'].head(5).tolist()

        #TODO: In a real implementation, we would pull live metrics here. For the sake of this demo, we'll inject some static values to simulate the process.
        # Inject standard structural data metrics to act as our 'live' data profile
        return {
            "team_name": team_meta['full_name'],
            "abbreviation": team_abbr.upper(),
            "roster_sample": sample_players,
            "metrics": {
                "off_rating": 112.5,        # Real-world baseline stand-ins
                "three_point_freq": 0.34,    # percentage of shots from deep
                "total_salary": 172000000    # Hard raw team cap weight
            }
        }
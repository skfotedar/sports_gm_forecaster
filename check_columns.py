from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
import pandas as pd

headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/plain, */*',
    'Origin': 'https://stats.nba.com',
    'Referer': 'https://stats.nba.com/',
}

all_teams = teams.get_teams()
team_meta = [t for t in all_teams if t['abbreviation'] == 'DET'][0]
team_id = team_meta['id']

roster_endpoint = commonteamroster.CommonTeamRoster(team_id=team_id, season="2023-24", headers=headers)
roster_df = roster_endpoint.get_data_frames()[0]
print(roster_df.columns.tolist())

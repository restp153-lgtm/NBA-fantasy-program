import pandas as pd

def load_player_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    return df

def filter_nba_players(df):

    NBA_TEAMS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET",
    "GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN",
    "NOP","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS",
    "TOR","UTA","WAS"
    }

    df = df[df['team_abbreviation'].isin(NBA_TEAMS)].copy()
    return df

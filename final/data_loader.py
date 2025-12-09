import pandas as pd

def load_player_data(filepath):
    """
    載入球員數據並將欄位名稱轉為小寫。
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        # 為了後續的 join 和查詢，確保 player_id 是索引
        if 'player_id' in df.columns:
            df.set_index('player_id', inplace=True)
            df.index.name = 'player_id'
        else:
            # 如果數據中沒有 player_id，創建一個
            df.insert(0, 'player_id', range(1, len(df) + 1))
            df.set_index('player_id', inplace=True)
            df.index.name = 'player_id'
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Using an empty DataFrame.")
        return pd.DataFrame()

def filter_nba_players(df):
    """
    只篩選出在 NBA 隊伍中的球員數據。
    """
    NBA_TEAMS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET",
    "GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN",
    "NOP","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS",
    "TOR","UTA","WAS"
    }

    # 確保 'team_abbreviation' 欄位存在
    if 'team_abbreviation' not in df.columns:
        print("Warning: 'team_abbreviation' column missing. Skipping team filtering.")
        return df.copy()

    df = df[df['team_abbreviation'].isin(NBA_TEAMS)].copy()
    return df

def standardize_column_names(df):
    """
    Placeholder for potential future standardization, 
    but for now, just returns the DataFrame as load_player_data handles lowercasing.
    """
    print("Column names standardized to lowercase.")
    return df
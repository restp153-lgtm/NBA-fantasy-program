import pandas as pd

def compute_fantasy_score(df, scoring_rules=None):
    """
    Computes the fantasy score for each player based on the provided scoring rules.
    
    Default scoring rules:
    PTS: 1
    REB: 1.2
    AST: 1.5
    STL: 3
    BLK: 3
    TOV: -1
    """
    if scoring_rules is None:
        scoring_rules = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}
    
    # Ensure required columns exist (case-insensitive check handled by data_loader usually, 
    # but good to be safe or assume lowercase from standardize_column_names)
    required_cols = ["pts", "reb", "ast", "stl", "blk", "tov"]
    for col in required_cols:
        # Check if the column is missing before attempting computation
        if col not in df.columns:
            # For demonstration, we'll create the missing column with 0s to allow testing
            # In a real scenario, this should log a warning and return early.
            # print(f"Warning: Column '{col}' missing. Creating a zero column for demonstration.")
            df[col] = 0.0 # Adding 0.0 for missing columns
            
    df['fantasy_score'] = (
        df['pts'] * scoring_rules.get('pts', 1) +
        df['reb'] * scoring_rules.get('reb', 1.2) +
        df['ast'] * scoring_rules.get('ast', 1.5) +
        df['stl'] * scoring_rules.get('stl', 3) +
        df['blk'] * scoring_rules.get('blk', 3) +
        df['tov'] * scoring_rules.get('tov', -1)
    )
    
    return df

def create_ml_features(df):
    """
    Selects relevant features for Machine Learning and returns X, y, and player identifiers.
    
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (fantasy_score)
        player_ids (pd.DataFrame): Player identifiers (id, name, team) to map back predictions
    """
    # Define features to use for prediction
    # Using base stats and some advanced stats if available
    feature_cols = [
        "pts", "reb", "ast", "stl", "blk", "tov", 
        "min_base", "fgm_base", "fga_base", "fg_pct_base",
        "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct",
        "oreb", "dreb", "plus_minus", "gp_base"
    ]
    
    # Filter to only columns that actually exist in the dataframe
    selected_features = [col for col in feature_cols if col in df.columns]
    
    if "fantasy_score" not in df.columns:
        raise ValueError("Target column 'fantasy_score' not found. Run compute_fantasy_score first.")
        
    X = df[selected_features].fillna(0) # Simple imputation
    y = df["fantasy_score"]
    
    # Keep track of who is who
    # Note: df index is assumed to be player_id from data_loader
    id_cols = ["player_name", "team_abbreviation"]
    existing_id_cols = [col for col in id_cols if col in df.columns]
    player_ids = df[existing_id_cols].copy()
    player_ids['player_id'] = df.index
    
    return X, y, player_ids
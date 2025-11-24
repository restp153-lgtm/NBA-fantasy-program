from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard
from fantasy_engine import simulate_match, draft_phase

def main():

    # ---- Step 1: Load Data ----
    df = load_player_data("NBA_PlayerStats_202425.csv")
    df = filter_nba_players(df)
    df = standardize_column_names(df)

    # ---- Step 2: Feature Engineering ----
    scoring_rules = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}
    df = compute_fantasy_score(df, scoring_rules)
    X, y, player_ids = create_ml_features(df)

    # ---- Step 3: Select Difficulty ----
    difficulty = input("Select difficulty (easy/medium/hard): ")

    if difficulty == "medium" or difficulty == "hard":
        draft_model = train_draft_model(X, y)
    else:
        draft_model = None

    # ---- Step 4: Draft Phase ----
    player_team, ai_team = draft_phase(df, difficulty, draft_model)

    # ---- Step 5: Simulate Match ----
    result = simulate_match(player_team, ai_team, df)

    print(f"Your score: {result['player_score']}")
    print(f"AI score: {result['ai_score']}")
    print(f"Winner: {result['winner']}")

if __name__ == "__main__":
    main()

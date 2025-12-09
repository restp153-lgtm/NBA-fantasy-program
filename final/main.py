from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard
from fantasy_engine import simulate_match, draft_phase
import pandas as pd

def main():

    # ---- Step 1: Load Data ----
    print("--- 1. Data Loading and Filtering ---")
    df = load_player_data("NBA_PlayerStats_202425.csv")
    
    if df.empty:
        print("Fatal Error: DataFrame is empty. Please check if the CSV file exists and is correctly named.")
        return

    df = filter_nba_players(df)
    df = standardize_column_names(df) 

    # ---- Step 2: Feature Engineering ----
    print("\n--- 2. Feature Engineering ---")
    scoring_rules = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}
    df = compute_fantasy_score(df, scoring_rules)
    X, y, player_ids = create_ml_features(df)

    # **修正點 1：將 player_name 欄位重新命名為 Player**
    # 確保主程式中的 df 包含 'Player' 欄位名稱
    if 'player_name' in df.columns:
        df.rename(columns={'player_name': 'Player'}, inplace=True)
    elif 'Player' not in df.columns and 'player_name' not in df.columns:
        print("Warning: Player name column (player_name or Player) not found. Roster will show IDs.")

    # ---- Step 3: Select Difficulty ----
    print("\n--- 3. Difficulty Selection and Model Training ---")
    valid_difficulties = ["easy", "medium", "hard"]
    while True:
        difficulty = input(f"Select difficulty ({'/'.join(valid_difficulties)}): ").lower()
        if difficulty in valid_difficulties:
            break
        print("Invalid difficulty selected. Please try again.")

    draft_model = None
    if difficulty == "medium" or difficulty == "hard":
        draft_model = train_draft_model(X, y)
    
    # 刪除上次修改中不必要的 merge 步驟，因為 df 已經包含所有需要的數據
    
    # ---- Step 4: Draft Phase ----
    player_team, ai_team = draft_phase(df, difficulty, draft_model)

    # ---- Step 5: Simulate Match ----
    print("\n--- 5. Match Simulation ---")
    # 注意：這裡使用的 df 是傳入 draft_phase 之前，包含 fantasy_score 的 df
    result = simulate_match(player_team, ai_team, df)

    # ---- Step 6: Display Results ----
    print("\n--- Final Results ---")
    print(f"Player's Team Score: {result['player_score']:.2f}")
    print(f"AI's Team Score: {result['ai_score']:.2f}")
    print(f"Winner: **{result['winner']}**")
    
    # 顯示隊伍陣容
    # 現在 'Player' in df.columns 應該為 True
    player_names = df.loc[player_team, 'Player'].tolist() if 'Player' in df.columns else player_team
    ai_names = df.loc[ai_team, 'Player'].tolist() if 'Player' in df.columns else ai_team
    
    print("\nPlayer's Team Roster:")
    for name in player_names:
        print(f"- {name}")
        
    print("\nAI's Team Roster:")
    for name in ai_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()
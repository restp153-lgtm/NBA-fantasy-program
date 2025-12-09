from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard # 為了讓 IDE 識別，雖然在 engine 中使用
from fantasy_engine import simulate_match, draft_phase
import pandas as pd # 需要 Pandas 來處理 DataFrame

def main():

    # ---- Step 1: Load Data ----
    # 注意: 您需要提供一個名為 "NBA_PlayerStats_202425.csv" 的文件在同一個目錄
    print("--- 1. Data Loading and Filtering ---")
    df = load_player_data("NBA_PlayerStats_202425.csv")
    
    if df.empty:
        print("Fatal Error: DataFrame is empty. Please check if the CSV file exists and is correctly named.")
        return

    df = filter_nba_players(df)
    df = standardize_column_names(df) # 確保欄位是小寫

    # ---- Step 2: Feature Engineering ----
    print("\n--- 2. Feature Engineering ---")
    scoring_rules = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}
    df = compute_fantasy_score(df, scoring_rules)
    X, y, player_ids = create_ml_features(df)

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
    
    # 將 player_ids 中的數據重新映射回 df (主要用於後續的名稱查詢)
    # 由於 load_player_data 設置了 player_id 為 index，這裡我們依賴 index
    df = df.merge(player_ids.drop(columns=['player_name', 'team_abbreviation'], errors='ignore'), 
                  left_index=True, right_index=True, how='left')


    # ---- Step 4: Draft Phase ----
    player_team, ai_team = draft_phase(df, difficulty, draft_model)

    # ---- Step 5: Simulate Match ----
    print("\n--- 5. Match Simulation ---")
    result = simulate_match(player_team, ai_team, df)

    # ---- Step 6: Display Results ----
    print("\n--- Final Results ---")
    print(f"Player's Team Score: {result['player_score']:.2f}")
    print(f"AI's Team Score: {result['ai_score']:.2f}")
    print(f"Winner: **{result['winner']}**")
    
    # 顯示隊伍陣容
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
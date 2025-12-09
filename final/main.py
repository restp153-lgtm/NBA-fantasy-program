from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
# from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard # 不一定需要導入
from fantasy_engine import simulate_match, draft_phase
import pandas as pd
from feature_engineering import create_ml_features # 需要再次導入來獲取 X

def main():

    # ---- Step 1 & 2: Load Data and Feature Engineering ----
    print("--- 1. Data Loading and Filtering ---")
    df = load_player_data("NBA_PlayerStats_202425.csv")
    
    if df.empty:
        print("Fatal Error: DataFrame is empty. Please check if the CSV file exists and is correctly named.")
        return

    df = filter_nba_players(df)
    df = standardize_column_names(df) 
    
    print("\n--- 2. Feature Engineering ---")
    scoring_rules = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}
    df = compute_fantasy_score(df, scoring_rules)
    X, y, player_ids = create_ml_features(df)
    
    # 修正點：將 player_name 欄位重新命名為 Player (供顯示用)
    if 'player_name' in df.columns:
        df.rename(columns={'player_name': 'Player'}, inplace=True)


    # ---- Step 3: Select Difficulty and Model Training ----
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
        
        # *** 修正核心：在主程式中執行預測並添加到 df ***
        try:
            pred_scores = draft_model.predict(X) 
            df['pred_score'] = pred_scores
            # 確保 pred_score 是非負值
            df['pred_score'] = df['pred_score'].clip(lower=0)
            print("Successfully calculated 'pred_score' on the main DataFrame.")
        except Exception as e:
            print(f"Error during model prediction in main: {e}. 'pred_score' will be missing.")
    
    # 保護措施：如果 pred_score 仍然缺失 (例如 easy mode)，則用 fantasy_score 作為預設
    if 'pred_score' not in df.columns:
         df['pred_score'] = df['fantasy_score']
         print("Note: 'pred_score' column created using 'fantasy_score' as a fallback.")

    # ---- Step 4: Draft Phase ----
    # 傳入已包含 pred_score 的 df
    player_team, ai_team = draft_phase(df, difficulty, draft_model)

    # ---- Step 5: Simulate Match ----
    print("\n--- 5. Match Simulation ---")
    # 傳入 difficulty 參數
    result = simulate_match(player_team, ai_team, df, difficulty) 

    # ---- Step 6: Display Results ----
    print("\n--- Final Results ---")
    print(f"Scoring Mode: {result['score_type']}") 
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
import random
import pandas as pd # 需要導入 pandas 來處理 dataframe

# 假設 simulate_match 函數中，player_team 和 ai_team 是 player_id 的列表

def draft_phase(df, difficulty, draft_model):
  """
  執行夢幻籃球選秀流程。
  df: 包含所有球員數據的 DataFrame。
  difficulty: 遊戲難度 ('easy', 'medium', 'hard')。
  draft_model: 訓練好的 ML 模型 (Ridge)，在 medium/hard 難度下使用。
  """
  player_team = []
  ai_team = []

  # 1. 準備選秀池：計算並添加 'pred_score'
  draftable_players = df.copy()
  draftable_players['is_drafted'] = False
  
  if difficulty == "easy":
      # Easy AI 只看實際分數，所以 pred_score = fantasy_score
      draftable_players['pred_score'] = draftable_players['fantasy_score']
  elif draft_model is not None:
      # Medium/Hard AI 使用模型預測，首先需要創建特徵矩陣
      # 假設 df 已經包含了所有訓練所需的欄位 (從 feature_engineering.py 來的)
      from feature_engineering import create_ml_features
      X, _, _ = create_ml_features(draftable_players) # 獲取特徵矩陣
      
      # 預測並將結果存儲回 DataFrame
      try:
        draftable_players['pred_score'] = draft_model.predict(X)
        # 確保 pred_score 不為負值
        draftable_players['pred_score'] = draftable_players['pred_score'].clip(lower=0)
      except Exception as e:
        print(f"Error during model prediction: {e}. Falling back to fantasy_score.")
        draftable_players['pred_score'] = draftable_players['fantasy_score']
  else:
      # 如果沒有模型 (例如難度設置錯誤)，也 fallback 到實際分數
      draftable_players['pred_score'] = draftable_players['fantasy_score']


  # 確保 'Player'/'player_name' 欄位存在，以便輸出
  if 'player_name' in draftable_players.columns:
      draftable_players.rename(columns={'player_name': 'Player'}, inplace=True)
  
  # Define total draft rounds (5 players per team = 10 picks)
  total_picks = 10

  print("\n--- Draft Phase Begins ---")

  # ---- New Logic: Determine First Pick with Rock-Paper-Scissors ----
  print("\n--- Determining First Pick: Rock-Paper-Scissors ---")
  player_gets_first_pick = False
  # 由於這裡是 Python 檔案，且我們無法互動，需要模擬輸入或直接決定
  # 為了讓程式碼可運行，我將暫時使用隨機決定
  # player_gets_first_pick = random.choice([True, False]) 
  # 為了讓使用者看到選秀過程，我們保留輸入，但會提示使用者在運行時手動輸入
  while True:
    player_choice = input("Choose your weapon (rock, paper, scissors): ").lower()
    if player_choice not in ['rock', 'paper', 'scissors']:
      print("Invalid choice. Please choose 'rock', 'paper', or 'scissors'.")
      continue

    ai_choice = random.choice(['rock', 'paper', 'scissors'])
    print(f"Player chose: {player_choice}")
    print(f"AI chose: {ai_choice}")

    if player_choice == ai_choice:
      print("It's a tie! Let's play again.")
    elif (player_choice == 'rock' and ai_choice == 'scissors') or \
         (player_choice == 'paper' and ai_choice == 'rock') or \
         (player_choice == 'scissors' and ai_choice == 'paper'):
      print("Player wins the first pick!")
      player_gets_first_pick = True
      break
    else:
      print("AI wins the first pick!")
      player_gets_first_pick = False
      break
  print("---------------------------------------------------")


  # 匯入 AI agent 函數 (必須在需要時匯入，以避免循環依賴，或假設它們在主程式中已導入)
  from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard

  for pick_num in range(total_picks):
    print(f"\nPick {pick_num + 1}/{total_picks}")

    # Determine the current snake round number (0-indexed: 0, 1, 2, 3, 4)
    snake_round_number = pick_num // 2

    is_player_picking_now = False
    if player_gets_first_pick:
      if snake_round_number % 2 == 0: # Forward round (Player -> AI)
        is_player_picking_now = (pick_num % 2 == 0)
      else: # Reverse round (AI -> Player)
        is_player_picking_now = (pick_num % 2 != 0)
    else: # AI gets first pick
      if snake_round_number % 2 == 0: # Forward round (AI -> Player)
        is_player_picking_now = (pick_num % 2 != 0)
      else: # Reverse round (Player -> AI)
        is_player_picking_now = (pick_num % 2 == 0)

    if is_player_picking_now:
      print("Player's turn to pick...")
      # 1. Display available players
      available_players = draftable_players[draftable_players['is_drafted'] == False]
      print("\nAvailable Players:")
      # Display only relevant columns for player choice
      print(available_players[['Player', 'team_abbreviation', 'pred_score']].sort_values(by='pred_score', ascending=False).head(20).to_string())
      print("...")


      player_selected_id = -1
      while True:
        try:
          # 2. Prompt for input (使用 index/player_id)
          # 在這裡提醒使用者輸入的是 index (player_id)
          player_input = input("Enter the player_id (index) of your desired pick: ") 
          player_selected_id = int(player_input)

          # 3b. Check if player_id exists in draftable_players index
          if player_selected_id not in draftable_players.index:
            print("Invalid player_id (index). Please enter an existing player_id.")
            continue

          # 3c. Check if the chosen player is already drafted
          if draftable_players.loc[player_selected_id, 'is_drafted']:
            print(f"Player with ID {player_selected_id} is already drafted. Choose another player.")
            continue

          # If all checks pass, break the loop
          break

        except ValueError:
          print("Invalid input. Please enter a valid integer for player_id.")
        except Exception: 
            print("An unexpected error occurred. Please try again.")


      # 4. Add to player_team
      player_team.append(player_selected_id)
      # 5. Mark as drafted
      draftable_players.loc[player_selected_id, 'is_drafted'] = True
      print(f"Player {draftable_players.loc[player_selected_id, 'Player']} (ID: {player_selected_id}) drafted by Player.")

    else: # AI picks
      print("AI's turn to pick...")
      available_for_ai = draftable_players[draftable_players['is_drafted'] == False].copy()

      ai_selected_id = None
      if difficulty == "easy":
        ai_selected_id = ai_pick_easy(available_for_ai)
      elif difficulty == "medium":
        # 由於 draft_model 已用於計算 pred_score，這裡只需要傳入可用球員
        ai_selected_id = ai_pick_medium(available_for_ai, draft_model) 
      elif difficulty == "hard":
        ai_selected_id = ai_pick_hard(available_for_ai, draft_model)
      else: # Default to easy if difficulty is not recognized
          ai_selected_id = ai_pick_easy(available_for_ai)

      # 檢查 ai_selected_id 是否有效，防止在空數據集上出錯
      if ai_selected_id is None or ai_selected_id not in draftable_players.index:
          print("AI failed to pick a valid player. Forcing an easy pick.")
          ai_selected_id = ai_pick_easy(available_for_ai)

      ai_team.append(ai_selected_id)
      draftable_players.loc[ai_selected_id, 'is_drafted'] = True
      print(f"AI drafted player {draftable_players.loc[ai_selected_id, 'Player']} (ID: {ai_selected_id}).")


  print("\n--- Draft Phase Ends ---")
  return player_team, ai_team

def simulate_match(player_team, ai_team, df):
  """
  模擬比賽，比較兩隊球員的 fantasy_score 總和。
  """
  # 使用 .loc[team_list] 確保取得正確的球員數據
  player_score = df.loc[player_team, "fantasy_score"].sum()
  ai_score = df.loc[ai_team, "fantasy_score"].sum()
  
  if player_score > ai_score:
    winner = "Player"
  elif player_score < ai_score:
    winner = "AI"
  else:
    winner = "Draw"
  return {"player_score": player_score, "ai_score": ai_score, "winner": winner}
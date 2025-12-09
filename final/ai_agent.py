import pandas as pd
import random
from sklearn.linear_model import Ridge # 假設我們在 ml_models.py 中用了這個模型

def ai_pick_easy(available_for_ai):
    """
    EASY AI: Picks the player with the highest actual 'fantasy_score'.
    """
    # 選擇實際分數最高的球員
    best_player = available_for_ai.sort_values(by='fantasy_score', ascending=False).iloc[0]
    return best_player.name # name of the Series/row is the player_id (index)

def ai_pick_medium(available_for_ai, draft_model):
    """
    MEDIUM AI: Picks the player with the highest predicted 'pred_score' from the model.
    """
    # 確保預測分數欄位存在
    if 'pred_score' not in available_for_ai.columns:
        # 如果沒有預測分數，降級到 Easy 邏輯
        print("Warning: 'pred_score' missing for MEDIUM AI. Falling back to EASY pick.")
        return ai_pick_easy(available_for_ai)
        
    # 選擇預測分數最高的球員
    best_player = available_for_ai.sort_values(by='pred_score', ascending=False).iloc[0]
    return best_player.name # player_id

def ai_pick_hard(available_for_ai, draft_model):
    """
    HARD AI: Picks a player with a high predicted 'pred_score', 
    but introduces a slight randomness to simulate different strategies/sleepers.
    
    Strategy: Pick a player from the top 5 predicted scores.
    """
    if 'pred_score' not in available_for_ai.columns:
        print("Warning: 'pred_score' missing for HARD AI. Falling back to EASY pick.")
        return ai_pick_easy(available_for_ai)
    
    # 選擇前 5 名預測分數的球員
    top_players = available_for_ai.sort_values(by='pred_score', ascending=False).head(5)
    
    # 隨機從這前 5 名中挑選一位
    selected_player = random.choice(top_players.index.tolist())
    
    return selected_player
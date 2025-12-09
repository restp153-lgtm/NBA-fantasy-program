import streamlit as st
import pandas as pd
import random
import io
import os # 引入 os 模組用於路徑檢查 (可選，但有助於除錯)

# 導入所有本地模組
from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
from fantasy_engine import simulate_match # draft_phase 保持在 engine.py 中
from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard

# ----------------------------------------------------
# 0. 固定配置與常數
# ----------------------------------------------------
# *** 修正點 1: 固定數據檔案路徑 ***
# 假設 NBA_PlayerStats_202425.csv 檔案與 stream.py 位於相同目錄
DATA_FILEPATH = "NBA_PlayerStats_202425.csv"

TOTAL_PICKS = 10 
SCORING_RULES = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}

# ----------------------------------------------------
# 1. 初始化 Session State (保持不變)
# ----------------------------------------------------
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'UPLOAD' # 狀態名稱不變，但代表自動載入
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'draftable_players' not in st.session_state:
    st.session_state.draftable_players = pd.DataFrame()
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 'easy'
if 'draft_model' not in st.session_state:
    st.session_state.draft_model = None
if 'player_team' not in st.session_state:
    st.session_state.player_team = []
if 'ai_team' not in st.session_state:
    st.session_state.ai_team = []
if 'player_gets_first_pick' not in st.session_state:
    st.session_state.player_gets_first_pick = None
if 'current_pick' not in st.session_state:
    st.session_state.current_pick = 0

# ----------------------------------------------------
# 2. 數據處理函數
# ----------------------------------------------------

@st.cache_data(show_spinner="正在自動載入與處理數據...")
# *** 修正點 2: 函數簽名變更，接受 filepath 而非 uploaded_file ***
def process_data(filepath, selected_difficulty):
    """執行數據載入、特徵工程和模型訓練/預測的步驟。"""
    
    # 讀取數據 (直接從路徑讀取)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"錯誤：找不到數據檔案於路徑: {filepath}。請確認檔案已存在於部署目錄中。")
        return pd.DataFrame(), None # 回傳空 DataFrame 和 None model
    except Exception as e:
        st.error(f"讀取數據時發生錯誤: {e}")
        return pd.DataFrame(), None

    # ---- 1. Data Loading and Filtering ----
    df.columns = [c.lower() for c in df.columns] 
    df = filter_nba_players(df)
    
    # 確保 player_id 設置為索引
    if 'player_id' not in df.columns:
        df.insert(0, 'player_id', range(1, len(df) + 1))
    df.set_index('player_id', inplace=True)
    df.index.name = 'player_id'

    # ---- 2. Feature Engineering ----
    df = compute_fantasy_score(df, SCORING_RULES)
    # 檢查數據是否足夠訓練模型
    if df.shape[0] < 5: 
        st.warning("數據不足，無法訓練模型。模型將被禁用。")

    X, y, _ = create_ml_features(df)
    
    # 修正：將 player_name 欄位重新命名為 Player (供顯示用)
    if 'player_name' in df.columns:
        df.rename(columns={'player_name': 'Player'}, inplace=True)
    
    # ---- 3. Model Training and Prediction ----
    draft_model = None
    if selected_difficulty == "medium" or selected_difficulty == "hard":
        try:
            draft_model = train_draft_model(X, y)
            pred_scores = draft_model.predict(X) 
            df['pred_score'] = pred_scores.clip(lower=0)
        except Exception as e:
            st.warning(f"模型訓練或預測錯誤: {e}。 'pred_score' 將使用 'fantasy_score' 作為後備。")
    
    # 保護措施
    if 'pred_score' not in df.columns:
         df['pred_score'] = df['fantasy_score']

    return df, draft_model

# ----------------------------------------------------
# 3. Streamlit 界面和邏輯
# ----------------------------------------------------

st.title("🏀 NBA 夢幻籃球選秀模擬器")

# --- 側邊欄控制 ---
with st.sidebar:
    st.header("遊戲設定")
    
    # *** 移除檔案上傳器 ***
    st.info(f"數據檔案 **{DATA_FILEPATH}** 將被自動載入。")
    
    # 難度選擇保持不變
    selected_difficulty = st.selectbox(
        "選擇 AI 難度",
        options=["easy", "medium", "hard"],
        index=0
    )
    
    if st.button("啟動遊戲 / 重新開始"):
        # 重置所有狀態
        st.session_state.app_state = 'UPLOAD' # 設為 UPLOAD 狀態觸發重新載入
        st.session_state.df = pd.DataFrame()
        st.session_state.player_team = []
        st.session_state.ai_team = []
        st.session_state.player_gets_first_pick = None
        st.session_state.current_pick = 0
        st.rerun()

# --- 主要應用邏輯 ---

# 階段 1: 數據自動載入
if st.session_state.app_state == 'UPLOAD':
    
    # *** 修正點 3: 自動開始數據載入 ***
    if st.session_state.df.empty:
        # 載入數據
        st.session_state.df, st.session_state.draft_model = process_data(DATA_FILEPATH, selected_difficulty)

    if not st.session_state.df.empty:
        st.session_state.difficulty = selected_difficulty
        st.session_state.app_state = 'READY'
        st.success("數據載入與模型訓練完成！")
        st.info("請在側邊欄選擇難度後點擊 '啟動遊戲 / 重新開始' 或直接進入猜拳階段。")
    elif st.session_state.df.empty:
        # 如果 process_data 因為找不到檔案而返回空 DF
        st.warning(f"等待數據載入，請確認檔案 {DATA_FILEPATH} 已在正確位置。")
        
# 階段 2: 準備就緒 / 猜拳決定首選 (保持不變)
if st.session_state.app_state == 'READY':
    st.header("🥊 決定首選：猜拳")
    # ... (猜拳邏輯保持不變) ...
    # 確保猜拳邏輯在這裡
    if st.session_state.player_gets_first_pick is None:
        rps_col1, rps_col2, rps_col3 = st.columns(3)
        
        player_choice_options = ['剪刀', '石頭', '布']
        player_choice = rps_col2.selectbox("你的選擇", player_choice_options)
        
        if rps_col2.button("決定先後手"):
            ai_choice = random.choice(player_choice_options)
            st.session_state.ai_choice = ai_choice
            
            st.info(f"你選擇: {player_choice} vs. AI 選擇: {ai_choice}")
            
            # 判斷勝負
            if player_choice == ai_choice:
                st.info("平手！請再選一次。")
            elif (player_choice == '石頭' and ai_choice == '剪刀') or \
                 (player_choice == '剪刀' and ai_choice == '布') or \
                 (player_choice == '布' and ai_choice == '石頭'):
                st.session_state.player_gets_first_pick = True
                st.success("你贏了！你獲得第一選秀權！🎉")
                st.session_state.app_state = 'DRAFTING'
            else:
                st.session_state.player_gets_first_pick = False
                st.error("AI 贏了！AI 獲得第一選秀權！🤖")
                st.session_state.app_state = 'DRAFTING'
        
    if st.session_state.app_state == 'DRAFTING':
        st.session_state.draftable_players = st.session_state.df.copy()
        st.session_state.draftable_players['is_drafted'] = False
        st.rerun()


# 階段 3 & 4: 選秀進行中 & 遊戲結束 (保持不變)
def process_draft_pick():
    """處理單次選秀邏輯"""
    # ... (與上次提供的版本保持一致，此處省略，請確保您使用了最新的 process_draft_pick 函數) ...
    
    current_pick = st.session_state.current_pick
    # 決定當前是誰的回合 (邏輯與之前版本相同)
    snake_round_number = current_pick // 2
    is_player_picking_now = False
    
    player_gets_first_pick = st.session_state.player_gets_first_pick

    # ... (判斷 is_player_picking_now 的邏輯) ...
    if player_gets_first_pick:
        if snake_round_number % 2 == 0: # 順序輪 (P1 -> AI)
            is_player_picking_now = (current_pick % 2 == 0)
        else: # 逆序輪 (AI -> P1)
            is_player_picking_now = (current_pick % 2 != 0)
    else: # AI gets first pick
        if snake_round_number % 2 == 0: # 順序輪 (AI -> P1)
            is_player_picking_now = (current_pick % 2 != 0)
        else: # 逆序輪 (P1 -> AI)
            is_player_picking_now = (current_pick % 2 == 0)

    draftable_players = st.session_state.draftable_players

    if is_player_picking_now:
        return True # 這是玩家回合，等待 Streamlit widget 輸入
    else: # AI 回合
        st.info(f"AI 回合... 正在思考中 (難度: {st.session_state.difficulty})...")
        available_for_ai = draftable_players[draftable_players['is_drafted'] == False].copy()
        ai_selected_id = None
        
        # 呼叫 AI 邏輯
        try:
            if st.session_state.difficulty == "easy":
                ai_selected_id = ai_pick_easy(available_for_ai)
            elif st.session_state.difficulty == "medium":
                ai_selected_id = ai_pick_medium(available_for_ai, st.session_state.draft_model) 
            elif st.session_state.difficulty == "hard":
                ai_selected_id = ai_pick_hard(available_for_ai, st.session_state.draft_model)
        except Exception:
             ai_selected_id = ai_pick_easy(available_for_ai)

        # 檢查選秀結果並更新狀態
        if ai_selected_id is not None and ai_selected_id in draftable_players.index and not draftable_players.loc[ai_selected_id, 'is_drafted']:
            st.session_state.ai_team.append(ai_selected_id)
            draftable_players.loc[ai_selected_id, 'is_drafted'] = True
            player_name = draftable_players.loc[ai_selected_id, 'Player']
            st.success(f"**AI** 選擇了：**{player_name}** (ID: {ai_selected_id})")
            
            # 推進選秀
            st.session_state.current_pick += 1
            st.session_state.draftable_players = draftable_players
            st.rerun()
        else:
            st.error("AI 選秀邏輯出錯或無可用球員，遊戲結束。")
            st.session_state.app_state = 'FINISHED'
            st.rerun() # 結束遊戲
            
    return is_player_picking_now


if st.session_state.app_state == 'DRAFTING' and st.session_state.current_pick < TOTAL_PICKS:
    st.header(f"Draft Pick {st.session_state.current_pick + 1} / {TOTAL_PICKS}")

    is_player_turn = process_draft_pick() 

    # 顯示當前陣容
    team_col1, team_col2 = st.columns(2)
    with team_col1:
        st.subheader("你的隊伍 🧑 (Player)")
        # 顯示玩家隊伍 (使用 .head(5) 可能會誤導，但保持與上次一致)
        roster_to_display = st.session_state.df.loc[st.session_state.player_team, ['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']].fillna(0)
        st.write(roster_to_display)
    with team_col2:
        st.subheader("AI 隊伍 🤖")
        roster_to_display = st.session_state.df.loc[st.session_state.ai_team, ['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']].fillna(0)
        st.write(roster_to_display)


    # 玩家選秀介面
    if is_player_turn:
        st.subheader("你的選秀回合 🎯")
        
        available_players = st.session_state.draftable_players[
            st.session_state.draftable_players['is_drafted'] == False
        ].copy()
        
        if 'Player' in available_players.columns and 'fantasy_score' in available_players.columns:
            
            AI_SORT_COLUMN = 'pred_score'
            
            sorted_players_display = available_players.sort_values(by=AI_SORT_COLUMN, ascending=False)
            
            player_options = sorted_players_display.apply(
                lambda row: f"{row['Player']} ({row['team_abbreviation']}) - ID: {row.name} (FScore: {row['fantasy_score']:.2f})", 
                axis=1
            ).tolist()
            
            player_selection = st.selectbox(
                "選擇要選秀的球員 (FScore = 傳統夢幻分數)",
                options=player_options,
                index=0
            )
            
            selected_player_id_str = player_selection.split(' - ID: ')[1].split(' (FScore:')[0].strip()
            player_selected_id = int(selected_player_id_str)
            
            # 顯示可用球員 (僅前 10 位)
            st.dataframe(
                sorted_players_display[['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']]
                .rename(columns={'fantasy_score': 'Display_Score (FScore)', 'pred_score': 'AI_Pred_Score (Hidden)'}) 
                .head(10),
                use_container_width=True
            )

            if st.button(f"Draft {sorted_players_display.loc[player_selected_id, 'Player']}"):
                # 執行選秀
                if not st.session_state.draftable_players.loc[player_selected_id, 'is_drafted']:
                    st.session_state.player_team.append(player_selected_id)
                    st.session_state.draftable_players.loc[player_selected_id, 'is_drafted'] = True
                    st.success(f"你選擇了：**{sorted_players_display.loc[player_selected_id, 'Player']}**")
                    
                    st.session_state.current_pick += 1
                    
                    if st.session_state.current_pick == TOTAL_PICKS:
                        st.session_state.app_state = 'FINISHED'
                    st.rerun()
                else:
                    st.warning("該球員已被選秀！請選擇另一位。")
        else:
            st.error("數據中缺少關鍵欄位，無法進行選秀。請檢查 CSV 文件。")

# 階段 5: 遊戲結束與模擬結果 (保持不變)
if st.session_state.app_state == 'FINISHED':
    st.header("🎉 選秀結束 - 比賽模擬結果")

    result = simulate_match(
        st.session_state.player_team, 
        st.session_state.ai_team, 
        st.session_state.df, 
        st.session_state.difficulty
    )

    st.subheader(f"計分模式: **{result['score_type'].upper()}**")
    
    col_p, col_a = st.columns(2)
    col_p.metric("你的隊伍得分 (Player)", f"{result['player_score']:.2f}")
    col_a.metric("AI 隊伍得分 (AI)", f"{result['ai_score']:.2f}")

    if result['winner'] == 'Player':
        st.balloons()
        st.success(f"🏆 **恭喜，你是贏家！**")
    elif result['winner'] == 'AI':
        st.error(f"👎 **AI 獲勝！** 繼續努力！")
    else:
        st.info("🤝 **平手！**")
        
    st.subheader("最終隊伍陣容與分數")
    
    roster_df = st.session_state.df.loc[
        st.session_state.player_team + st.session_state.ai_team, 
        ['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']
    ].copy()
    roster_df['Team'] = ['Player'] * len(st.session_state.player_team) + ['AI'] * len(st.session_state.ai_team)
    
    st.dataframe(roster_df, use_container_width=True)
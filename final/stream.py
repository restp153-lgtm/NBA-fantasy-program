import streamlit as st
import pandas as pd
import random
import io

# 導入所有本地模組
# 假設這些模組都在同一個目錄下
from data_loader import load_player_data, filter_nba_players, standardize_column_names
from feature_engineering import compute_fantasy_score, create_ml_features
from ml_models import train_draft_model
from fantasy_engine import simulate_match, draft_phase
from ai_agent import ai_pick_easy, ai_pick_medium, ai_pick_hard # 為了在 app.py 中直接使用 AI 邏輯

# ----------------------------------------------------
# 1. 初始化 Session State
# ----------------------------------------------------
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'UPLOAD' # UPLOAD, READY, DRAFTING, FINISHED
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

# 總選秀輪次
TOTAL_PICKS = 10 
SCORING_RULES = {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1}

# ----------------------------------------------------
# 2. 數據處理函數
# ----------------------------------------------------

@st.cache_data(show_spinner="正在載入與處理數據...")
def process_data(uploaded_file, selected_difficulty):
    """執行數據載入、特徵工程和模型訓練/預測的步驟。"""
    
    # 讀取數據 (使用 BytesIO 處理上傳檔案)
    data = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(data))

    # ---- 1. Data Loading and Filtering ----
    df.columns = [c.lower() for c in df.columns] # 確保欄位小寫
    df = filter_nba_players(df)
    
    # 確保 player_id 設置為索引，這是整個程式碼的基礎假設
    if 'player_id' not in df.columns:
        df.insert(0, 'player_id', range(1, len(df) + 1))
    df.set_index('player_id', inplace=True)
    df.index.name = 'player_id'

    # ---- 2. Feature Engineering ----
    df = compute_fantasy_score(df, SCORING_RULES)
    X, y, _ = create_ml_features(df)
    
    # 修正：將 player_name 欄位重新命名為 Player (供顯示用)
    if 'player_name' in df.columns:
        df.rename(columns={'player_name': 'Player'}, inplace=True)
    
    # ---- 3. Model Training and Prediction ----
    draft_model = None
    if selected_difficulty == "medium" or selected_difficulty == "hard":
        draft_model = train_draft_model(X, y)
        
        # 在主程式中執行預測並添加到 df
        try:
            pred_scores = draft_model.predict(X) 
            df['pred_score'] = pred_scores.clip(lower=0)
        except Exception as e:
            st.warning(f"Error during model prediction: {e}. 'pred_score' will use 'fantasy_score' as fallback.")
    
    # 保護措施：如果 pred_score 仍然缺失 (例如 easy mode)，則用 fantasy_score 作為預設
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
    
    uploaded_file = st.file_uploader("上傳 NBA 球員數據 (CSV)", type="csv")
    
    selected_difficulty = st.selectbox(
        "選擇 AI 難度",
        options=["easy", "medium", "hard"],
        index=0,
        help="Easy: 隨機或根據傳統分數選秀; Medium/Hard: 根據機器學習模型預測分數選秀。"
    )
    
    if st.button("啟動遊戲 / 重新開始"):
        # 重置所有狀態
        st.session_state.app_state = 'UPLOAD'
        st.session_state.df = pd.DataFrame()
        st.session_state.player_team = []
        st.session_state.ai_team = []
        st.session_state.player_gets_first_pick = None
        st.session_state.current_pick = 0
        st.rerun()

# --- 主要應用邏輯 ---

# 階段 1: 數據載入
if st.session_state.app_state == 'UPLOAD' and uploaded_file is not None:
    st.session_state.df, st.session_state.draft_model = process_data(uploaded_file, selected_difficulty)
    st.session_state.difficulty = selected_difficulty
    st.session_state.app_state = 'READY'
    st.success("數據載入與模型訓練完成！")

# 階段 2: 準備就緒 / 猜拳決定首選
if st.session_state.app_state == 'READY':
    st.header("🥊 決定首選：猜拳")
    
    if st.session_state.player_gets_first_pick is None:
        rps_col1, rps_col2, rps_col3 = st.columns(3)
        
        player_choice = rps_col2.selectbox("你的選擇", ['剪刀', '石頭', '布'])
        
        if rps_col2.button("決定先後手"):
            ai_choice = random.choice(['剪刀', '石頭', '布'])
            st.session_state.ai_choice = ai_choice
            
            # 判斷勝負
            if player_choice == ai_choice:
                st.info(f"AI 選擇了 {ai_choice}，平手！請再選一次。")
            elif (player_choice == '石頭' and ai_choice == '剪刀') or \
                 (player_choice == '剪刀' and ai_choice == '布') or \
                 (player_choice == '布' and ai_choice == '石頭'):
                st.session_state.player_gets_first_pick = True
                st.success(f"你贏了！AI 選擇了 {ai_choice}。你獲得第一選秀權！🎉")
                st.session_state.app_state = 'DRAFTING'
            else:
                st.session_state.player_gets_first_pick = False
                st.error(f"AI 贏了！AI 選擇了 {ai_choice}。AI 獲得第一選秀權！🤖")
                st.session_state.app_state = 'DRAFTING'
        
    # 如果已經決定了，自動進入選秀階段
    if st.session_state.app_state == 'DRAFTING':
        st.session_state.draftable_players = st.session_state.df.copy()
        st.session_state.draftable_players['is_drafted'] = False
        st.rerun()


# 階段 3: 選秀進行中
def process_draft_pick():
    """處理單次選秀邏輯"""
    current_pick = st.session_state.current_pick
    
    # 決定當前是誰的回合
    snake_round_number = current_pick // 2
    is_player_picking_now = False
    
    player_gets_first_pick = st.session_state.player_gets_first_pick

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
        # 玩家回合：交給 Streamlit Widget 處理
        pass 
    else: # AI 回合
        st.info(f"AI 回合... 正在思考中 (難度: {st.session_state.difficulty})...")
        available_for_ai = draftable_players[draftable_players['is_drafted'] == False].copy()
        ai_selected_id = None
        
        # 呼叫 AI 邏輯
        try:
            if st.session_state.difficulty == "easy":
                ai_selected_id = ai_pick_easy(available_for_ai)
            elif st.session_state.difficulty == "medium":
                # 這裡 draft_model 不用於預測，只用於函數簽名，預測分數已在 DF 中
                ai_selected_id = ai_pick_medium(available_for_ai, st.session_state.draft_model) 
            elif st.session_state.difficulty == "hard":
                ai_selected_id = ai_pick_hard(available_for_ai, st.session_state.draft_model)
        except Exception:
             # 如果 AI 選擇失敗，使用 easy 邏輯作為 fallback
             ai_selected_id = ai_pick_easy(available_for_ai)

        # 檢查選秀結果並更新狀態
        if ai_selected_id is not None and not draftable_players.loc[ai_selected_id, 'is_drafted']:
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
        
    return is_player_picking_now


if st.session_state.app_state == 'DRAFTING' and st.session_state.current_pick < TOTAL_PICKS:
    st.header(f"Draft Pick {st.session_state.current_pick + 1} / {TOTAL_PICKS}")

    is_player_turn = process_draft_pick() # 執行 AI 回合，並判斷是否是玩家回合

    # 顯示當前陣容
    team_col1, team_col2 = st.columns(2)
    with team_col1:
        st.subheader("你的隊伍 🧑 (Player)")
        st.write(st.session_state.df.loc[st.session_state.player_team, ['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']].fillna(0).head(5))
    with team_col2:
        st.subheader("AI 隊伍 🤖")
        st.write(st.session_state.df.loc[st.session_state.ai_team, ['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']].fillna(0).head(5))


    # 玩家選秀介面
    if is_player_turn:
        st.subheader("你的選秀回合 🎯")
        
        # 準備顯示給玩家的選秀列表
        available_players = st.session_state.draftable_players[
            st.session_state.draftable_players['is_drafted'] == False
        ].copy()
        
        # 確保有 Player 和 fantasy_score 欄位
        if 'Player' in available_players.columns and 'fantasy_score' in available_players.columns:
            
            # 排序：使用 pred_score 排序，但只顯示 fantasy_score
            AI_SORT_COLUMN = 'pred_score'
            
            sorted_players_display = available_players.sort_values(by=AI_SORT_COLUMN, ascending=False)
            
            # 創建選項列表: "球員名稱 (隊伍縮寫) - ID"
            player_options = sorted_players_display.apply(
                lambda row: f"{row['Player']} ({row['team_abbreviation']}) - ID: {row.name} (FScore: {row['fantasy_score']:.2f})", 
                axis=1
            ).tolist()
            
            player_selection = st.selectbox(
                "選擇要選秀的球員 (FScore = 傳統夢幻分數)",
                options=player_options,
                index=0
            )
            
            # 獲取選中的 player_id
            selected_player_id_str = player_selection.split(' - ID: ')[1].split(' (FScore:')[0].strip()
            player_selected_id = int(selected_player_id_str)
            
            # 顯示可用球員 (僅前 10 位，避免過長)
            st.dataframe(
                sorted_players_display[['Player', 'team_abbreviation', 'fantasy_score', 'pred_score']]
                .rename(columns={'fantasy_score': 'Display_Score (FScore)', 'pred_score': 'AI_Pred_Score (Hidden)'}) # pred_score 隱藏, fantasy_score 顯示
                .head(10)
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
            st.error("數據中缺少 'Player' 或 'fantasy_score' 欄位，無法進行選秀。請檢查 CSV 文件。")

# 階段 4: 遊戲結束與模擬結果
if st.session_state.app_state == 'FINISHED':
    st.header("🎉 選秀結束 - 比賽模擬結果")

    # 呼叫 simulate_match (已確保 df 中有 pred_score)
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
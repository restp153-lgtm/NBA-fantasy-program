Fantasy Basketball Draft Simulator
===

Python程式設計基礎課程 H組 期末作業
王勁程 黃紹庭 褚勵丞 蔡紹捷
---
## 專案簡介
這是一個基於 Python 和 Streamlit 的互動式應用程式，旨在模擬 NBA 夢幻籃球（Fantasy Basketball）的選秀過程。使用者可以選擇不同的 AI 難度，與電腦進行選秀對抗。

核心特色：
- 數據驅動：使用真實的NBA球員數據進行模擬
- AI智能：導入機器學習模型（Ridge Regression）預測球員未來表現，使 AI 能夠做出有策略的選秀決策。
- 難度分級： 提供 Easy, Medium, Hard 三種難度，挑戰不同的 AI 策略。
- Web 互動： 使用 Streamlit 部署，提供直觀的網頁互動介面。

遊玩連結：
[Streamlit.app](https://restp153-lgtm-nba-fantasy-program-finalstream-mx9hel.streamlit.app/)
## 模擬流程
1. **數據載入與處理：** 自動載入 __NBA_PlayerStats_202425.csv__ ，計算基礎數據和Fantasy Score。
2. **模型訓練（M/H難度）：** 根據歷史數據訓練迴歸模型，預測球員的潛在夢幻分數 (pred_score)。
3. **首選決定：** 玩家與 AI 進行猜拳，決定選秀的先後手。
4. **選秀回合 (Snake Draft)：** 進行 10 輪的蛇形選秀，每隊選 5 名球員。
5. **比賽模擬：** 選秀結束後，比較兩隊球員的總分數，決定勝負。
    - Easy模式：用傳統Fantasy Score。
    - Median/Hard模式：使用AI預測分數（Pred_score）。

## 如何運行專案
1. 部署要求
確保您的 Streamlit 部署環境滿足以下要求：
    1. Python 3.8+
    2. 所有專案檔案（包括 Python 模組和 requirements.txt）。
2. requirements.txt內容：  
    ```
    streamlit  
    pandas  
    numpy  
    scikit-learn  
    ```
3. 檔案結構  
    ```
    nba-fantasy-program/  
    ├── requirements.txt  
    ├── final/  
    │   ├── stream.py  
    │   ├── data_loader.py  
    │   ├── feature_engineering.py  
    │   ├── ml_models.py  
    │   ├── fantasy_engine.py  
    │   ├── ai_agent.py  
    │   └── main.py (本地運行)  
    ├── NBA_PlayerStats_202425.csv (數據檔案)  
    └── ... (其他專案文件)  
    ```
    final資料夾內各檔案為經各組員分工產出模組後整合產出
4. 啟動streamlit  
    於資料夾終端機中輸入   
    ```
    streamlit run final/stream.py
    ```
## 技術核心：AI運行方式說明
AI的選秀是本遊戲的核心功能。其難度的差異主要體現在使用的評估分數和策略上：

Easy 難度：
- **選秀標準：** 直接使用球員的**Fantasy Score**進行貪婪式（Greedy）選擇
- **策略：** 始終選擇當前可選球員分數最高者

Median/Hard難度：
- **選秀標準：** 使用 機器學習模型預測分數 (pred_score) 進行選秀。
- **模型：** 採用 Ridge Regression 進行訓練。
- **Medium 策略：** 使用 pred_score 進行貪婪式選擇。
- **Hard 策略 (假設的優化)：** 使用 pred_score，但可能加入簡單的位置平衡或對高風險高回報球員的偏好等優化邏輯（需在 ai_agent.py 中實現）。

## 未來發展方向
1. 擴展遊戲規模與多玩家功能
- **多玩家/多 AI 選秀：** 允許三個或更多的隊伍（人類玩家 + AI）同時參與選秀，使選秀池的競爭更加激烈。
- **位置限制：** 導入標準夢幻籃球陣容位置要求（例如：PG, SG, SF, PF, C），使選秀策略更具挑戰性。
- **交易模擬：** 增加選秀期間或賽季期間的球員交易功能。

2. 數據獲取與即時性
- **實時爬蟲整合 (Web Scraping Integration)：** 將靜態的 CSV 數據源替換為動態的爬蟲模組（使用開源程式碼nba_api：<https://github.com/swar/nba_api>），實時更新最新賽季資料。

3. 機器學習與 AI 進階
- **進階 AI 策略：** 引入 MCTS（Monte Carlo Tree Search）或其他強化學習方法，讓 AI 能夠考慮對手的需求和隊伍的平衡性，而不僅僅是貪婪式選擇。
- **模型優化：** 嘗試更複雜的 ML 模型（如 XGBoost、神經網路）或導入時間序列分析，以提高 pred_score 的準確性。
- **自定義計分規則：** 允許使用者在 UI 上自定義夢幻分數的權重（例如：PTS: 1.5, REB: 1.0），並重新計算分數和預測。
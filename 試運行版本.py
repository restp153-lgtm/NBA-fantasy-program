import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor

NBA_TEAMS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET",
    "GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN",
    "NOP","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS",
    "TOR","UTA","WAS"
}

filepath = "C:/Users/jef81/NBA_PlayerStats_202425.csv"

def load_and_filter_players(filepath):
    df = pd.read_csv(filepath)

    # 統一欄位名稱
    df.columns = [c.lower() for c in df.columns]

    # 篩選出 NBA 球隊（Team 必須是簡寫）
    df = df[df['team_abbreviation'].isin(NBA_TEAMS)].copy()

    # 計算 fantasy 分數（最簡單版本）
    df['fantasy_score'] = (
        df['pts'] +
        df['reb'] * 1.2 +
        df['ast'] * 1.5 +
        df['stl'] * 3 +
        df['blk'] * 3 -
        df['tov']
    )

    # 依分數排序
    df = df.sort_values(by='fantasy_score', ascending=False)

    return df

df = load_and_filter_players(filepath)
fantasy_features = ["pts", "reb", "ast", "stl", "blk", "tov"]

X = df[fantasy_features]
y = (
    df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 +
    df["stl"] * 3 + df["blk"] * 3 - df["tov"]
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# 預測 fantasy score
df["pred_score"] = model.predict(X)


def show_player(idx):
    row = df.loc[idx]
    return f"{idx}. {row['player_name']} ({row['team_abbreviation']}) - fantasy_score={row['fantasy_score']:.1f}"

def ai_pick(available):
    """AI 選剩下球員中 pred_score 最高者"""
    best_idx = available["pred_score"].idxmax()
    return best_idx

# 起始資料
available = df.copy()
your_team = []
ai_team = []

print("========== NBA Fantasy Draft（你 vs AI）==========")

ROUNDS = 5
snake = False  # 控制輪向 → 從第二輪開始蛇行

for r in range(ROUNDS):
    print(f"\n===== ROUND {r+1} =====")

    if not snake:
        # 你 -> AI
        print("\n可選球員 Top 15：")
        print("\n".join(show_player(i) for i in available.nlargest(15, "fantasy_score").index))

        while True:
            try:
                pid = int(input("請輸入你要選的球員編號： "))
                if pid in available.index:
                    your_team.append(pid)
                    available = available.drop(pid)
                    break
                else:
                    print("不可選，重來。")
            except:
                print("輸入錯誤。")

        ai_choice = ai_pick(available)
        ai_team.append(ai_choice)
        print(f"AI 選擇：{show_player(ai_choice)}")
        available = available.drop(ai_choice)

    else:
        # AI -> 你
        ai_choice = ai_pick(available)
        ai_team.append(ai_choice)
        print(f"AI 選擇：{show_player(ai_choice)}")
        available = available.drop(ai_choice)

        print("\n可選球員 Top 15：")
        print("\n".join(show_player(i) for i in available.nlargest(15, "pred_score").index))

        while True:
            try:
                pid = int(input("請輸入你要選的球員編號： "))
                if pid in available.index:
                    your_team.append(pid)
                    available = available.drop(pid)
                    break
                else:
                    print("不可選，重來。")
            except:
                print("輸入錯誤。")

    # 換方向
    snake = not snake

### ------------------------------------------------------
### 5. 結果與勝負判斷
### ------------------------------------------------------
def calc_fantasy(idx_list):
    return sum(df.loc[i, "pred_score"] for i in idx_list)

your_score = calc_fantasy(your_team)
ai_score = calc_fantasy(ai_team)

print("\n========== Draft 結束 ==========")

print("\n你的球隊：")
for i in your_team:
    print(show_player(i))
print(f"你的總分：{your_score:.1f}")

print("\nAI 的球隊：")
for i in ai_team:
    print(show_player(i))
print(f"AI 總分：{ai_score:.1f}")

print("\n========== 比賽結果 ==========")
if your_score > ai_score:
    print(f"🎉 你贏了！{your_score:.1f} vs {ai_score:.1f}")
else:
    print(f"🤖 AI 獲勝！{ai_score:.1f} vs {your_score:.1f}")
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def train_draft_model(X, y):
    """
    使用 Ridge Regression 訓練選秀模型，並返回已訓練的模型。
    
    Args:
        X (pd.DataFrame): 特徵矩陣。
        y (pd.Series): 目標變數 (fantasy_score)。
        
    Returns:
        Ridge: 訓練好的 Ridge 回歸模型。
    """
    print("\n--- Training Draft Model (Ridge Regression) ---")
    
    # 初始化模型
    model = Ridge(alpha=1.0)
    
    # 執行簡單的交叉驗證來評估性能 (可選)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        # print(f"Cross-Validation MSE Scores: {scores.mean():.2f}")
    except ValueError:
        print("Warning: Could not perform cross-validation (e.g., too few samples/features).")
        
    # 在所有數據上訓練最終模型
    model.fit(X, y)
    print("Draft model training complete.")
    
    return model
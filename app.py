import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import pandas as pd
import re
import warnings
from scipy.stats import spearmanr  # Spearmanの順位相関係数計算用

# 不要な警告を抑制
warnings.filterwarnings("ignore")

# -------------------------------
# モデル関数の定義
# -------------------------------
def log_func(x, a, b, c):
    """対数関数モデル: 売上 = a * log(b * x) + c"""
    return a * np.log(b * x) + c

def quad_func(x, a, b, c):
    """二次関数モデル: 売上 = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c

def sigmoid_func(x, a, b, c):
    """シグモイド関数モデル: 売上 = a / (1 + exp(-b*(x - c)))"""
    return a / (1 + np.exp(-b * (x - c)))

def gompertz_func(x, a, b, c):
    """ゴンペルツ関数モデル: 売上 = a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))

def frac_func(x, a, b, c):
    """分数関数モデル: 売上 = (a * x + b) / (x + c)"""
    return (a * x + b) / (x + c)

def compute_r2(y_true, y_pred):
    """決定係数 R² の計算"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def get_fit_symbol(r2):
    """決定係数に基づく当てはまりの評価（◎：0.8以上、◯：0.5以上、△：0.25以上、✗：0.25未満）"""
    if r2 >= 0.8:
        return "◎"
    elif r2 >= 0.5:
        return "◯"
    elif r2 >= 0.25:
        return "△"
    else:
        return "✗"

def get_correlation_strength(rho):
    """スピアマンの順位相関係数に基づく相関の強さの評価"""
    if rho >= 0.8:
        return "強い"
    elif rho >= 0.5:
        return "やや強い"
    elif rho >= 0.3:
        return "やや弱い"
    elif rho > 0:
        return "非常に弱い"
    else:
        return "なし"

# -------------------------------
# モデル設定（辞書形式）
# -------------------------------
model_settings = {
    "対数関数": {
        "func": log_func,
        "p0": [500, 1e-4, 500],
        "bounds": ((1, 1e-7, 0), (1e6, 1, 1e7))
    },
    "二次関数": {
        "func": quad_func,
        "p0": [-1e-5, 3, 400000],
        "bounds": ((-1e-4, -100, 0), (0, 10, 1e7))
    },
    "シグモイド関数": {
        "func": sigmoid_func,
        "p0": [1e6, 1e-5, 200000],
        "bounds": ((1e5, 1e-8, 0), (1e7, 1e-3, 1e6))
    },
    "ゴンペルツ関数": {
        "func": gompertz_func,
        "p0": [1e6, 5, 1e-5],
        "bounds": ((1e5, 0, 1e-8), (1e7, 10, 1e-3))
    },
    "分数関数": {
        "func": frac_func,
        "p0": [1e6, 0, 100000],
        "bounds": ((1e5, -1e6, 1), (1e7, 1e6, 1e7))
    }
}

# グラフ上の凡例用に英語表記
model_names_en = {
    "対数関数": "Logarithmic Model",
    "二次関数": "Quadratic Model",
    "シグモイド関数": "Sigmoid Model",
    "ゴンペルツ関数": "Gompertz Model",
    "分数関数": "Rational Model"
}

# -------------------------------
# Streamlit 基本設定・タイトル（入力フォーム等は日本語）
# -------------------------------
st.set_page_config(page_title="広告費利益最大化予測", layout="wide")
st.title("広告費と売上から利益最大化予測")
st.markdown("""
このアプリケーションは、広告費と売上のデータを基に、利益が最大となる広告費の予測を行います。  

**入力形式:**  
- 一行のコンマ区切り  
- 複数行の改行区切り 
どちらにも対応しています。
""")

# -------------------------------
# モデル選択の入力（散布図選択は全モデル比較内で表示）
# -------------------------------
model_options = ["対数関数", "二次関数", "シグモイド関数", "ゴンペルツ関数", "分数関数", "全モデル比較"]
model_type = st.selectbox("モデル選択", model_options)

# -------------------------------
# 入力データの入力（デフォルト値）
# -------------------------------
default_x = "100000,200000,300000"
default_y = "400000,700000,900000"
x_input = st.text_area("広告費データ (円単位)", default_x, height=100)
y_input = st.text_area("売上データ (円単位)", default_y, height=100)

# -------------------------------
# 解析開始ボタン押下時の処理
# -------------------------------
if st.button("解析開始"):
  try:
      # 入力文字列をカンマまたは改行で分割して数値配列に変換
      x_data = np.array([float(val.strip()) for val in re.split(r'[\n,]+', x_input.strip()) if val.strip()])
      y_data = np.array([float(val.strip()) for val in re.split(r'[\n,]+', y_input.strip()) if val.strip()])
      
      if len(x_data) != len(y_data) or len(x_data) == 0:
          st.error("エラー: 広告費と売上のデータ数が一致していないか、十分なデータがありません。")
      else:
          # 入力データを表形式で表示（日本語）
          df_input = pd.DataFrame({
              "広告費 (円)": x_data,
              "売上 (円)": y_data
          })
          st.subheader("入力データ")
          st.table(df_input)
          
          # 広告費最適化のパラメータ
          ad_bounds = [(1, 10_000_000)]  # 探索範囲：1円～10,000,000円
          initial_guess = 300000.0       # 初期値：300,000円
          
          # -------------- 全モデル比較の場合 --------------
          if model_type == "全モデル比較":
              results = []
              for name, setting in model_settings.items():
                  func = setting["func"]
                  p0 = setting["p0"]
                  bounds = setting["bounds"]
                  try:
                      params, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds)
                      a, b, c = params
                      y_fit = func(x_data, a, b, c)
                      r2 = compute_r2(y_data, y_fit)
                      fit_symbol = get_fit_symbol(r2)
                      profit_func = lambda x: func(x, a, b, c) - x
                      res = minimize(lambda x: -profit_func(x), x0=initial_guess, method="L-BFGS-B", bounds=ad_bounds)
                      if res.success:
                          optimal_x = res.x[0]
                          optimal_sales = func(optimal_x, a, b, c)
                          optimal_profit = profit_func(optimal_x)
                          optimal_roas = (func(optimal_x, a, b, c) / optimal_x) * 100
                      else:
                          optimal_x, optimal_sales, optimal_profit, optimal_roas = (np.nan, np.nan, np.nan, np.nan)
                      results.append({
                          "モデル": name,
                          "最適広告費 (万円)": np.round(optimal_x / 10000, 2),
                          "予測売上 (万円)": np.round(optimal_sales / 10000, 2),
                          "予測利益 (万円)": np.round(optimal_profit / 10000, 2),
                          "ROAS (%)": np.round(optimal_roas, 2),
                          "決定係数": np.round(r2, 3),
                          "当てはまり": fit_symbol
                      })
                  except Exception as e:
                      results.append({
                          "モデル": name,
                          "最適広告費 (万円)": "計算エラー",
                          "予測売上 (万円)": "計算エラー",
                          "予測利益 (万円)": "計算エラー",
                          "ROAS (%)": "計算エラー",
                          "決定係数": "計算エラー",
                          "当てはまり": "計算エラー"
                      })
              st.subheader("全モデル比較結果")
              st.table(pd.DataFrame(results))
              
              # スピアマンの順位相関係数を計算し、テキストとして表示（日本語）
              rho, p_val = spearmanr(x_data, y_data)
              correlation_strength = get_correlation_strength(rho)
              st.write(f"スピアマンの順位相関係数：{rho:.3f} (p = {p_val:.3f})（相関：{correlation_strength}）")
              
              # 全モデル比較結果の下部に散布図（Ad Cost vs Sales）を英語表記で表示
              x_data_10k = x_data / 10000.0
              y_data_10k = y_data / 10000.0
              fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
              ax.scatter(x_data_10k, y_data_10k, color="blue", label="Observed Data")
              ax.set_title("Scatter Plot: Ad Cost vs Sales (10k JPY)")
              ax.set_xlabel("Ad Cost (×10k JPY)")
              ax.set_ylabel("Sales (×10k JPY)")
              ax.legend()
              ax.set_xlim(0, np.max(x_data_10k) * 1.05)
              ax.set_ylim(0, np.max(y_data_10k) * 1.1)
              st.pyplot(fig, use_container_width=True)
          
          # -------------- 単一モデル選択の場合 --------------
          else:
              if model_type in model_settings:
                  setting = model_settings[model_type]
                  func = setting["func"]
                  p0 = setting["p0"]
                  bounds = setting["bounds"]
              else:
                  st.error("選択したモデルはサポートされていません。")
                  st.stop()
              
              params, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds)
              a, b, c = params
              perr = np.sqrt(np.diag(pcov))
              y_fit = func(x_data, a, b, c)
              r2 = compute_r2(y_data, y_fit)
              
              st.subheader("モデルパラメータ推定結果")
              st.write(f"選択モデル: {model_type}")
              st.write(f"a = {a:.3f} (std: {perr[0]:.3f})")
              st.write(f"b = {b:.6f} (std: {perr[1]:.6f})")
              st.write(f"c = {c:.3f} (std: {perr[2]:.3f})")
              st.write(f"決定係数 R² = {r2:.3f}")
              
              profit_func = lambda x: func(x, a, b, c) - x
              res = minimize(lambda x: -profit_func(x), x0=initial_guess, method="L-BFGS-B", bounds=ad_bounds)
              if res.success:
                  optimal_x = res.x[0]
                  optimal_sales = func(optimal_x, a, b, c)
                  optimal_profit = profit_func(optimal_x)
                  optimal_roas = (func(optimal_x, a, b, c) / optimal_x) * 100
              else:
                  st.error("利益最大化の最適化が収束しませんでした。")
                  optimal_x = None
              
              if optimal_x is not None:
                  st.subheader("利益最大化の結果")
                  st.write(f"最適な広告費: {optimal_x / 10000:.2f} (10k JPY)")
                  st.write(f"予測売上: {optimal_sales / 10000:.2f} (10k JPY)")
                  st.write(f"予測利益: {optimal_profit / 10000:.2f} (10k JPY)")
                  st.write(f"予測ROAS: {optimal_roas:.2f} %")
              
              # Plot prediction over the entire ad cost range
              x_plot = np.linspace(1, ad_bounds[0][1], 300)
              y_plot = func(x_plot, a, b, c)
              profit_plot = profit_func(x_plot)
              
              x_plot_10k = x_plot / 10000
              y_plot_10k = y_plot / 10000
              profit_plot_10k = profit_plot / 10000
              x_data_10k = x_data / 10000
              y_data_10k = y_data / 10000
              optimal_x_10k = (optimal_x / 10000) if optimal_x is not None else None
              
              x_all = np.concatenate([x_data_10k, x_plot_10k])
              if optimal_x_10k is not None:
                  x_all = np.append(x_all, optimal_x_10k)
              x_min = 0
              x_max = np.max(x_all) * 1.05
              
              y_all_sales = np.concatenate([y_data_10k, y_plot_10k])
              y_max_sales = np.max(y_all_sales) * 1.1
              
              profit_max = np.max(profit_plot_10k) * 1.1
              
              fig, axes = plt.subplots(1, 2, figsize=(21, 7), constrained_layout=True)
              
              # Ad Cost vs Sales plot (英語表記)
              axes[0].scatter(x_data_10k, y_data_10k, color="blue", label="Observed Data")
              fitted_label = f"Fitted {model_names_en.get(model_type, model_type)}"
              axes[0].plot(x_plot_10k, y_plot_10k, color="red", label=fitted_label)
              if optimal_x_10k is not None:
                  axes[0].axvline(optimal_x_10k, color="green", linestyle="--", label="Optimal Ad Cost")
              axes[0].set_title("Ad Cost vs Sales (10k JPY)")
              axes[0].set_xlabel("Ad Cost (×10k JPY)")
              axes[0].set_ylabel("Sales (×10k JPY)")
              axes[0].legend()
              axes[0].set_xlim(x_min, x_max)
              axes[0].set_ylim(0, y_max_sales)
              
              # Ad Cost vs Profit plot (英語表記)
              axes[1].plot(x_plot_10k, profit_plot_10k, color="red", label="Predicted Profit")
              if optimal_x_10k is not None:
                  axes[1].axvline(optimal_x_10k, color="green", linestyle="--", label="Optimal Ad Cost")
              axes[1].set_title("Ad Cost vs Profit (10k JPY)")
              axes[1].set_xlabel("Ad Cost (×10k JPY)")
              axes[1].set_ylabel("Profit (×10k JPY)")
              axes[1].legend()
              axes[1].set_xlim(x_min, x_max)
              axes[1].set_ylim(0, profit_max)
              
              st.pyplot(fig, use_container_width=True)
              
  except Exception as e:
      st.error(f"エラーが発生しました: {e}")

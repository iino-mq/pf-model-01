import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import pandas as pd

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

# -------------------------------
# Streamlit の基本設定・タイトル
# -------------------------------
st.set_page_config(page_title="広告費利益最大化予測", layout="wide")
st.title("広告費と売上から利益最大化予測")
st.markdown("""
このアプリケーションは、広告費と売上のデータを基に、利益が最大となる広告費の予測を行います。  
以下の7種類の選択肢からモデルを選べます：  
- 対数関数  
- 二次関数  
- シグモイド関数  
- ゴンペルツ関数  
- 分数関数  
- 散布図  
- 全モデル比較  

**入力形式:** コンマで区切られたテキストを入力してください。  
""")

# -------------------------------
# モデル選択の入力
# -------------------------------
model_type = st.selectbox("モデル選択", 
                          ["対数関数", "二次関数", "シグモイド関数", "ゴンペルツ関数", "分数関数", "散布図", "全モデル比較"])

# -------------------------------
# 入力データの入力 (デフォルト値あり)
# -------------------------------
default_x = "100000,200000,300000"
default_y = "400000,700000,900000"

x_input = st.text_area("広告費データ (円単位)", default_x, height=100)
y_input = st.text_area("売上データ (円単位)", default_y, height=100)

if st.button("解析開始"):
    try:
        # 入力文字列から float 配列へ変換
        x_data = np.array([float(val.strip()) for val in x_input.split(",") if val.strip()], dtype=float)
        y_data = np.array([float(val.strip()) for val in y_input.split(",") if val.strip()], dtype=float)
        
        if len(x_data) != len(y_data):
            st.error("エラー: 広告費と売上のデータ数が一致しません。")
        else:
            # 入力データを1つの表にまとめて表示
            df_input = pd.DataFrame({
                "広告費 (円)": x_data,
                "売上 (円)": y_data
            })
            st.subheader("入力データ")
            st.table(df_input)
            
            # 広告費最適化の探索範囲と初期値
            ad_bounds = [(1, 2_000_000)]
            initial_guess = 300000.0  # 広告費の初期値 (円)
            
            # -------------------------------
            # 「全モデル比較」の場合
            # -------------------------------
            if model_type == "全モデル比較":
                models = {
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
                
                results = []
                for name, setting in models.items():
                    model_func = setting["func"]
                    p0 = setting["p0"]
                    param_bounds = setting["bounds"]
                    
                    try:
                        # モデルパラメータ推定
                        params, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=param_bounds)
                        a, b, c = params
                        
                        # 決定係数 R² の計算
                        y_model = model_func(x_data, a, b, c)
                        residuals = y_data - y_model
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((y_data - np.mean(y_data))**2)
                        r2 = 1 - ss_res / ss_tot
                        
                        # 利益・ROAS の定義
                        profit_func = lambda x: model_func(x, a, b, c) - x
                        roas_func = lambda x: (model_func(x, a, b, c) / x) * 100
                        
                        # 利益最大化の最適化
                        res_opt = minimize(lambda x: -profit_func(x), x0=initial_guess, method='L-BFGS-B', bounds=ad_bounds)
                        if res_opt.success:
                            optimal_x = res_opt.x[0]
                            optimal_sales = model_func(optimal_x, a, b, c)
                            optimal_profit = profit_func(optimal_x)
                            optimal_roas = roas_func(optimal_x)
                        else:
                            optimal_x = np.nan
                            optimal_sales = np.nan
                            optimal_profit = np.nan
                            optimal_roas = np.nan
                        
                        # 円 -> 万円 への変換
                        optimal_x_10k = optimal_x / 10000.0
                        optimal_sales_10k = optimal_sales / 10000.0
                        optimal_profit_10k = optimal_profit / 10000.0
                        
                        results.append({
                            "モデル": name,
                            "最適広告費 (万円)": np.round(optimal_x_10k, 2),
                            "予測売上 (万円)": np.round(optimal_sales_10k, 2),
                            "予測利益 (万円)": np.round(optimal_profit_10k, 2),
                            "ROAS (%)": np.round(optimal_roas, 2),
                            "決定係数": np.round(r2, 3)
                        })
                    except Exception as e:
                        results.append({
                            "モデル": name,
                            "最適広告費 (万円)": "計算エラー",
                            "予測売上 (万円)": "計算エラー",
                            "予測利益 (万円)": "計算エラー",
                            "ROAS (%)": "計算エラー",
                            "決定係数": "計算エラー"
                        })
                
                df_results = pd.DataFrame(results)
                st.subheader("全モデル比較結果")
                st.table(df_results)
            
            # -------------------------------
            # 「散布図」の場合
            # -------------------------------
            elif model_type == "散布図":
                # 単に入力データの散布図（広告費 vs 売上）を表示する
                x_data_10k = x_data / 10000.0
                y_data_10k = y_data / 10000.0
                x_min = 0
                x_max = np.max(x_data_10k) * 1.05
                y_min = 0
                y_max = np.max(y_data_10k) * 1.1

                fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
                ax.scatter(x_data_10k, y_data_10k, color='blue', label='Observed Data')
                ax.set_title("散布図：広告費 vs 売上 (10k JPY)")
                ax.set_xlabel("広告費 (×10k JPY)")
                ax.set_ylabel("売上 (×10k JPY)")
                ax.legend()
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                st.pyplot(fig, use_container_width=True)
            
            # -------------------------------
            # 単一モデル選択の場合（対数関数、二次関数、シグモイド関数、ゴンペルツ関数、分数関数）
            # -------------------------------
            else:
                if model_type == "対数関数":
                    model_func   = log_func
                    model_name   = "対数関数モデル"
                    p0           = [500, 1e-4, 500]
                    param_bounds = ((1, 1e-7, 0), (1e6, 1, 1e7))
                elif model_type == "二次関数":
                    model_func   = quad_func
                    model_name   = "二次関数モデル"
                    p0           = [-1e-5, 3, 400000]
                    param_bounds = ((-1e-4, -100, 0), (0, 10, 1e7))
                elif model_type == "シグモイド関数":
                    model_func   = sigmoid_func
                    model_name   = "シグモイド関数モデル"
                    p0           = [1e6, 1e-5, 200000]
                    param_bounds = ((1e5, 1e-8, 0), (1e7, 1e-3, 1e6))
                elif model_type == "ゴンペルツ関数":
                    model_func   = gompertz_func
                    model_name   = "ゴンペルツ関数モデル"
                    p0           = [1e6, 5, 1e-5]
                    param_bounds = ((1e5, 0, 1e-8), (1e7, 10, 1e-3))
                elif model_type == "分数関数":
                    model_func   = frac_func
                    model_name   = "分数関数モデル"
                    p0           = [1e6, 0, 100000]
                    param_bounds = ((1e5, -1e6, 1), (1e7, 1e6, 1e7))
                else:
                    st.error("選択したモデルはサポートされていません。")
                    st.stop()
                
                # 利益・ROAS の計算用関数
                def profit_func(x, a, b, c):
                    return model_func(x, a, b, c) - x
                def roas_func(x, a, b, c):
                    return (model_func(x, a, b, c) / x) * 100
                
                # モデルパラメータ推定
                params, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=param_bounds)
                a, b, c = params
                perr = np.sqrt(np.diag(pcov))
                
                st.subheader("モデルパラメータ推定結果")
                st.write(f"選択モデル: {model_name}")
                st.write(f"a = {a:.3f} (std: {perr[0]:.3f})")
                st.write(f"b = {b:.6f} (std: {perr[1]:.6f})")
                st.write(f"c = {c:.3f} (std: {perr[2]:.3f})")
                
                # 決定係数 R² の計算
                y_model = model_func(x_data, a, b, c)
                residuals = y_data - y_model
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_data - np.mean(y_data))**2)
                r2 = 1 - ss_res / ss_tot
                st.write(f"決定係数 R² = {r2:.3f}")
                
                # 利益最大化の最適化
                res_opt = minimize(lambda x: -profit_func(x, a, b, c),
                                   x0=initial_guess, method='L-BFGS-B', bounds=ad_bounds)
                if res_opt.success:
                    optimal_x = res_opt.x[0]
                    optimal_sales = model_func(optimal_x, a, b, c)
                    optimal_profit = profit_func(optimal_x, a, b, c)
                    optimal_roas = roas_func(optimal_x, a, b, c)
                else:
                    st.error("利益最大化の計算が収束しませんでした。")
                    optimal_x = None
                
                if optimal_x is not None:
                    optimal_x_10k = optimal_x / 10000.0
                    optimal_sales_10k = optimal_sales / 10000.0
                    optimal_profit_10k = optimal_profit / 10000.0
                    st.subheader("利益最大化の結果")
                    st.write(f"最適な広告費: {optimal_x_10k:.2f} (万円)")
                    st.write(f"予測売上: {optimal_sales_10k:.2f} (万円)")
                    st.write(f"予測利益: {optimal_profit_10k:.2f} (万円)")
                    st.write(f"予測ROAS: {optimal_roas:.2f} %")
                
                # -------------------------------
                # グラフ描画（ROASグラフは非表示、残り2種類のグラフを自動調整して表示）
                # -------------------------------
                x_plot = np.linspace(1, 500000, 300)
                y_pred = model_func(x_plot, a, b, c)
                profit_pred = profit_func(x_plot, a, b, c)
                
                # 単位変換（円 → 万円）
                x_plot_10k = x_plot / 10000.0
                y_pred_10k = y_pred / 10000.0
                profit_pred_10k = profit_pred / 10000.0
                x_data_10k = x_data / 10000.0
                y_data_10k = y_data / 10000.0
                if optimal_x is not None:
                    optimal_x_10k = optimal_x / 10000.0
                else:
                    optimal_x_10k = None
                
                # 各軸の最小・最大値を自動調整
                x_all = np.concatenate([x_data_10k, x_plot_10k])
                if optimal_x_10k is not None:
                    x_all = np.append(x_all, optimal_x_10k)
                x_min = 0
                x_max = np.max(x_all) * 1.05
                
                y_all_sales = np.concatenate([y_data_10k, y_pred_10k])
                y_max_sales = np.max(y_all_sales) * 1.1
                
                profit_max = np.max(profit_pred_10k) * 1.1
                
                # 2つのサブプロットでグラフ作成
                fig, axes = plt.subplots(1, 2, figsize=(21, 7), constrained_layout=True)
                
                # ① 広告費 vs 売上
                axes[0].scatter(x_data_10k, y_data_10k, color='blue', label='Observed Data')
                axes[0].plot(x_plot_10k, y_pred_10k, color='red', label=f'Fitted {model_name}')
                if optimal_x_10k is not None:
                    axes[0].axvline(optimal_x_10k, color='green', linestyle='--', label='Optimal Ad Cost')
                axes[0].set_title('Ad Cost vs Sales (10k JPY)')
                axes[0].set_xlabel('Ad Cost (×10k JPY)')
                axes[0].set_ylabel('Sales (×10k JPY)')
                axes[0].legend()
                axes[0].set_xlim(x_min, x_max)
                axes[0].set_ylim(0, y_max_sales)
                
                # ② 広告費 vs 利益
                axes[1].plot(x_plot_10k, profit_pred_10k, color='red', label='Predicted Profit')
                if optimal_x_10k is not None:
                    axes[1].axvline(optimal_x_10k, color='green', linestyle='--', label='Optimal Ad Cost')
                axes[1].set_title('Ad Cost vs Profit (10k JPY)')
                axes[1].set_xlabel('Ad Cost (×10k JPY)')
                axes[1].set_ylabel('Profit (×10k JPY)')
                axes[1].legend()
                axes[1].set_xlim(x_min, x_max)
                axes[1].set_ylim(0, profit_max)
                
                st.pyplot(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

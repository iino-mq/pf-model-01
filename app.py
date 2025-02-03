import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

# -------------------------------
# モデル関数の定義
# -------------------------------

def log_func(x, a, b, c):
    """対数関数モデル: 売上 = a * log(b * x) + c"""
    return a * np.log(b * x) + c

def quad_func(x, a, b, c):
    """二次関数モデル: 売上 = a * x^2 + b * x + c  
    ※ 広告費増加により飽和（あるいは下落）することを期待する場合、aは負の値となる設定にしています。  
    """
    return a * x**2 + b * x + c

def sigmoid_func(x, a, b, c):
    """シグモイド関数モデル: 売上 = a / (1 + exp(-b*(x - c)))  
    ・a : 上限（最大売上）  
    ・b : 成長の速さ  
    ・c : 転換点（x軸上の中央値）  
    """
    return a / (1 + np.exp(-b * (x - c)))

def gompertz_func(x, a, b, c):
    """ゴンペルツ関数モデル: 売上 = a * exp(-b * exp(-c * x))  
    ・a : 上限（最大売上）  
    ・b, c : 形状を決めるパラメータ
    """
    return a * np.exp(-b * np.exp(-c * x))

# -------------------------------
# Streamlit の基本設定・タイトル
# -------------------------------
st.set_page_config(page_title="広告費利益最大化予測", layout="wide")
st.title("広告費と売上から利益最大化予測")
st.markdown("""
このアプリケーションは、広告費と売上のデータを基に、利益が最大となる広告費の予測を行います。  
以下の4種類のモデルから選択できます：  
- 対数関数  
- 二次関数  
- シグモイド関数  
- ゴンペルツ関数  
**入力形式:** コンマで区切られたテキストを入力してください。  
""")

# -------------------------------
# モデル選択の入力
# -------------------------------
model_type = st.selectbox("モデル選択", ["対数関数", "二次関数", "シグモイド関数", "ゴンペルツ関数"])

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
            st.subheader("入力データ")
            st.write("広告費 (円):", x_data)
            st.write("売上 (円):", y_data)
            
            # -------------------------------
            # モデルごとの設定
            # -------------------------------
            if model_type == "対数関数":
                model_func   = log_func
                model_name   = "対数関数モデル"
                p0           = [500, 1e-4, 500]  
                param_bounds = ((1, 1e-7, 0), (1e6, 1, 1e7))
            elif model_type == "二次関数":
                model_func   = quad_func
                model_name   = "二次関数モデル"
                # ※ 広告費が大きい場合を考慮し、aは小さく負の値、bは適度な大きさ、cはオフセットとして設定
                p0           = [-1e-5, 3, 400000]  
                # 二次関数が上に凸にならないよう、aは負の範囲に制限
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
            else:
                st.error("選択したモデルはサポートされていません。")
                st.stop()
            
            # -------------------------------
            # 利益・ROAS の計算（選択したモデルを利用）
            # -------------------------------
            def profit_func(x, a, b, c):
                """利益 = 売上 - 広告費"""
                return model_func(x, a, b, c) - x
            
            def roas_func(x, a, b, c):
                """ROAS (%) = (売上 / 広告費) * 100"""
                return (model_func(x, a, b, c) / x) * 100
            
            # -------------------------------
            # 1. モデルのパラメータ推定
            # -------------------------------
            params, pcov = curve_fit(model_func, x_data, y_data, p0=p0, bounds=param_bounds)
            a, b, c = params
            perr = np.sqrt(np.diag(pcov))
            
            st.subheader("モデルパラメータ推定結果")
            st.write(f"選択モデル: {model_name}")
            st.write(f"a = {a:.3f} (std: {perr[0]:.3f})")
            st.write(f"b = {b:.6f} (std: {perr[1]:.6f})")
            st.write(f"c = {c:.3f} (std: {perr[2]:.3f})")
            
            # -------------------------------
            # 決定係数 R² の計算
            # -------------------------------
            y_model = model_func(x_data, a, b, c)
            residuals = y_data - y_model
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - ss_res / ss_tot
            st.write(f"決定係数 R² = {r2:.3f}")
            
            # -------------------------------
            # 2. 利益最大化の最適化
            # -------------------------------
            initial_guess = 300000.0  # 広告費の初期値 (円)
            ad_bounds = [(1, 2_000_000)]  # 探索範囲 (円)
            res = minimize(lambda x: -profit_func(x, a, b, c),
                           x0=initial_guess, method='L-BFGS-B', bounds=ad_bounds)
            
            if res.success:
                optimal_x = res.x[0]
                optimal_sales = model_func(optimal_x, a, b, c)
                optimal_profit = profit_func(optimal_x, a, b, c)
                optimal_roas = roas_func(optimal_x, a, b, c)
                
                # 円 -> 万円 単位に変換
                optimal_x_10k = optimal_x / 10000.0
                optimal_sales_10k = optimal_sales / 10000.0
                optimal_profit_10k = optimal_profit / 10000.0
                
                st.subheader("利益最大化の結果")
                st.write(f"最適な広告費: {optimal_x_10k:.2f} (万円)")
                st.write(f"予測売上: {optimal_sales_10k:.2f} (万円)")
                st.write(f"予測利益: {optimal_profit_10k:.2f} (万円)")
                st.write(f"予測ROAS: {optimal_roas:.2f} %")
            else:
                st.error("利益最大化の計算が収束しませんでした。")
                optimal_x = None
            
            # -------------------------------
            # 3. グラフ描画
            # -------------------------------
            x_plot = np.linspace(1, 500000, 300)
            y_pred = model_func(x_plot, a, b, c)
            profit_pred = profit_func(x_plot, a, b, c)
            roas_pred = roas_func(x_plot, a, b, c)
            
            # 円 -> 万円 への変換（表示用）
            x_plot_10k = x_plot / 10000.0
            y_pred_10k = y_pred / 10000.0
            profit_pred_10k = profit_pred / 10000.0
            x_data_10k = x_data / 10000.0
            y_data_10k = y_data / 10000.0
            optimal_x_10k = None if optimal_x is None else (optimal_x / 10000.0)
            
            # matplotlib による描画
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            
            # ① 広告費 vs 売上
            axes[0].scatter(x_data_10k, y_data_10k, color='blue', label='Observed Data')
            axes[0].plot(x_plot_10k, y_pred_10k, color='red', label=f'Fitted {model_name}')
            if optimal_x_10k is not None:
                axes[0].axvline(optimal_x_10k, color='green', linestyle='--', label='Optimal Ad Cost')
            axes[0].set_title('Ad Cost vs Sales (10k JPY)')
            axes[0].set_xlabel('Ad Cost (×10k JPY)')
            axes[0].set_ylabel('Sales (×10k JPY)')
            axes[0].legend()
            axes[0].set_xlim(0, max(x_plot_10k)*1.05)
            axes[0].set_ylim(0, max(y_pred_10k)*1.1)
            
            # ② 広告費 vs 利益
            axes[1].plot(x_plot_10k, profit_pred_10k, color='red', label='Predicted Profit')
            if optimal_x_10k is not None:
                axes[1].axvline(optimal_x_10k, color='green', linestyle='--', label='Optimal Ad Cost')
            axes[1].set_title('Ad Cost vs Profit (10k JPY)')
            axes[1].set_xlabel('Ad Cost (×10k JPY)')
            axes[1].set_ylabel('Profit (×10k JPY)')
            axes[1].legend()
            axes[1].set_xlim(0, max(x_plot_10k)*1.05)
            axes[1].set_ylim(0, max(profit_pred_10k)*1.1)
            
            # ③ 広告費 vs ROAS
            axes[2].plot(x_plot_10k, roas_pred, color='red', label='Predicted ROAS')
            if optimal_x_10k is not None:
                axes[2].axvline(optimal_x_10k, color='green', linestyle='--', label='Optimal Ad Cost')
            axes[2].set_title('Ad Cost vs ROAS')
            axes[2].set_xlabel('Ad Cost (×10k JPY)')
            axes[2].set_ylabel('ROAS (%)')
            axes[2].legend()
            axes[2].set_xlim(0, max(x_plot_10k)*1.05)
            axes[2].set_ylim(0, max(roas_pred)*1.1)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

# -------------------------------
# 関数定義
# -------------------------------
def log_func(x, a, b, c):
    """対数モデル: y = a * log(b * x) + c"""
    return a * np.log(b * x) + c

def quadratic_func(x, a, b, c):
    """二次関数モデル: y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c

def sigmoid_func(x, a, b, c):
    """
    シグモイド関数モデル (ロジスティック関数):
    y = a / (1 + exp(-b*(x - c)))
      ・a : 上限値（最大売上）
      ・b : カーブの急峻さ
      ・c : 中央値（xの中央値付近）
    """
    return a / (1 + np.exp(-b * (x - c)))

def gompertz_func(x, a, b, c):
    """
    ゴンペルツ関数モデル:
    y = a * exp(-b * exp(-c * x))
      ・a : 上限値（最大売上）
      ・b, c : 形状パラメータ
    """
    return a * np.exp(-b * np.exp(-c * x))

def profit_func(x, model_func, params):
    """利益 = 予測売上 - 広告費"""
    return model_func(x, *params) - x

def roas_func(x, model_func, params):
    """ROAS (%) = (予測売上 / 広告費) * 100"""
    return (model_func(x, *params) / x) * 100

# -------------------------------
# Streamlit の基本設定・タイトル
# -------------------------------
st.set_page_config(page_title="広告費利益最大化予測", layout="wide")
st.title("広告費と売上から利益最大化予測")
st.markdown("""
このアプリケーションは、広告費と売上のデータを基に各種モデルによるフィッティングを行い、  
どのモデルがデータに適合するかを視覚的に確認できます。  
**入力形式:** コンマで区切られたテキストを入力してください。  
""")

# -------------------------------
# 入力データの入力 (デフォルト値あり)
# -------------------------------
default_x = "100000,200000,300000"
default_y = "400000,700000,900000"

x_input = st.text_area("広告費データ (円単位)", default_x, height=100)
y_input = st.text_area("売上データ (円単位)", default_y, height=100)

if st.button("解析開始"):
    try:
        # 入力文字列を float 配列へ変換
        x_data = np.array([float(val.strip()) for val in x_input.split(",") if val.strip()], dtype=float)
        y_data = np.array([float(val.strip()) for val in y_input.split(",") if val.strip()], dtype=float)
        
        if len(x_data) != len(y_data):
            st.error("エラー: 広告費と売上のデータ数が一致しません。")
        else:
            st.subheader("入力データ")
            st.write("広告費 (円):", x_data)
            st.write("売上 (円):", y_data)
            
            # -------------------------------
            # 1. 各モデルによるフィッティング
            # -------------------------------
            results = {}  # 各モデルのフィッティング結果を格納する辞書
            
            # (a) 対数モデル
            try:
                p0_log = [500, 1e-4, 500]  # 初期値
                bounds_log = ((1, 1e-7, 0), (1e6, 1, 1e7))
                params_log, pcov_log = curve_fit(log_func, x_data, y_data, p0=p0_log, bounds=bounds_log)
                results["対数モデル"] = {"func": log_func, "params": params_log}
                st.write("【対数モデル】")
                st.write(f"パラメータ: a = {params_log[0]:.3f}, b = {params_log[1]:.6f}, c = {params_log[2]:.3f}")
            except Exception as e:
                st.error("対数モデルのフィッティングに失敗しました: " + str(e))
            
            # (b) 二次関数モデル
            try:
                # 広告費・売上の値のオーダーを考慮して初期値を設定
                p0_quad = [0, 3, 100000]  # 例: y = 0*x^2 + 3*x + 100000
                params_quad, pcov_quad = curve_fit(quadratic_func, x_data, y_data, p0=p0_quad)
                results["二次関数"] = {"func": quadratic_func, "params": params_quad}
                st.write("【二次関数モデル】")
                st.write(f"パラメータ: a = {params_quad[0]:.6e}, b = {params_quad[1]:.3f}, c = {params_quad[2]:.3f}")
            except Exception as e:
                st.error("二次関数のフィッティングに失敗しました: " + str(e))
            
            # (c) シグモイド関数モデル
            try:
                # 初期値: aは最大売上の1.2倍程度, bは急峻さ, cは中央値付近
                p0_sigmoid = [max(y_data)*1.2, 1e-4, np.median(x_data)]
                # パラメータに制約: a>0, b>0, cはx_data内
                lower_sigmoid = [0, 0, min(x_data)]
                upper_sigmoid = [1e7, 1e-2, max(x_data)]
                params_sigmoid, pcov_sigmoid = curve_fit(sigmoid_func, x_data, y_data, p0=p0_sigmoid, bounds=(lower_sigmoid, upper_sigmoid))
                results["シグモイド関数"] = {"func": sigmoid_func, "params": params_sigmoid}
                st.write("【シグモイド関数モデル】")
                st.write(f"パラメータ: a = {params_sigmoid[0]:.3f}, b = {params_sigmoid[1]:.6f}, c = {params_sigmoid[2]:.3f}")
            except Exception as e:
                st.error("シグモイド関数のフィッティングに失敗しました: " + str(e))
            
            # (d) ゴンペルツ関数モデル
            try:
                # 初期値: aは最大売上程度, b,cは小さい正の値
                p0_gompertz = [max(y_data), 1, 1e-5]
                lower_gompertz = [0, 0, 0]
                upper_gompertz = [1e7, 1e3, 1e-2]
                params_gompertz, pcov_gompertz = curve_fit(gompertz_func, x_data, y_data, p0=p0_gompertz, bounds=(lower_gompertz, upper_gompertz))
                results["ゴンペルツ関数"] = {"func": gompertz_func, "params": params_gompertz}
                st.write("【ゴンペルツ関数モデル】")
                st.write(f"パラメータ: a = {params_gompertz[0]:.3f}, b = {params_gompertz[1]:.3f}, c = {params_gompertz[2]:.6e}")
            except Exception as e:
                st.error("ゴンペルツ関数のフィッティングに失敗しました: " + str(e))
            
            # -------------------------------
            # 2. ４つのモデルのフィッティング結果を比較するグラフの描画
            # -------------------------------
            # プロット用の x 軸の範囲 (例: 入力データの最小値～最大値の1.1倍)
            x_min = min(x_data)
            x_max = max(x_data) * 1.1
            x_plot = np.linspace(x_min, x_max, 300)
            
            # 円単位を「万円」に変換するための係数
            conv = 10000.0  
            x_plot_10k = x_plot / conv
            x_data_10k = x_data / conv
            y_data_10k = y_data / conv
            
            # サブプロット: 2行×2列
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (model_name, res) in enumerate(results.items()):
                func = res["func"]
                params = res["params"]
                y_fit = func(x_plot, *params)
                y_fit_10k = y_fit / conv
                
                axes[i].scatter(x_data_10k, y_data_10k, color='blue', label='観測データ')
                axes[i].plot(x_plot_10k, y_fit_10k, color='red', label='フィッティング曲線')
                axes[i].set_title(f"{model_name}\nパラメータ: {np.around(params, decimals=3)}", fontsize=12)
                axes[i].set_xlabel("広告費 (万円)", fontsize=10)
                axes[i].set_ylabel("売上 (万円)", fontsize=10)
                axes[i].legend(fontsize=10)
                axes[i].grid(True)
            
            plt.tight_layout()
            st.subheader("各モデルのフィッティング結果")
            st.pyplot(fig)
            
            # -------------------------------
            # （※必要に応じて）対数モデルによる利益最大化の最適化結果も表示可能
            # -------------------------------
            # ここでは例として対数モデルを用いた最適広告費（利益最大化）の最適化も実施
            initial_guess = 300000.0  # 広告費の初期値 (円)
            ad_bounds = [(1, 2_000_000)]  # 広告費の探索範囲 (円)
            res_opt = minimize(lambda x: -profit_func(x, log_func, results["対数モデル"]["params"]),
                               x0=initial_guess, method='L-BFGS-B', bounds=ad_bounds)
            
            if res_opt.success:
                optimal_x = res_opt.x[0]                # 最適広告費 (円)
                optimal_sales = log_func(optimal_x, *results["対数モデル"]["params"])  # 予測売上 (円)
                optimal_profit = profit_func(optimal_x, log_func, results["対数モデル"]["params"])  # 予測利益 (円)
                optimal_roas = roas_func(optimal_x, log_func, results["対数モデル"]["params"])      # 予測ROAS (%)
                
                optimal_x_10k = optimal_x / conv
                st.subheader("対数モデルによる利益最大化の結果")
                st.write(f"最適な広告費: {optimal_x_10k:.2f} (万円)")
                st.write(f"予測売上: {optimal_sales/conv:.2f} (万円)")
                st.write(f"予測利益: {optimal_profit/conv:.2f} (万円)")
                st.write(f"予測ROAS: {optimal_roas:.2f} %")
            else:
                st.error("対数モデルによる利益最大化の計算が収束しませんでした。")
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

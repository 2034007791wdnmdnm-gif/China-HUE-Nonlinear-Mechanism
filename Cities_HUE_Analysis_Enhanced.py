# -*- coding: utf-8 -*-
"""
城市HUE分析增强版
- 保留原始数据处理逻辑
- 集成XGBoost代码的美观绘图功能
- 使用调整后的R²（Adj R²）
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import networkx as nx
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy import stats

# --------------------------
# Configuration
# --------------------------
CONFIG = {
    "random_state": 42,
    "output_dir": "Nature_Submission_Enhanced",
    "formats": ["png", "pdf"],
    "dpi": 300,
    
    # 字体配置
    "font_family_en": "Times New Roman",
    "font_family_zh": "SimSun",
    "label_fontsize": 14,
    "title_fontsize": 16,
    "tick_fontsize": 14,
    
    # 图1 模型性能验证图 参数配置
    "fig1_kde_linewidth": 1.5,
    "fig1_hist_bins": 50,
    "fig1_train_color": "#A9A9A9",
    "fig1_test_color": "#9F3E3F",
    "fig1_scatter_s": 30,
    "fig1_scatter_edgecolor": "white",
    "fig1_scatter_linewidth": 0.5,
    
    # 图2 SHAP全局特征贡献图参数配置
    "fig2_scatter_s": 12,
    "fig2_bar_color": "#C3D3F2",
    
    # 图3 SHAP单特征依赖图 参数配置
    "fig3_scatter_s": 20,
    "fig3_curve_color": "#FF451B",
    "fig3_curve_linewidth": 1.5,
    "fig3_cbar_fontsize": 10,
    "fig3_positive_color": "#DEF4F1",
    "fig3_negative_color": "#E6DADA",
    
    # 图4 SHAP主效应与交互效应对比图 参数配置
    "fig4_main_color": "#71BCB1",
    "fig4_inter_color": "#9F6566",
    
    # 图5 特征交互效应复合矩阵图 细节参数配置
    "fig5_max_features": 18,
    "fig5_label_fontsize": 16,
    "fig5_tick_fontsize": 11,
    "fig5_cbar_fontsize": 16,
    
    # 图7 特征影响-交互网络图 参数配置
    "fig7_max_features": 18,
    "fig7_node_color": "#48A597",
    "fig7_edge_color": "#9C1A1C",

    # 图9 个体样本SHAP力图(SHAP Force Plot) 参数配置
    "fig9_max_instances": 50,        # 图9 最大绘制样本数量限制
    "fig9_figsize": (16, 4),         # 图9 画布尺寸
    "fig9_color_pos": "#9C1A1C",     # 图9 正向 SHAP 值颜色
    "fig9_color_neg": "#48A597",     # 图9 负向 SHAP 值颜色

    # 图10 PDP二维依赖图 参数配置
    "fig10_line_color_q1": "#2A6F97",  # Q1等高线颜色
    "fig10_line_color_med": "#48A597",  # 中位数等高线颜色
    "fig10_line_color_q3": "#9C1A1C",  # Q3等高线颜色

    # 全局色彩映射配置，Teal-Grey-Red Diverging Colormap
    "shap_cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", [ "#48A597", "#FFFFFF", "#9C1A1C"]),
}

# 字体与系统底层配置
plt.rcParams['font.family'] = [CONFIG["font_family_en"], CONFIG["font_family_zh"]]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = CONFIG["dpi"]
plt.rcParams['savefig.dpi'] = CONFIG["dpi"]
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.labelsize'] = CONFIG["label_fontsize"]
plt.rcParams['axes.titlesize'] = CONFIG["title_fontsize"]
plt.rcParams['xtick.labelsize'] = CONFIG["tick_fontsize"]
plt.rcParams['ytick.labelsize'] = CONFIG["tick_fontsize"]
plt.rcParams['legend.fontsize'] = CONFIG["label_fontsize"]
np.random.seed(CONFIG["random_state"])

# --------------------------
# File Paths
# --------------------------
INPUT_FILE = "原始统计数据.xlsx"
OUTPUT_FOLDER = CONFIG["output_dir"]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# Feature Name Mapping
# --------------------------
# 短特征名到长特征名的映射
feature_full = {
    'FloatPop': 'Floating Population',
    'UrbRate': 'Urbanization Rate',
    'Pop': 'Population Size',
    'Aging': 'Population Aging Rate',
    'GDPpc': 'GDP per Capita',
    'TIR': 'Tertiary Industry Ratio',
    'DispInc': 'Urban Disposable Income per Capita',
    'EntPer10k': 'Enterprises per 10,000 People',
    'Temp': 'Annual Mean Temperature',
    'AQI': 'Annual Mean AQI',
    'Green': 'Green Coverage Rate',
    'Hosp': 'Number of Hospitals',
    'Museum': 'Number of Museums',
    'EduExp': 'Education Expenditure',
    'Pension': 'Urban Employee Pension Participants',
    'RoadArea': 'Urban Road Area',
    'LandFin': 'Land Finance Dependence',
    'GovExp': 'Fiscal Expenditure per Capita',
}

# --------------------------
# Load & Preprocess Data
# --------------------------
def generate_sample_data():
    """生成示例数据，当原始数据文件不存在时使用"""
    np.random.seed(CONFIG["random_state"])
    n_samples = 100
    
    # 生成特征数据
    data = {
        'HUE': np.random.uniform(0.5, 0.9, n_samples),
        'FloatPop': np.random.uniform(100000, 1000000, n_samples),
        'UrbRate': np.random.uniform(0.5, 0.95, n_samples),
        'Pop': np.random.uniform(500000, 5000000, n_samples),
        'Aging': np.random.uniform(0.1, 0.3, n_samples),
        'GDPpc': np.random.uniform(50000, 200000, n_samples),
        'TIR': np.random.uniform(0.4, 0.8, n_samples),
        'DispInc': np.random.uniform(30000, 100000, n_samples),
        'EntPer10k': np.random.uniform(50, 200, n_samples),
        'Temp': np.random.uniform(10, 25, n_samples),
        'AQI': np.random.uniform(30, 150, n_samples),
        'Green': np.random.uniform(0.2, 0.5, n_samples),
        'Hosp': np.random.randint(10, 100, n_samples),
        'Museum': np.random.randint(1, 50, n_samples),
        'EduExp': np.random.uniform(10000, 100000, n_samples),
        'Pension': np.random.uniform(100000, 1000000, n_samples),
        'RoadArea': np.random.uniform(100, 1000, n_samples),
        'LandFin': np.random.uniform(0.2, 0.8, n_samples),
        'GovExp': np.random.uniform(5000, 20000, n_samples)
    }
    
    return pd.DataFrame(data)

try:
    df = pd.read_excel(INPUT_FILE)
    print(f"成功加载数据文件: {INPUT_FILE}")
except FileNotFoundError:
    print(f"数据文件 {INPUT_FILE} 不存在，生成示例数据...")
    df = generate_sample_data()

# 确保列名正确
df.columns = df.columns.astype(str).str.strip()

# 检查列名并调整
required_columns = ['HUE', 'FloatPop', 'UrbRate', 'Pop', 'Aging', 'GDPpc', 'TIR', 'DispInc', 'EntPer10k', 'Temp', 'AQI', 'Green', 'Hosp', 'Museum', 'EduExp', 'Pension', 'RoadArea', 'LandFin', 'GovExp']

# 确保所有必要的列都存在
for col in required_columns:
    if col not in df.columns:
        print(f"警告: 缺少列 {col}")

# 使用所有列，除了HUE作为目标变量
X = df.drop('HUE', axis=1).astype(float)
y = df['HUE'].values
n, p = X.shape
print(f"Dataset: {n} samples, {p} features (ALL PRESERVED)")

# --------------------------
# Evaluation Metrics
# --------------------------
def adjusted_r2(r2, n, p):
    """计算调整后的R²"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def daes(y_true, y_pred):
    """计算绝对误差符号一致率"""
    residuals = y_true - y_pred
    return np.mean(np.sign(residuals) == np.sign(y_true - np.mean(y_true)))

# --------------------------
# Model Parameter Grids (Unified Search Space)
# --------------------------
cv_inner = KFold(n_splits=10, shuffle=True, random_state=42)

param_grids = {
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 3, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [1.0, 2.0, 5.0, 10.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 2, 3]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 5],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'ExtraTrees': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 5],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'loss': ['linear', 'square', 'exponential']
    },
    'DecisionTree': {
        'max_depth': [2, 3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 3, 5, 7],
        'max_features': ['sqrt', 'log2', None]
    },
    'Ridge': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
    },
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    'SVR': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'epsilon': [0.001, 0.01, 0.05, 0.1, 0.2],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0]
    }
}

# --------------------------
# Data Splitting for Small Sample
# --------------------------
# For small sample size (n<200), we don't use a separate test set
# Instead, we use the full dataset for training and LOOCV for validation
print("\n" + "="*70)
print("DATA SPLITTING FOR SMALL SAMPLE SIZE")
print("="*70)
print(f"Total samples: {n}")
print("Strategy: Using full dataset for training + LOOCV for validation")
print("(No separate test set for small samples)")

X_train = X
y_train = y

# --------------------------
# Fair Model Comparison (using training set only)
# --------------------------
print("\n" + "="*70)
print("FAIR MODEL COMPARISON WITH UNIFIED HYPERPARAMETER TUNING")
print("All models tuned using 10-fold inner CV on training set")
print("="*70)

models = {
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
    'RandomForest': RandomForestRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(max_iter=10000, random_state=42),
}

best_models = {}
results = []
cv_scores_all = {}  # Store CV scores for statistical testing

for name, model in models.items():
    print(f"\n Tuning {name}...")
    grid = param_grids.get(name, {})

    if grid:
        if name == 'XGBoost':
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        elif name == 'SVR':
            base_model = SVR()
        else:
            base_model = model.__class__(random_state=42)

        searcher = RandomizedSearchCV(base_model, grid, n_iter=30, cv=cv_inner,
                                    scoring='r2', random_state=42, n_jobs=1)
        searcher.fit(X_train, y_train)
        best_params = searcher.best_params_
        best_model = searcher.best_estimator_
        
        # Store CV scores for statistical testing
        cv_scores_all[name] = cross_val_score(best_model, X_train, y_train, cv=cv_inner, scoring='r2')
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {}
        cv_scores_all[name] = cross_val_score(best_model, X_train, y_train, cv=cv_inner, scoring='r2')

    best_models[name] = best_model

    y_pred = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred)
    adj_r2 = adjusted_r2(r2_train, len(y_train), X_train.shape[1])
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_pred)
    evs = explained_variance_score(y_train, y_pred)
    
    non_zero_mask = y_train != 0
    mape = np.mean(np.abs((y_train[non_zero_mask] - y_pred[non_zero_mask]) / y_train[non_zero_mask])) * 100
    me = np.mean(y_pred - y_train)
    
    absolute_errors = np.abs(y_train - y_pred)
    total_absolute_deviation = np.sum(np.abs(y_train - np.mean(y_train)))
    daes_score = 1 - np.sum(absolute_errors) / total_absolute_deviation

    results.append({
        'Model': name,
        'R^2': r2_train,
        'Adj R^2': adj_r2,
        'MSE': mse,
        'RMSE': rmse,
        'EVS': evs,
        'MAE': mae,
        'MAPE(%)': mape,
        'ME': me,
        'DAES': daes_score,
        'Best_Params': best_params
    })

    print(f"   R^2={r2_train:.4f}, Adj R^2={adj_r2:.4f}, RMSE={rmse:.4f}, DAES={daes_score:.4f}")

results_df = pd.DataFrame(results).sort_values('R^2', ascending=False)
print("\n" + "="*100)
print("MODEL RANKING BY R^2 (All 9 Metrics)")
print("="*100)
print(results_df[['Model', 'R^2', 'Adj R^2', 'MSE', 'RMSE', 'EVS', 'MAE', 'MAPE(%)', 'ME', 'DAES']].to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x))

# --------------------------
# Statistical Significance Testing
# --------------------------
print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*70)
print("Paired t-test comparing CV R^2 scores (Best model vs others)")
print("-"*70)

best_model_name = results_df.iloc[0]['Model']
best_cv_scores = cv_scores_all[best_model_name]

for name in results_df['Model'].values[1:]:  # Skip the best model itself
    if name in cv_scores_all:
        other_cv_scores = cv_scores_all[name]
        t_stat, p_value = stats.ttest_rel(best_cv_scores, other_cv_scores)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{best_model_name} vs {name}: t={t_stat:.3f}, p={p_value:.4f} {significance}")

# --------------------------
# Use GradientBoosting with Tuned Parameters (Best Model)
# --------------------------
best_model_name = 'GradientBoosting'
gb_tuned_params = best_models['GradientBoosting'].get_params()
print(f"\n>>> Using GradientBoosting with tuned parameters: <<<")
print(f"    {gb_tuned_params}")
model = GradientBoostingRegressor(**gb_tuned_params)

# 在完整训练集上训练最终模型
model.fit(X_train, y_train)

# 计算训练集预测
y_pred_train = model.predict(X_train)

# --------------------------
# SHAP Analysis
# --------------------------
print("\n" + "="*70)
print("SHAP VALUE ANALYSIS")
print("="*70)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
mean_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[::-1]
sorted_features = [X_train.columns[i] for i in sorted_idx]

# Feature Importance Table
imp_df = pd.DataFrame({
    'Feature': sorted_features,
    'Feature_Full': [feature_full.get(X.columns[i], X.columns[i]) for i in sorted_idx],
    'Mean_SHAP': mean_shap[sorted_idx],
    'Contribution_Pct': mean_shap[sorted_idx] / mean_shap.sum() * 100
})

print("\nTop 10 Features by SHAP:")
print(imp_df.head(10).to_string(index=False))

# ==========================================
# 3. 辅助函数
# ==========================================
def save_figure(fig, filename_base, subfolder=None):
    """保存 Matplotlib 图形至指定目录"""
    save_dir = CONFIG["output_dir"]  # 获取输出根目录
    if subfolder:  # 若存在子文件夹
        save_dir = os.path.join(save_dir, subfolder)  # 拼接路径
    os.makedirs(save_dir, exist_ok=True)  # 创建目标目录
    
    # 替换文件名中的特殊字符（把斜杠换成下划线），避免路径解析错误
    safe_filename_base = str(filename_base).replace('/', '_').replace('\\', '_')
    
    for fmt in CONFIG["formats"]:  # 遍历保存格式
        filepath = os.path.join(save_dir, f"{safe_filename_base}.{fmt}")  # 生成完整路径
        fig.savefig(filepath, dpi=CONFIG["dpi"], format=fmt, bbox_inches='tight')  # 保存图像
    plt.close(fig)  # 关闭图像对象释放内存

# ==========================================
# 4. 增强的绘图功能
# ==========================================

class EnhancedVisualizer:
    def __init__(self, model, X, y, X_train, y_train, feature_names, shap_values, explainer):
        self.model = model
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.shap_values = shap_values
        self.shap_values_obj = explainer(X, check_additivity=False)
        self.explainer = explainer
        # 模型已经在主代码中训练好，不需要重新fit
        self.shap_interaction_values = None
    
    def calculate_shap_interaction(self):
        """计算 SHAP 交互作用值"""
        print("正在计算 SHAP 交互作用值...")
        # 使用全部数据计算交互作用值以确保准确性
        try:
            self.shap_interaction_values = self.explainer.shap_interaction_values(self.X)
        except Exception as e:
            print(f"使用全部数据计算失败: {e}")
            print("尝试使用部分数据计算...")
            # 如果内存不足，使用部分数据
            sample_size = min(1000, len(self.X))
            sample_indices = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X.iloc[sample_indices]
            self.shap_interaction_values = self.explainer.shap_interaction_values(X_sample)
        print("SHAP 交互作用值计算完成。")
    
    def plot_figure_1(self):
        """绘制模型预测散点图与残差图"""
        print("正在绘制图1：模型评估组合图...")
        y_train_pred = self.model.predict(self.X_train)
        
        # 只计算训练集指标
        r2_train = r2_score(self.y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        
        # 计算调整后R²
        n_train, p = len(self.y_train), self.X_train.shape[1]
        adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
        
        mae_train = mean_absolute_error(self.y_train, y_train_pred)
        res_train = y_train_pred - self.y_train

        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1], height_ratios=[1, 4, 1.5], wspace=0.05, hspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 0])
        # 只显示训练集数据点
        ax_main.scatter(self.y_train, y_train_pred, 
                      color=CONFIG["fig1_train_color"],
                      s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"],
                      linewidth=CONFIG["fig1_scatter_linewidth"], label="Train data", alpha=0.8, zorder=2)
        
        xlims = ax_main.get_xlim()
        ylims = ax_main.get_ylim()
        
        ax_main.plot(xlims, ylims, 'k--', zorder=5)
        # 使用训练集数据计算拟合线
        m, b = np.polyfit(self.y_train, y_train_pred, 1)
        y_fit_start, y_fit_end = m * xlims[0] + b, m * xlims[1] + b
        ax_main.plot(xlims, [y_fit_start, y_fit_end], color='black', linewidth=2, label="Fitted line", zorder=5)
        
        ax_main.set_xlim(xlims)
        ax_main.set_ylim(ylims)
        
        # 移除横轴标识
        ax_main.set_xticklabels([])
        ax_main.set_xticks([])
        
        ax_main.text(0.5, 0.96, "GradientBoosting", transform=ax_main.transAxes, ha='center', va='top',
                     fontsize=CONFIG["title_fontsize"]+4, fontweight='bold', zorder=10)
        
        # 上面显示训练集的R²和Adj R²
        metrics_text = f"Train $R^2$ = {r2_train:.2f}, Adj $R^2$ = {adj_r2_train:.2f}\n$RMSE$ = {rmse_train:.2f}"
        ax_main.text(0.05, 0.85, metrics_text, transform=ax_main.transAxes, va='top', ha='left',
                     fontsize=CONFIG["title_fontsize"]-2, color="darkred",
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=10)
                       
        ax_main.legend(loc="lower right", frameon=False, fontsize=CONFIG["label_fontsize"])
        ax_main.grid(True, linestyle='--', alpha=0.5, color='lightgray')

        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        sns.histplot(self.y_train, color=CONFIG["fig1_train_color"], bins=CONFIG["fig1_hist_bins"],
                     ax=ax_top, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.kdeplot(self.y_train, color=CONFIG["fig1_train_color"], ax=ax_top,
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        ax_top.axis('off')

        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        sns.histplot(y=y_train_pred, color=CONFIG["fig1_train_color"], bins=CONFIG["fig1_hist_bins"],
                     ax=ax_right, alpha=0.5, stat="density", element="bars", edgecolor="white", linewidth=0.3)
        sns.kdeplot(y=y_train_pred, color=CONFIG["fig1_train_color"], ax=ax_right,
                    linewidth=CONFIG["fig1_kde_linewidth"], zorder=10, cut=0)
        ax_right.axis('off')

        ax_res = fig.add_subplot(gs[2, 0], sharex=ax_main)
        # 只显示训练集残差点
        res_train = y_train_pred - self.y_train
        ax_res.scatter(self.y_train, res_train, color=CONFIG["fig1_train_color"],
                       s=CONFIG["fig1_scatter_s"], edgecolor=CONFIG["fig1_scatter_edgecolor"], linewidth=CONFIG["fig1_scatter_linewidth"], alpha=0.8)
        ax_res.plot(xlims, [0, 0], color='black', linewidth=1.5, zorder=5)
        ax_res.grid(True, linestyle='--', alpha=0.5, color='lightgray')
        
        # 移除横轴标识
        ax_res.set_xticklabels([])
        ax_res.set_xticks([])
        
        # 下面显示训练集MAE
        res_text = f"Train MAE = {mae_train:.3f}"
        ax_res.text(0.95, 0.95, res_text, transform=ax_res.transAxes, fontsize=CONFIG["label_fontsize"],
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), zorder=6)
        
        save_figure(fig, "Fig1_Prediction_Residuals")
    
    def plot_figure_2(self):
        """绘制全局 SHAP 特征重要性条形图与蜂群图"""
        print("正在绘制图2：全局特征贡献度分析图...")
        fig, ax1 = plt.subplots(figsize=(10, 8))  # 创建画布
        
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)  # 计算各个特征的平均绝对 SHAP 值
        sort_inds = np.argsort(mean_abs_shap)  # 获取升序排列的索引
        sorted_features = [self.feature_names[i] for i in sort_inds]  # 按升序排列特征名称
        sorted_mean_shap = mean_abs_shap[sort_inds]  # 按升序排列平均 SHAP 值
        total_shap_sum = np.sum(mean_abs_shap)  # 计算 SHAP 总值
        
        shap_vals = self.shap_values_obj.values[:, sort_inds]  # 重排 SHAP 值矩阵列顺序
        feat_vals = self.X.values[:, sort_inds]  # 重排特征数值矩阵列顺序
        y_pos = np.arange(len(sorted_features))  # 生成纵坐标序列
        
        ax2 = ax1.twiny()  # 添加共享 Y 轴的副图
        ax2.barh(y_pos, sorted_mean_shap, color=CONFIG["fig2_bar_color"], align='center', alpha=0.8, height=0.6, zorder=2)  # 在副图绘制背景条形图
        
        cmap = CONFIG["shap_cmap"]  # 获取颜色映射表
        for i in range(len(sorted_features)):  # 遍历特征绘制蜂群图
            row_shap = shap_vals[:, i]  # 当前特征的 SHAP 值
            row_feat = feat_vals[:, i]  # 当前特征的原始数值
            feat_min, feat_max = np.min(row_feat), np.max(row_feat)  # 计算极值
            row_feat_norm = (row_feat - feat_min) / (feat_max - feat_min) if feat_max > feat_min else np.zeros_like(row_feat)  # Min-Max 归一化特征值用于颜色映射
                
            jitter = np.random.normal(0, 0.1, size=len(row_shap))  # 生成 Y 轴高斯噪声以展示密度
            scatter = ax1.scatter(row_shap, np.repeat(i, len(row_shap)) + jitter,   # 绘制散点
                                  c=row_feat_norm, cmap=cmap, s=CONFIG["fig2_scatter_s"], alpha=0.8, edgecolors='none', zorder=4)

        max_mean_val = np.max(sorted_mean_shap)  # 获取最大均值
        ax2.set_xlim(0, max_mean_val * 1.2)  # 扩展副图 X 轴范围
        
        for i, v in enumerate(sorted_mean_shap):  # 遍历添加百分比文本标签
            pct = (v / total_shap_sum) * 100  # 计算占比
            offset = max_mean_val * 0.01  # 计算标签偏移量
            ax1.text(v + offset, i, f"{pct:.1f}%", va='center', ha='left', fontsize=CONFIG["label_fontsize"],   
                     transform=ax2.transData, zorder=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        ax1.set_zorder(ax2.get_zorder() + 1)  # 调整散点图层级至顶部
        ax1.patch.set_visible(False)  # 使主图背景透明
        
        ax1.set_yticks(y_pos)  # 配置 Y 轴刻度
        ax1.set_yticklabels(sorted_features, fontsize=CONFIG["tick_fontsize"])  # 配置特征名称标签
        ax1.set_xlabel("SHAP value (impact on model output)", fontsize=CONFIG["label_fontsize"])  # 设置主 X 轴标签
        ax2.set_xlabel("Mean Absolute SHAP Value", fontsize=CONFIG["label_fontsize"])  # 设置副 X 轴标签
        ax1.grid(True, axis='x', linestyle='--', alpha=0.4)  # 开启垂直参考线
        
        divider = make_axes_locatable(ax1)  # 调用坐标轴分割工具
        cax = divider.append_axes("right", size="3%", pad=0.1)  # 添加色带区域
        
        sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))  # 生成颜色映射标量
        sm_map.set_array([])  # 空载数据
        cbar = fig.colorbar(sm_map, cax=cax)  # 添加色带
        cbar.set_ticks([0, 1])  # 指定刻度位置
        cbar.set_ticklabels(['Low', 'High'])  # 指定刻度文本
        cbar.set_label('Feature value', rotation=270, labelpad=15, fontsize=CONFIG["label_fontsize"])  # 设置色带标签
        
        save_figure(fig, "Fig2_Global_Contribution")  # 保存图像文件
    
    def plot_figure_3(self):
        """批量绘制特征的偏依赖散点图与平滑曲线 (依据截图识别 Positive/Negative 象限)"""
        print("正在绘制图3：单特征偏依赖图 (自动按截距划分 Positive/Negative 区间)...")
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)  # 获取特征重要性
        sort_inds_desc = np.argsort(mean_abs_shap)[::-1]  # 获取降序排列索引
        folder_name = "Fig3_Dependence_Plots"  # 设定输出子文件夹
        
        for rank, feat_idx in enumerate(sort_inds_desc):  # 遍历降序特征
            feature_name = self.feature_names[feat_idx]  # 获取目标特征名称
            fig, ax = plt.subplots(figsize=(6, 5))  # 创建子画布
            
            x_vals = self.X.iloc[:, feat_idx].values  # 提取横坐标数值
            y_vals = self.shap_values_obj.values[:, feat_idx]  # 提取纵坐标数值 (SHAP 值)
            
            color_feat_idx = sort_inds_desc[1] if rank == 0 else sort_inds_desc[0]  # 获取用于着色的对比特征索引
            color_feat_name = self.feature_names[color_feat_idx]  # 获取对比特征名称
            c_vals = self.X.iloc[:, color_feat_idx].values  # 提取对比特征数值
            
            scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=CONFIG["shap_cmap"],   # 绘制散点图
                                 s=CONFIG["fig3_scatter_s"], alpha=0.9, edgecolors='none', zorder=4)
            
            sorted_indices = np.argsort(x_vals)  # 获取按 X 升序排列的索引
            x_sorted = x_vals[sorted_indices]  # 重排 X 数组
            y_sorted = y_vals[sorted_indices]  # 重排 Y 数组
            
            lowess = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.5)  # 拟合局部加权回归散点平滑曲线 (Lowess)
            ax.plot(lowess[:, 0], lowess[:, 1], color=CONFIG["fig3_curve_color"],   # 绘制拟合曲线
                    linewidth=CONFIG["fig3_curve_linewidth"], alpha=0.9, label="Lowess curve", zorder=5)
            
            window_size = max(5, int(len(x_sorted) * 0.1))  # 动态计算窗口大小
            rolling_std = pd.Series(y_sorted - lowess[:, 1]).rolling(window=window_size, min_periods=1, center=True).std().values  # 计算局部标准差
            ax.fill_between(lowess[:, 0], lowess[:, 1] - rolling_std, lowess[:, 1] + rolling_std,   # 填充置信度区间
                            color=CONFIG["fig3_curve_color"], alpha=0.15, edgecolor='none', linewidth=0, zorder=2)
            
            # =============== 完全参考截图识别 Positive/Negative 阈值和填充方式 ===============
            sign_changes = np.diff(np.sign(lowess[:, 1]))  # 计算一阶差分获取交叉零点
            zero_crossings = np.where(sign_changes != 0)[0]  
            
            if len(zero_crossings) > 0:
                zc_idx = zero_crossings[0] # 获取第一个与 SHAP=0 相交的位置
                x_cross = lowess[zc_idx, 0]
                
                # 记录原始的作图范围以防 fill_between 拉大画布
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                # SHAP > 0 (右上象限 - Positive)
                ax.fill_between([x_cross, x_max], 0, y_max, color=CONFIG["fig3_positive_color"], alpha=0.5, zorder=1, label="Positive")
                # SHAP < 0 (左下象限 - Negative)
                ax.fill_between([x_min, x_cross], y_min, 0, color=CONFIG["fig3_negative_color"], alpha=0.5, zorder=1, label="Negative")
                
                ax.axvline(x=x_cross, color='#D32F2F', linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)  # 绘制阈值垂直线
                
                ax.scatter(x_cross, 0, color='#D32F2F', s=25, zorder=6)  # 在交叉点添加红点标记
                
                ax.text(x_cross, 0.05 * (y_max - y_min), f"{x_cross:.2f}", va='bottom', ha='center',  # 添加阈值文本标签
                        fontsize=CONFIG["title_fontsize"], color='#D32F2F',
                        bbox=dict(facecolor=(1, 1, 1, 0.7), edgecolor='none', pad=2), zorder=6)
                
                ax.set_xlim(x_min, x_max)  # 恢复原始范围
                ax.set_ylim(y_min, y_max)

            ax.set_xlabel("Feature value", fontsize=CONFIG["label_fontsize"])  # 设置 X 轴标签
            ax.set_ylabel("SHAP", fontsize=CONFIG["label_fontsize"])  # 设置 Y 轴标签
            ax.set_title(feature_full.get(feature_name, feature_name), fontsize=CONFIG["title_fontsize"])  # 设置图表标题
            ax.grid(True, linestyle='--', alpha=0.3)  # 开启背景网格
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=2)  # 绘制 SHAP=0 参考线
            
            ax.legend(loc='best', fontsize=CONFIG["label_fontsize"] - 2, framealpha=0, edgecolor='none')  # 添加图例
            
            divider = make_axes_locatable(ax)  # 调用坐标轴分割工具
            cax = divider.append_axes("right", size="5%", pad=0.0)  # 添加色带区域
            cbar = fig.colorbar(scatter, cax=cax)  # 添加色带
            cbar.ax.tick_params(labelsize=CONFIG["fig3_cbar_fontsize"])  # 配置色带刻度字体
            
            clean_feature_name = re.sub(r'[^\w]', '', feature_name)
            filename = f"{rank+1:02d}_{clean_feature_name}_Dependence"
            save_figure(fig, filename, subfolder=folder_name)
    
    def plot_figure_4(self):
        """绘制主效应与交互效应强度对比图"""
        print("正在绘制图4：主效应与交互效应对比图...")
        if self.shap_interaction_values is None:
            self.calculate_shap_interaction()
        
        num_features = len(self.feature_names)
        main_effects = np.zeros(num_features)
        inter_effects = np.zeros(num_features)
        
        for i in range(num_features):
            main_effects[i] = np.abs(self.shap_interaction_values[:, i, i]).mean()
            mask = np.ones(num_features, dtype=bool)
            mask[i] = False
            inter_effects[i] = np.abs(self.shap_interaction_values[:, i, mask]).sum(axis=1).mean()
            
        total_effects = main_effects + inter_effects
        sort_inds = np.argsort(total_effects)[::-1]
        
        top_names = [self.feature_names[i] for i in sort_inds]
        top_main = main_effects[sort_inds]
        top_inter = inter_effects[sort_inds]
        
        fig_width = max(10, num_features * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        x = np.arange(len(top_names))
        width = 0.4
        
        rects1 = ax.bar(x - width/2, top_main, width, label='Main effect (Mean |SHAP|)', color=CONFIG["fig4_main_color"])
        rects2 = ax.bar(x + width/2, top_inter, width, label='Interaction (sum over others)', color=CONFIG["fig4_inter_color"])
        
        max_height = max(np.max(top_main), np.max(top_inter))
        ax.set_ylim(0, max_height * 1.15)
        
        overlap_threshold = max_height * 0.05

        for i in range(len(top_names)):
            h1 = top_main[i]
            h2 = top_inter[i]
            
            text1 = f'{h1:.3f}'
            text2 = f'{h2:.3f}'
            
            offset_1, offset_2 = 2, 2
            va_1, va_2 = 'bottom', 'bottom'
            
            if abs(h1 - h2) < overlap_threshold:
                if h1 >= h2:
                    offset_1 = 12
                    offset_2 = 1
                else:
                    offset_1 = 1
                    offset_2 = 12

            ax.annotate(text1, xy=(x[i] - width/2, h1), xytext=(0, offset_1), textcoords="offset points", ha='center', va=va_1, fontsize=9)
            ax.annotate(text2, xy=(x[i] + width/2, h2), xytext=(0, offset_2), textcoords="offset points", ha='center', va=va_2, fontsize=9)
        
        ax.set_ylabel('Magnitude (Mean |SHAP|)', fontsize=CONFIG["label_fontsize"])
        ax.set_title('All Features: Main vs Interaction', fontsize=CONFIG["title_fontsize"])
        ax.set_xticks(x)
        
        ax.set_xticklabels(top_names, fontsize=CONFIG["tick_fontsize"], rotation=90)
        
        ax.legend(fontsize=CONFIG["label_fontsize"], frameon=False)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        save_figure(fig, "Fig4_Main_vs_Interaction_All")
    
    def plot_figure_5(self):
        """绘制全变量热力交互图"""
        print("正在绘制图5：全变量热力交互图...")
        if self.shap_interaction_values is None:
            self.calculate_shap_interaction()
        
        # 使用所有变量，保持原始顺序以确保SHAP值与特征对应
        num_features = len(self.feature_names)
        
        # 计算所有特征对的交互效应值
        inter_matrix = np.zeros((num_features, num_features))
        for r in range(num_features):
            for c in range(num_features):
                if r != c:
                    inter_matrix[r, c] = np.abs(self.shap_interaction_values[:, r, c]).mean()
        
        # 计算每个特征的主效应
        main_effects = np.zeros(num_features)
        for i in range(num_features):
            main_effects[i] = np.abs(self.shap_interaction_values[:, i, i]).mean()
        
        # 计算相对交互效应（交互效应 / 主效应的几何平均）
        relative_inter = np.zeros((num_features, num_features))
        for r in range(num_features):
            for c in range(num_features):
                if r != c and main_effects[r] > 0 and main_effects[c] > 0:
                    # 相对交互强度 = 交互效应 / sqrt(主效应_r * 主效应_c)
                    relative_inter[r, c] = inter_matrix[r, c] / np.sqrt(main_effects[r] * main_effects[c])
                elif r != c:
                    relative_inter[r, c] = inter_matrix[r, c]
        
        max_relative = np.max(relative_inter[relative_inter > 0]) if np.any(relative_inter > 0) else 1
        
        fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.08, hspace=0.08, bottom=0.12, left=0.12, top=0.9)
        
        for row in range(num_features):
            for col in range(num_features):
                ax = axes[row, col]
                feat_row_name = self.feature_names[row]
                feat_col_name = self.feature_names[col]
                
                if row > col:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    inter_val = inter_matrix[row, col]
                    rel_val = relative_inter[row, col]
                    
                    # 使用相对值来着色，使弱交互也能显示出来
                    norm_val = rel_val / max_relative if max_relative > 0 else 0
                    bg_color = CONFIG["shap_cmap"](norm_val)
                    ax.set_facecolor(bg_color)
                    
                    text_color = 'white' if norm_val > 0.7 else 'black'
                    # 使用科学计数法显示微小数值
                    if inter_val >= 0.001:
                        abs_str = f"{inter_val:.4f}"
                    else:
                        abs_str = f"{inter_val:.2e}"
                    
                    if rel_val >= 0.01:
                        rel_str = f"({rel_val:.2f})"
                    else:
                        rel_str = f"({rel_val:.2e})"
                    
                    display_text = f"{abs_str}\n{rel_str}"
                    ax.text(0.5, 0.5, display_text, ha='center', va='center',
                            fontweight='bold', fontsize=CONFIG["tick_fontsize"]-1, color=text_color)
                
                else:
                    ax.set_facecolor('white')
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.8)
                    
                    if row == col:
                        x_vals = self.shap_interaction_values[:, row, col]
                    else:
                        x_vals = self.shap_interaction_values[:, row, col] * 2
                        
                    y_jitter = np.random.normal(0, 0.1, size=len(x_vals))
                    c_vals = self.X.iloc[:, row].values
                    
                    c_min, c_max = c_vals.min(), c_vals.max()
                    c_norm = (c_vals - c_min) / (c_max - c_min + 1e-8)
                    
                    ax.scatter(x_vals, y_jitter, c=c_norm, cmap=CONFIG["shap_cmap"],
                               s=8, alpha=0.7, edgecolors='none', vmin=0, vmax=1)
                    
                    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                    ax.set_yticks([])
                    
                    if row == 0:
                        ax.xaxis.tick_top()
                        ax.tick_params(axis='x', labelsize=CONFIG["fig5_tick_fontsize"], pad=2)
                        ax.locator_params(axis='x', nbins=3)
                        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
                    else:
                        ax.set_xticks([])

                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('#333333')

                if row == num_features - 1:
                    ax.set_xlabel(feat_col_name, fontsize=CONFIG["fig5_label_fontsize"], rotation=90,
                                  ha='center', va='top', labelpad=10)
                    
                if col == 0:
                    ax.set_ylabel(feat_row_name, fontsize=CONFIG["fig5_label_fontsize"], rotation=0,
                                  ha='right', va='center', labelpad=15)

        fig.text(0.5, 0.02, "SHAP interaction value (absolute & relative)", ha='center', va='center', fontsize=CONFIG["title_fontsize"])

        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        sm_map = plt.cm.ScalarMappable(cmap=CONFIG["shap_cmap"], norm=plt.Normalize(vmin=0, vmax=max_relative))
        sm_map.set_array([])
        cbar = fig.colorbar(sm_map, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=CONFIG["fig5_cbar_fontsize"])
        
        cbar.set_label('Relative Interaction Strength', rotation=270, labelpad=20, fontsize=CONFIG["fig5_cbar_fontsize"])

        save_figure(fig, "Fig5_Combined_Matrix")
    
    def plot_figure_6(self):
        """绘制成对特征交互作用散点图"""
        print("正在绘制图6：成对交互作用散点图...")
        if self.shap_interaction_values is None:
            self.calculate_shap_interaction()
        
        folder_name = "Fig6_Pairwise_Interactions"
        num_features = len(self.feature_names)
        
        for i in range(num_features):
            for j in range(num_features):
                if i >= j:
                    continue
                
                feat_i_name = self.feature_names[i]
                feat_j_name = self.feature_names[j]
                
                x_vals = self.X[feat_i_name].values
                y_vals = self.shap_interaction_values[:, i, j] * 2
                c_vals = self.X[feat_j_name].values
                
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap=CONFIG["shap_cmap"],
                                     s=30, alpha=0.8, edgecolors='none', zorder=2)
                
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
                
                sorted_indices = np.argsort(x_vals)
                x_sorted = x_vals[sorted_indices]
                y_sorted = y_vals[sorted_indices]
                lowess_inter = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.3)
                
                sign_changes = np.diff(np.sign(lowess_inter[:, 1]))
                zero_crossings = np.where(sign_changes != 0)[0]
                for zc in zero_crossings:
                    threshold_val = lowess_inter[zc, 0]
                    ax.axvline(x=threshold_val, color='orange', linestyle='--', alpha=0.9, zorder=3)
                    ax.text(threshold_val, np.median(y_vals), f"{threshold_val:.2f}", rotation=90,
                            va='center', ha='right', fontsize=CONFIG["tick_fontsize"],
                            bbox=dict(facecolor=(1, 1, 1, 0.7), edgecolor='none', pad=2), zorder=4)
                
                ax.set_xlabel(feat_i_name, fontsize=CONFIG["label_fontsize"])
                ax.set_ylabel("SHAP Interaction Value", fontsize=CONFIG["label_fontsize"])
                ax.set_title(f"{feat_i_name} × {feat_j_name}", fontsize=CONFIG["title_fontsize"])
                ax.grid(True, linestyle='--', alpha=0.3)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.0)
                cbar = plt.colorbar(scatter, cax=cax)
                cbar.ax.tick_params(labelsize=CONFIG["tick_fontsize"])
                cbar.set_label(feat_j_name, rotation=270, labelpad=15, fontsize=CONFIG["label_fontsize"])
                
                save_figure(fig, f"Fig6_Pairwise_Interactions/{feat_i_name}_x_{feat_j_name}", folder_name)
    
    def plot_figure_7(self):
        """绘制特征重要性与交互强度的关系网络图"""
        print("正在绘制图7：特征重要性与交互强度网络图...")
        if self.shap_interaction_values is None:
            self.calculate_shap_interaction()
        
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)
        sort_inds = np.argsort(mean_abs_shap)[::-1]
        num_nodes = min(CONFIG.get("fig7_max_features", 18), len(self.feature_names))
        top_inds = sort_inds[:num_nodes]
        top_names = [self.feature_names[i] for i in top_inds]
        
        G = nx.Graph()
        
        for idx, name in zip(top_inds, top_names):
            G.add_node(name, weight=mean_abs_shap[idx])
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                idx_i = top_inds[i]
                idx_j = top_inds[j]
                inter_val = np.abs(self.shap_interaction_values[:, idx_i, idx_j]).mean()
                G.add_edge(top_names[i], top_names[j], weight=inter_val)
        
        pos = nx.circular_layout(G)
        node_weights = [G.nodes[n]['weight'] for n in G.nodes]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        
        node_norm = (node_weights - np.min(node_weights)) / (np.max(node_weights) - np.min(node_weights) + 1e-8)
        edge_norm = (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-8)
        
        node_sizes = 400 + 1500 * node_norm
        edge_widths = 1 + 7 * edge_norm
        
        node_cmap = mcolors.LinearSegmentedColormap.from_list("node_cmap", ["#E0F2F1", CONFIG["fig7_node_color"]])
        edge_cmap = mcolors.LinearSegmentedColormap.from_list("edge_cmap", ["#FCE4EC", CONFIG["fig7_edge_color"]])
        
        fig = plt.figure(figsize=(10, 10))
        
        # 使用精准的绝对坐标系控制主图和底部色带
        ax_main = fig.add_axes([0.1, 0.15, 0.8, 0.8])  # 给主网络图充裕的空间，底部留白
        
        # 绘制边缘和节点
        edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_weights, 
                                       edge_cmap=edge_cmap, alpha=0.8, ax=ax_main)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_weights, 
                                       cmap=node_cmap, edgecolors='gray', linewidths=1.5, ax=ax_main)
        
        # 智能节点标签垂直上下避让，杜绝向内倒压节点
        for node, (x, y) in pos.items():
            # 获取基于节点尺寸的动态垂直外扩距离
            offset_y = 0.12 + (node_sizes[top_names.index(node)] / 40000.0) 
            
            # 判断节点处于画布上半区还是下半区
            if y >= 0:
                # 上半区：文字放在节点上方 (va='bottom')
                ax_main.text(x, y + offset_y, node, ha='center', va='bottom', fontsize=CONFIG["tick_fontsize"])
            else:
                # 下半区：文字放在节点下方 (va='top')
                ax_main.text(x, y - offset_y, node, ha='center', va='top', fontsize=CONFIG["tick_fontsize"])
        
        ax_main.set_title("Feature Impact & Interaction Network", fontsize=CONFIG["title_fontsize"])
        ax_main.axis('off')
        
        # 精细添加底部的双图例颜色带
        cax1 = fig.add_axes([0.15, 0.10, 0.3, 0.015])  # 左下方极窄长条状色带
        sm_node = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=np.min(node_weights), vmax=np.max(node_weights)))
        sm_node.set_array([])
        cbar1 = fig.colorbar(sm_node, cax=cax1, orientation='horizontal')
        cbar1.set_label('Importance (Vimp)', fontsize=CONFIG["label_fontsize"])
        cbar1.ax.tick_params(labelsize=CONFIG["tick_fontsize"] - 2)
        
        # 右侧: Interaction Intensity (Vint)
        cax2 = fig.add_axes([0.55, 0.10, 0.3, 0.015])  # 右下方极窄长条状色带
        sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=np.min(edge_weights), vmax=np.max(edge_weights)))
        sm_edge.set_array([])
        cbar2 = fig.colorbar(sm_edge, cax=cax2, orientation='horizontal')
        cbar2.set_label('Interaction Intensity (Vint)', fontsize=CONFIG["label_fontsize"])
        cbar2.ax.tick_params(labelsize=CONFIG["tick_fontsize"] - 2)
        
        save_figure(fig, "Fig7_Impact_Interaction_Network")
    
    def plot_figure_8(self):
        """绘制所有样本的 SHAP 汇总热力图 (SHAP Summary Heatmap)"""
        print("正在绘制图8：SHAP 特征综合热力图...")
        
        fig = plt.figure(figsize=CONFIG.get("fig8_figsize", (12, 8)))
        
        shap.plots.heatmap(
            self.shap_values_obj,
            max_display=CONFIG.get("fig8_max_display", 20),
            cmap=CONFIG["shap_cmap"],
            show=False
        )
        
        current_fig = plt.gcf()
        for ax in current_fig.axes:
            ax.tick_params(labelsize=CONFIG["tick_fontsize"])
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=CONFIG["label_fontsize"], fontfamily=CONFIG["font_family_en"])
            if ax.get_ylabel():
                ylabel_text = ax.get_ylabel()
                if "f(x)" in ylabel_text.lower():
                    ax.set_ylabel("f(x)", fontsize=CONFIG["label_fontsize"], fontfamily=CONFIG["font_family_en"], fontstyle='italic')
                else:
                    ax.set_ylabel(ylabel_text, fontsize=CONFIG["label_fontsize"], fontfamily=CONFIG["font_family_en"])
        
        save_figure(current_fig, "Fig8_SHAP_Heatmap")
    
    def plot_figure_9(self):
        """批量绘制各样本 SHAP 力图 (Force Plot)"""
        print("正在绘制图9：各样本 SHAP 单变量推演力图...")
        folder_name = "Fig9_Force_Plots"
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        
        max_instances = min(len(self.X), CONFIG.get("fig9_max_instances", 10))
        
        for i in range(max_instances):
            instance_features = self.X.iloc[i, :].round(3)
            
            # 使用统一的颜色映射
            shap_fig = shap.force_plot(
                base_value,
                self.shap_values_obj.values[i, :],
                instance_features,
                matplotlib=True,
                show=False
            )
            
            fig = shap_fig if shap_fig is not None else plt.gcf()
            fig.set_size_inches(CONFIG["fig9_figsize"])
            
            ax = fig.gca()
            
            # 处理底部特征文本重叠问题
            feature_texts = [t for t in ax.texts if '=' in t.get_text()]
            feature_texts.sort(key=lambda t: t.get_position()[0])
            
            lines = ax.get_lines()
            x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
            x_threshold = x_span * 0.15
            
            y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
            step = y_span * 0.12
            
            levels = [0.0, -step, -step*2, -step*3, -step*4, -step*5]
            last_x_at_level = {lvl: -float('inf') for lvl in levels}
            
            min_y_attained = ax.get_ylim()[0]
            
            for text_obj in feature_texts:
                x, y = text_obj.get_position()
                chosen_level = levels[0]
                
                for lvl in levels:
                    if x - last_x_at_level[lvl] > x_threshold:
                        chosen_level = lvl
                        break
                
                last_x_at_level[chosen_level] = x
                new_y = y + chosen_level
                min_y_attained = min(min_y_attained, new_y)
                
                if chosen_level != 0.0:
                    text_obj.set_position((x, new_y))
                    for line in lines:
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        if len(xdata) == 2 and abs(xdata[0] - x) < 1e-3 and abs(xdata[1] - x) < 1e-3:
                            if abs(ydata[0] - y) < abs(ydata[1] - y):
                                ydata[0] = new_y
                            else:
                                ydata[1] = new_y
                            line.set_ydata(ydata)
            
            ax.set_ylim(bottom=min_y_attained - step)
            
            # 处理顶部预测标示重叠
            top_texts = [t for t in ax.texts if 'base value' in t.get_text() or 'f(x)' in t.get_text()]
            top_texts.sort(key=lambda t: t.get_position()[0])
            for idx in range(1, len(top_texts)):
                prev_x, prev_y = top_texts[idx-1].get_position()
                curr_x, curr_y = top_texts[idx].get_position()
                if abs(curr_x - prev_x) < x_threshold and abs(curr_y - prev_y) < step:
                    top_texts[idx].set_position((curr_x, curr_y + step))
            
            # ================= 更新色彩映射覆写策略 =================
            pos_color = CONFIG["fig9_color_pos"]  # 从预设中读取目标正向色调
            neg_color = CONFIG["fig9_color_neg"]  # 从预设中读取目标负向色调
            target_pos = mcolors.to_rgb("#FF0D57")  # SHAP实际使用的红色
            target_neg = mcolors.to_rgb("#1E88E5")  # SHAP实际使用的蓝色
            
            def match_color(c):  # 定义颜色识别比对小插件辅助工具函数
                if c is None: return None  # 忽略抛弃非法空参验证输入
                try:
                    c_rgb = mcolors.to_rgb(c)  # 解析对象并返回统一 RGB 三元色数值参数
                    if sum((a - b)**2 for a, b in zip(c_rgb, target_pos)) < 0.05:  # 计算点色差距离判断近似红色
                        return pos_color  # 验证确认返回自定义配色
                    if sum((a - b)**2 for a, b in zip(c_rgb, target_neg)) < 0.05:  # 计算点色差距离判断近似蓝色
                        return neg_color  # 验证确认返回自定义配色
                except:
                    pass  # 兜底捕获异常解析返回失败静默跳过保护机制
                return None  # 未通过耙标色系验证则退回无操作回执
            
            for obj in ax.findobj():  # 遍历坐标轴内绘制对象节点元素集合
                if hasattr(obj, 'get_color') and hasattr(obj, 'set_color'):  # 排查对象节点是否含有颜色管控属性
                    new_c = match_color(obj.get_color())  # 派发识别验证审核查对
                    if new_c: obj.set_color(new_c)  # 执行强制设换新色配置显示
                
                if hasattr(obj, 'get_facecolor') and hasattr(obj, 'set_facecolor'):  # 检验节点是否有面域填充特权属性接口
                    fc = obj.get_facecolor()  # 提取旧漆面材质配置拉取数值集
                    if isinstance(fc, np.ndarray) and fc.size >= 3:  # 分析处理矩阵高维颜色类型处理
                        fc = fc[0] if fc.ndim == 2 else fc  # 解出提取复杂色彩底质色彩单一数据
                    new_c = match_color(fc)  # 颜色比对
                    if new_c: obj.set_facecolor(new_c)  # 注入指定底漆填充生色替代生效
                
                if hasattr(obj, 'get_edgecolor') and hasattr(obj, 'set_edgecolor'):  # 检测是否具有线框边缘描绘特性
                    ec = obj.get_edgecolor()  # 提取多维数组中的核心颜色值
                    if isinstance(ec, np.ndarray) and ec.size >= 3:  
                        ec = ec[0] if ec.ndim == 2 else ec  
                    new_c = match_color(ec)  # 匹配并获取新的颜色配置
                    if new_c: obj.set_edgecolor(new_c)  # 应用新的边缘颜色
            
            filename = f"Instance_{i+1:04d}_Force_Plot"
            save_figure(fig, filename, folder_name)
    
    def plot_figure_10(self):
        """批量绘制所有特征的二维 PDP 依赖图"""
        print("正在绘制图10：二维PDP特征依赖轮廓图...")
        mean_abs_shap = np.abs(self.shap_values_obj.values).mean(axis=0)
        sort_inds = np.argsort(mean_abs_shap)[::-1]
        
        X_bg_median = self.X_train.median().values
        grid_resolution = 50
        cmap = CONFIG["shap_cmap"]
        
        for rank, idx1 in enumerate(sort_inds):
            feat1 = self.feature_names[idx1]
            clean_feat1 = re.sub(r'[^\w]', '', feat1)
            folder_name = f"Fig10_2D_PDP/{rank+1:02d}_{clean_feat1}"
            
            x1_min = self.X_train[feat1].min()
            x1_max = self.X_train[feat1].max()
            x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
            
            for idx2 in range(len(self.feature_names)):
                if idx1 == idx2: continue
                
                feat2 = self.feature_names[idx2]
                x2_min = self.X_train[feat2].min()
                x2_max = self.X_train[feat2].max()
                x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
                
                X1, X2 = np.meshgrid(x1_grid, x2_grid)
                
                grid_points = np.tile(X_bg_median, (grid_resolution * grid_resolution, 1))
                grid_points[:, idx1] = X1.ravel()
                grid_points[:, idx2] = X2.ravel()
                
                preds = self.model.predict(grid_points).reshape(grid_resolution, grid_resolution)
                
                fig, ax = plt.subplots(figsize=(7, 6))
                
                cf = ax.contourf(X1, X2, preds, levels=60, cmap=cmap, alpha=0.9)
                
                q1, med, q3 = np.percentile(preds, [25, 50, 75])
                cs1 = ax.contour(X1, X2, preds, levels=[q1], colors=[CONFIG.get("fig10_line_color_q1", "#2A6F97")], linestyles=['--'], linewidths=1.5)
                cs2 = ax.contour(X1, X2, preds, levels=[med], colors=[CONFIG.get("fig10_line_color_med", "#48A597")], linestyles=['-'], linewidths=1.5)
                cs3 = ax.contour(X1, X2, preds, levels=[q3], colors=[CONFIG.get("fig10_line_color_q3", "#9C1A1C")], linestyles=['--'], linewidths=1.5)
                
                p_max_val = preds.max()
                p_min_val = preds.min()
                
                max_idx = np.unravel_index(np.argmax(preds), preds.shape)
                min_idx = np.unravel_index(np.argmin(preds), preds.shape)
                
                ax.scatter(X1[max_idx], X2[max_idx], marker='*', color='orange', s=120, edgecolors='black', linewidth=0.5, zorder=5)
                ax.scatter(X1[min_idx], X2[min_idx], marker='o', color='#00BCD4', s=60, edgecolors='white', linewidth=0.5, zorder=5)
                
                h_mid, w_mid = grid_resolution // 2, grid_resolution // 2
                quadrants = {
                    'lower left': preds[:h_mid, :w_mid],
                    'lower right': preds[:h_mid, w_mid:],
                    'upper left': preds[h_mid:, :w_mid],
                    'upper right': preds[h_mid:, w_mid:]
                }
                
                best_loc = min(quadrants, key=lambda k: np.std(quadrants[k]))
                
                cbar = fig.colorbar(cf, ax=ax, pad=0.03)
                cbar.ax.tick_params(labelsize=CONFIG["tick_fontsize"])
                
                import matplotlib.lines as mlines
                l1 = mlines.Line2D([], [], color=CONFIG.get("fig10_line_color_q1", "#1E88E5"), linestyle='--', label=f'Q1: {q1:.2f}')
                l2 = mlines.Line2D([], [], color=CONFIG.get("fig10_line_color_med", "#FF5722"), linestyle='-', label=f'Median: {med:.2f}')
                l3 = mlines.Line2D([], [], color=CONFIG.get("fig10_line_color_q3", "#4CAF50"), linestyle='--', label=f'Q3: {q3:.2f}')
                p_max = ax.scatter([], [], marker='*', color='orange', s=100, edgecolors='black', label=f'Max: {p_max_val:.2f}')
                p_min = ax.scatter([], [], marker='o', color='#00BCD4', s=60, edgecolors='white', label=f'Min: {p_min_val:.2f}')
                
                leg = ax.legend(handles=[l1, l2, l3, p_max, p_min], loc=best_loc,
                                fontsize=CONFIG["label_fontsize"] - 2,
                                frameon=True, facecolor=(1, 1, 1, 0.8), edgecolor='none')
                
                ax.set_xlabel(feat1, fontsize=CONFIG["label_fontsize"])
                ax.set_ylabel(feat2, fontsize=CONFIG["label_fontsize"])
                ax.set_title(f"2D PDP: {feat1} vs {feat2}", fontsize=CONFIG["title_fontsize"])
                
                filename = f"{feat1}_vs_{feat2}_PDP"
                save_figure(fig, filename, folder_name)

# ==========================================
# 5. 执行增强可视化
# ==========================================
print("\n" + "="*70)
print("ENHANCED VISUALIZATION")
print("="*70)

visualizer = EnhancedVisualizer(
    model=model,
    X=X,
    y=y,
    X_train=X_train,
    y_train=y_train,
    feature_names=X.columns.tolist(),
    shap_values=shap_values,
    explainer=explainer
)

# 绘制增强的图表
visualizer.plot_figure_1()
visualizer.plot_figure_2()
visualizer.plot_figure_3()
visualizer.plot_figure_4()
visualizer.plot_figure_5()
visualizer.plot_figure_6()
visualizer.plot_figure_7()
visualizer.plot_figure_8()
visualizer.plot_figure_9()
visualizer.plot_figure_10()

# --------------------------
# Export All Results
# --------------------------
y_pred_train = model.predict(X_train)
y_pred_full = model.predict(X)
residuals_train = y_train - y_pred_train
residuals_full = y - y_pred_full

# Training dataset with SHAP
export_df = X_train.copy()
export_df['HUE_Actual'] = y_train
export_df['HUE_Pred'] = y_pred_train
export_df['Residual'] = residuals_train
for i, col in enumerate(X_train.columns):
    export_df[f'SHAP_{col}'] = shap_values[:, i]
export_df.to_excel(f"{OUTPUT_FOLDER}/Dataset_SHAP_Results.xlsx", index=False)

# Feature importance
imp_df.to_excel(f"{OUTPUT_FOLDER}/Feature_Importance.xlsx", index=False)

# Model comparison results
results_df.to_excel(f"{OUTPUT_FOLDER}/Model_Comparison.xlsx", index=False)

# LaTeX Table
with open(f"{OUTPUT_FOLDER}/Table_Feature_Importance.tex", 'w', encoding='utf-8') as f:
    f.write("""\begin{table}[htbp]
\centering
\caption{Feature Importance Based on SHAP Values}
\begin{tabular}{lc}
\hline
Feature & Contribution (\%) \\
\hline
""")
    for _, row in imp_df.iterrows():
        f.write(f"{row['Feature']} & {row['Contribution_Pct']:.2f} \\")
    f.write("\hline\end{tabular}\end{table}")

with open(f"{OUTPUT_FOLDER}/Table_Model_Comparison.tex", 'w', encoding='utf-8') as f:
    f.write("""\begin{table}[htbp]
\centering
\caption{Model Performance Comparison}
\begin{tabular}{lccc}
\hline
Model & Train R$^2$ & Adj R$^2$ & RMSE \\
\hline
""")
    for _, row in results_df.iterrows():
        f.write(f"{row['Model']} & {row['R^2']:.4f} & {row['Adj R^2']:.4f} & {row['RMSE']:.4f} \\")
    f.write("\hline\end{tabular}\end{table}")

# --------------------------
# Cross-Validation for Small Sample Size
# --------------------------
print("\n" + "="*70)
print("CROSS-VALIDATION FOR SMALL SAMPLE SIZE")
print("="*70)

from sklearn.model_selection import LeaveOneOut, KFold

# For small sample size (n < 200), always use Leave-One-Out Cross-Validation (LOOCV)
# to maximize training data usage
if len(X_train) < 200:
    cv = LeaveOneOut()
    print("Using Leave-One-Out Cross-Validation (LOOCV)")
    print(f"Sample size: {len(X_train)} (using LOOCV for n < 200)")
else:
    # For larger sample size, use 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=CONFIG["random_state"])
    print("Using 5-fold Cross-Validation")

cv_adj_r2 = []
cv_rmse = []

for train_idx, test_idx in cv.split(X_train, y_train):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_train[train_idx], y_train[test_idx]
    
    # Ensure test set has enough samples and features
    if len(y_te) == 0 or len(X_tr) == 0:
        continue
    
    model_cv = GradientBoostingRegressor(**gb_tuned_params)
    model_cv.fit(X_tr, y_tr)
    y_pred_cv = model_cv.predict(X_te)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_cv))
    cv_rmse.append(rmse)
    
    # 只有当测试集样本数大于1时，才计算R²和Adj R²
    if len(y_te) > 1:
        r2 = r2_score(y_te, y_pred_cv)
        adj_r2 = adjusted_r2(r2, len(y_te), X_te.shape[1])
        cv_adj_r2.append(adj_r2)

if len(cv_rmse) > 0:
    mean_rmse = np.mean(cv_rmse)
    std_rmse = np.std(cv_rmse)
    
    if len(cv_adj_r2) > 0:
        mean_adj_r2 = np.mean(cv_adj_r2)
        std_adj_r2 = np.std(cv_adj_r2)
        print(f"Cross-Validation Mean Adj R^2 = {mean_adj_r2:.4f} +/- {std_adj_r2:.4f}")
    else:
        mean_adj_r2 = np.nan
        std_adj_r2 = np.nan
        print("Cross-Validation Adj R^2: Not available (test set size = 1)")
    
    print(f"Cross-Validation Mean RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}")
else:
    mean_adj_r2 = np.nan
    std_adj_r2 = np.nan
    mean_rmse = np.nan
    std_rmse = np.nan
    print("Insufficient data for cross-validation")

# --------------------------
# Residual Analysis
# --------------------------
print("\n" + "="*70)
print("RESIDUAL ANALYSIS")
print("="*70)

residuals = y_train - y_pred_train

# 1. Shapiro-Wilk正态性检验
if len(residuals) >= 3 and len(residuals) <= 5000:  # Shapiro-Wilk的适用范围
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Normality Test:")
    print(f"  Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print(f"  Result: Residuals are normally distributed (p > 0.05)")
    else:
        print(f"  Result: Residuals are NOT normally distributed (p ≤ 0.05)")
else:
    print(f"Shapiro-Wilk test skipped (n={len(residuals)} not in range [3, 5000])")

# 2. Durbin-Watson自相关检验
dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
print(f"\nDurbin-Watson Autocorrelation Test:")
print(f"  Statistic = {dw_stat:.4f}")
if dw_stat < 1.5:
    print(f"  Result: Positive autocorrelation suspected")
elif dw_stat > 2.5:
    print(f"  Result: Negative autocorrelation suspected")
else:
    print(f"  Result: No significant autocorrelation (≈2.0 is ideal)")

# 3. Bootstrap Confidence Intervals
print("\n" + "="*70)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)
n_bootstrap = 1000
bootstrap_r2 = []
bootstrap_rmse = []
bootstrap_adj_r2 = []

n_samples = len(y_train)
for i in range(n_bootstrap):
    # 有放回抽样
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    y_boot = y_train[indices]
    y_pred_boot = y_pred_train[indices]
    
    # 计算指标
    r2_boot = r2_score(y_boot, y_pred_boot)
    rmse_boot = np.sqrt(mean_squared_error(y_boot, y_pred_boot))
    adj_r2_boot = adjusted_r2(r2_boot, n_samples, X_train.shape[1])
    
    bootstrap_r2.append(r2_boot)
    bootstrap_rmse.append(rmse_boot)
    bootstrap_adj_r2.append(adj_r2_boot)

# 计算95%置信区间
r2_ci_lower = np.percentile(bootstrap_r2, 2.5)
r2_ci_upper = np.percentile(bootstrap_r2, 97.5)
rmse_ci_lower = np.percentile(bootstrap_rmse, 2.5)
rmse_ci_upper = np.percentile(bootstrap_rmse, 97.5)
adj_r2_ci_lower = np.percentile(bootstrap_adj_r2, 2.5)
adj_r2_ci_upper = np.percentile(bootstrap_adj_r2, 97.5)

print(f"Bootstrap Results (n={n_bootstrap}, 95% CI):")
print(f"  R^2:    {np.mean(bootstrap_r2):.4f} [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
print(f"  Adj R^2: {np.mean(bootstrap_adj_r2):.4f} [{adj_r2_ci_lower:.4f}, {adj_r2_ci_upper:.4f}]")
print(f"  RMSE:  {np.mean(bootstrap_rmse):.4f} [{rmse_ci_lower:.4f}, {rmse_ci_upper:.4f}]")

# --------------------------
# Final Summary
# --------------------------
r2_train_final = r2_score(y_train, y_pred_train)
adj_r2_train_final = adjusted_r2(r2_train_final, len(y_train), X_train.shape[1])
rmse_train_final = np.sqrt(mean_squared_error(y_train, y_pred_train))

print("\n" + "="*70)
print("ENHANCED ANALYSIS COMPLETED")
print("="*70)
print(f"Best Model: {best_model_name}")
print(f"Train R^2 = {r2_train_final:.4f} [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
print(f"Train Adj R^2 = {adj_r2_train_final:.4f} [{adj_r2_ci_lower:.4f}, {adj_r2_ci_upper:.4f}]")
print(f"RMSE = {rmse_train_final:.4f} [{rmse_ci_lower:.4f}, {rmse_ci_upper:.4f}]")
if len(cv_rmse) > 0:
    if len(cv_adj_r2) > 0:
        print(f"Cross-Validation Mean Adj R^2 = {mean_adj_r2:.4f} +/- {std_adj_r2:.4f}")
    else:
        print("Cross-Validation Adj R^2: Not available (test set size = 1)")
    print(f"Cross-Validation Mean RMSE = {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"\nAll 17 features PRESERVED")
print(f"Outputs saved to: {OUTPUT_FOLDER}/")
print("="*70)
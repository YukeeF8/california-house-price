Markdown
# 🏡 加州房价预测机器学习项目实战总览

本项目基于 Scikit-learn 内置的''加州房价数据集（California Housing）''进行回归分析全流程实战。通过使用工业界极稳健的''随机森林回归（RandomForestRegressor）''算法，实现了从原始数据加载到全新样本预测的完整闭环。

---

## 一、 项目概述

* **项目目标**：根据区域的人口普查与地理特征，预测该街区的房屋价值中位数（典型的**监督学习-回归任务**）。
* **使用数据**：Scikit-learn 内置加州房价数据集（1990 年美国人口普查数据，共 **20,640** 条样本）。
* **使用算法**：随机森林回归（RandomForestRegressor）。
* **核心流程**：数据加载 ➔ 划分数据集 ➔ 特征标准化 ➔ 模型训练 ➔ 模型评估 ➔ 新样本预测。

---

## 二、 完整代码

如果你想在 Jupyter Notebook、Google Colab 或本地 `.py` 脚本中运行，直接复制并运行 加州房价.py 即可

```python
# ==========================================
# 1. 导入工具库与加载数据
# ==========================================
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("【1】正在从网上下载并加载加州房价数据...")
california = fetch_california_housing(as_frame=True)
df = california.data
df['MedHouseValue'] = california.target  # 添加目标列：房价中位数

# ==========================================
# 2. 划分特征 (X) 与目标值 (y)
# ==========================================
X = df.drop(columns=['MedHouseValue'])
y = df['MedHouseValue']

# ==========================================
# 3. 划分训练集与测试集 (80% vs 20%)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"【2】数据集划分完毕。训练集：{X_train.shape[0]}行，测试集：{X_test.shape[0]}行。")

# ==========================================
# 4. 数据标准化 (特征缩放)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("【3】数据特征标准化缩放完成。")

# ==========================================
# 5. 构建并训练随机森林模型
# ==========================================
print("【4】正在构建包含 100 棵树的随机森林，并开始训练（拟合）...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("     模型训练完毕！")

# ==========================================
# 6. 模型评估 (期末考试打分)
# ==========================================
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 【5】模型期末考试成绩单：")
print(f"   - R² 决定系数: {r2:.4f}")
print(f"     (含义：模型看穿并解释了测试集中 {r2*100:.1f}% 的房价波动原因)")
print(f"   - 平均绝对误差 (MAE): {mae:.4f}")
print(f"     (含义：平均下来，模型每套房子的预测偏离了真实房价约 {mae*10:.2f} 万美元)")

# ==========================================
# 7. 全新未知样本预测
# ==========================================
print("\n🔮 【6】尝试对一个全新街区的未知样本进行房价预测...")
# 假定新街区的 8 个特征指标：[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
new_house = [[8.0, 30, 6, 1, 500, 300, 37.8, -122.2]]

# 先用训练集同样的 scaler 标准化，再送入模型预测
new_house_scaled = scaler.transform(new_house)
predict_price = model.predict(new_house_scaled)

print(f"   - 该全新街区的预测房价中位数为: {predict_price[0]:.4f} (十万美元)")
print(f"   - 💰 折合实际价格: ${predict_price[0]*100000:,.2f} 美元")

三、 原理解析
1. 导入工具库
所有工具均来自 Python 机器学习标配库 sklearn（Scikit-learn）。它负责了数据下载、数据集切分、数据标准化缩放、随机森林算法建模、以及最终的成绩评估。

2. 加载并整理数据
as_frame=True：将原始的纯数组格式转化为结构化的 Pandas DataFrame 表格，极大方便了后续的可视化与处理。

MedHouseValue：拼进入表格的目标列/标签列（Target，因变量），代表该街区房价中位数（单位：十万美元）。

3. 划分特征与目标值
X（特征）：自变量。剥离掉房价后的 8 列物理和经济指标（如收入、房龄、经纬度），作为模型的“输入线索”。

y（标签）：因变量。单独提取出来的房价列。在监督学习中，必须显式地把标准答案告诉模型，它才能进行针对性学习。

4. 划分训练集和测试集
test_size=0.2：将 80% 的数据划归为训练集，20% 的数据划归为测试集。考试时只给模型看题目（X_test），看它猜出来的房价和真实答案（y_test）能对上多少。

random_state=42：固定随机数种子。保证实验的可复现性，确保任何人运行这段代码切分出的样本都完全一致。

目的：防止模型过拟合，用全新数据检验模型的泛化能力。

5. 数据标准化（特征缩放）
原因：数据集中的“总人口数 Population”动辄上千，而“平均卧室数 AveBedrms”基本是个位数。如果直接喂给模型，算法往往会误认为数值大的特征更重要。

原理：StandardScaler 将所有特征转化为均值为 0、方差为 1 的标准正态分布，抹除量纲影响，让各特征站在同一起跑线上。

规范细节：训练集使用 fit_transform（边计算训练集的均值边转换）；测试集只使用 transform（直接借用训练集的规则转换），而不能在测试集上 fit，否则属于“数据泄露”的作弊行为。

6. 构建并训练随机森林模型
随机森林（Random Forest）：属于集成学习（Ensemble Learning）。它在后台同时构建 100 棵（由 n_estimators=100 指定）独立的决策树。

底层机制：每棵树随机抽取一部分样本、一部分特征进行层层条件割裂（如：收入是否大于5万？是否靠近海边？）。输入新数据后，100 棵树各自给出预测值，最终取所有树的平均值作为输出。这种机制使其比单棵决策树更准，且极不易过拟合。

7. 模型评估与打分
R2分数（决定系数）：衡量模型拟合优度。范围在 0~1 之间，越接近 1 越好。跑完随机森林代码通常能达到 0.80 - 0.81，意味着加州房价 80% 以上的错综复杂波动原因，已被模型看穿并合理解释。

MAE（平均绝对误差）：真实房价与预测房价差值的绝对值的平均。本案中 MAE≈0.32，意味着平均下来，模型每套房子会猜偏 3.2 万美元 左右。整体表现优秀。

8. 全新未知样本预测
⚠️ 高频误区：任意新样本（如自定义的 new_house 矩阵）在喂给模型前，必须先通过同一个 scaler.transform 进行相同的缩放，否则由于尺度不一致，模型会输出极其离谱的错误预测值。



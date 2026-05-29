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

# 三、原理解析

### 1. 导入工具库
所有工具均来自 Python 机器学习标配库 **sklearn（Scikit-learn）**。它负责数据下载、数据集切分、数据标准化缩放、随机森林算法建模以及模型效果评估。

---

### 2. 加载并整理数据
- `as_frame=True`：将原始数组转为 Pandas DataFrame 表格，方便后续数据处理与可视化。
- `MedHouseValue`：目标标签列（Target/因变量），代表街区房价中位数，单位：十万美元。

---

### 3. 划分特征与目标值
- **X（特征/自变量）**：剔除房价列后剩余8项指标（收入、房龄、经纬度等），作为模型输入。
- **y（标签/因变量）**：单独提取的房价列。监督学习需提供标准答案，供模型训练学习。

---

### 4. 划分训练集和测试集
- `test_size=0.2`：80% 数据作为训练集，20% 作为测试集，用未知数据检验预测效果。
- `random_state=42`：固定随机种子，保证代码运行结果可复现。
- 作用：避免模型**过拟合**，验证模型泛化能力。

---

### 5. 数据标准化（特征缩放）
1. **问题原因**：不同特征数值量级差距大（人口数千、卧室数为个位数），易导致模型权重判断偏差。
2. **实现原理**：使用 `StandardScaler` 将所有特征转换为**均值0、方差1**的标准正态分布，消除量纲影响。
3. **使用规范**
   - 训练集：`fit_transform`（计算统计值 + 完成转换）
   - 测试集/新数据：仅用 `transform`（复用训练集规则）
   - 禁止在测试集执行 `fit`，会造成**数据泄露**。

---

### 6. 构建并训练随机森林模型
随机森林（Random Forest）属于**集成学习**，参数 `n_estimators=100` 代表构建 100 棵独立决策树。

运行逻辑：
每棵树随机选取部分样本与特征进行判断；新数据输入后，综合100棵树的预测结果取平均值输出。相比单棵决策树，预测精度更高、更难出现过拟合。

---

### 7. 模型评估与打分
- **R² 决定系数**：衡量拟合效果，取值 0~1，越接近1效果越好。本次结果约 0.80~0.81，模型可解释80%以上房价波动规律。
- **MAE 平均绝对误差**：真实值与预测值误差的平均值。本次 MAE≈0.32，即平均预测偏差约 3.2 万美元，模型表现良好。

---

### 8. 全新未知样本预测
> ⚠️ 重要误区
自定义新样本（如 `new_house`）输入模型前，**必须使用同一个 scaler 执行 transform 缩放**。若跳过此步骤，特征尺度不统一，会出现严重预测错误。

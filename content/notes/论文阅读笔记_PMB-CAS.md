# 论文阅读：PMB-CAS — 多目标神经架构搜索实现认知诊断的准确性与可解释性平衡

> **原文标题**：Multi-Objective Neural Architecture Search for Cognitive Diagnosis: Balancing Accuracy and Interpretability With Probabilistic Models
>
> **作者**：Yifei Sun, Mingkai Duan, Sicheng Hou, Shi Cheng, Maoguo Gong, Zhi-Hui Zhan
>
> **发表**：IEEE Transactions on Computational Social Systems, 2026
>
> **DOI**：10.1109/TCSS.2026.3689070

---

## 一、研究背景与问题

### 背景
- **认知诊断（CD）**：通过分析学生的答题数据，推断其对细粒度知识概念的掌握程度，是个性化学习的基础
- 传统的认知诊断模型（CDM）分两类：
  - **心理测量模型**（IRT、DINA、G-DINA）：结构简单、可解释性强，但表达能力有限
  - **深度学习模型**（NCDM、MCD、KSCD）：精度高，但结构复杂、**不可解释**（"黑箱模型"）

### 核心矛盾
> 精度 vs 可解释性 — 二者的权衡是智能教育系统落地的关键障碍

### 现有局限
1. 传统模型简单但精度不够
2. 深度模型精度高但不透明，教育者难以信任
3. 模型设计**依赖人工**，耗时且受限于已有的理论框架

---

## 二、核心贡献

| 贡献 | 说明 |
|------|------|
| **框架创新** | 首个专为 CD 设计的多目标 NAS 框架，将可解释性与精度作为**同等重要的优化目标** |
| **搜索空间设计** | 新颖的**树形搜索空间**，融合心理测量算子（IRT）和深度神经网络算子 |
| **可解释性度量** | 提出算子加权模型可解释性评分 **MIS**（Model Interpretability Score），基于算子语义复杂度而非简单深度约束 |

---

## 三、方法论

### 3.1 整体流程

PMB-CAS 是一个迭代式的 **EDA（估计分布算法）** 进化过程：

```
初始化种群 → 评估（AUC + MIS）→ 精英选择（NSGA-II）→ 
更新概率模型 → 从模型采样新架构 → 循环直至收敛
```

### 3.2 搜索空间

#### 终端集（叶子节点）
从 CD 理论出发，定义四种特征嵌入：

| 终端 | 含义 | 维度 |
|------|------|------|
| $h_s$ | 学生能力 | $D_k$ |
| $h_{e,diff}$ | 习题难度 | $D_k$ |
| $h_{kc}$ | 知识点关联（Q-matrix） | $D_k$ |
| $h_{e,disc}$ | 习题区分度 | 1 |

#### 算子集（内部节点）

| 类别 | 算子 | 可解释性成本 |
|------|------|:---:|
| 算术 | +, -, ×, Square, Abs | 1（低） |
| 激活 | Sigmoid, Tanh, Softplus | 2 |
| 聚合 | Sum, Mean | 2 |
| 神经网络 | FFN, Concat | 5（高） |
| 领域专用 | NCDM-Core | 5（高） |

关键：引入了 **NCDM-Core** 作为高元领域专用算子，让搜索能站在已有知识的肩膀上：

$$f(h_s, h_{e,diff}, h_{kc}, h_{e,disc}) = h_{e,disc} \cdot \sigma((h_s - h_{e,diff}) \odot h_{kc})$$

### 3.3 多目标优化

目标函数：

$$\max_{\mathcal{A} \in \mathcal{S}} F(\mathcal{A}) = (f_{auc}(\mathcal{A}), f_{int}(\mathcal{A}))$$

- **$f_{auc}$**：诊断精度（验证集 AUC）
- **$f_{int}$**：拓扑可解释性评分（MIS）

MIS 的计算：

$$C_{total}(\mathcal{A}) = \sum_{i \in nodes(\mathcal{A})} C(op_i)$$

$$f_{int}(\mathcal{A}) = \max\left(0, 1 - (w_d \frac{d}{d_{max}} + w_c \frac{C_{total}(\mathcal{A})}{C_{max}})\right)$$

最终得到 **Pareto 前沿** —— 一组非支配解，供使用者根据需求选择。

### 3.4 概率模型引导搜索（EDA核心）

相比传统 EA 的随机变异/交叉，PMB-CAS：

1. **选择精英**：基于 NSGA-II 的非支配排序 + 拥挤度距离
2. **学习分布**：从精英架构中统计学习条件概率（父算子 → 子节点类型/具体算子）
3. **采样新架构**：从概率模型中递归采样生成新树，使用**温度参数 $\tau$** 控制探索-利用平衡
   - $\tau$ 高 → 广泛探索
   - $\tau$ 低 → 局部精调（逐步退火）
4. **拉普拉斯平滑**：防止未见过结构概率为零，保持多样性

### 3.5 计算复杂度

每轮复杂度 $O(N \cdot (\mathcal{C}_{eval} + \mathcal{C}_{update} + \mathcal{C}_{sample}))$

- 瓶颈在 $\mathcal{C}_{eval}$：需要训练 N 个 CDM
- 对于最大的数据集 ASSIST09，50 轮搜索约需 **144 GPU 小时**（RTX 4060）
- 但推理极快：单次学生-习题交互 **< 2.5ms**

---

## 四、实验结果

### 数据集

| 数据集 | 学生数 | 习题数 | 知识点 | 交互记录 | 稀疏度 |
|--------|:------:|:------:|:------:|:--------:|:------:|
| ASSIST09 | 4163 | 17746 | 123 | 346924 | 99.53% |
| Junyi | 10747 | 708 | 25 | 707842 | 99.07% |
| SLP | 2336 | 3021 | 749 | 995680 | 99.86% |

### 主要结果

**PMB-CAS-D（高精度版本）在三个数据集上全面超越 SOTA：**

| 数据集 | IRT | DINA | NCDM | KSCD | **PMB-CAS-D** |
|--------|:---:|:----:|:----:|:----:|:-------------:|
| ASSIST09 | 0.7123 | 0.7105 | 0.7421 | 0.7433 | **0.7752** |
| Junyi | 0.8123 | 0.8015 | 0.8421 | 0.8405 | **0.8567** |
| SLP | 0.6856 | 0.6712 | 0.7256 | 0.7223 | **0.7512** |

**可解释性版本 PMB-CAS-A** 的拓扑可解释性评分高达 **0.92**（接近满分 1.0），远超 NCDM 的 0.45。

### 消融实验
- PMB-CAS **搜索效率**显著优于 EA-NAS 和 Random Search
- PMB-CAS **稳定性更好**（多次运行方差小）
- MIS 的超参数 $(w_d, w_c)$ 在广泛范围内相关性 > 0.85，**对扰动不敏感**

---

## 五、亮点与不足

### 亮点
1. **首次**将 CD 模型设计形式化为多目标优化问题（AUC vs 可解释性）
2. 算子加权 MIS 比简单的深度/参数量更公平地衡量可解释性
3. 树形搜索空间自然兼容心理测量和深度学习算子，有理论深度
4. 温度退火机制优雅地平衡了探索与利用
5. 给出 Pareto 前沿而非单一模型，实用性更强

### 不足之处（原文自述）
1. 搜索空间局限于预定义算子，可能无法捕捉某些领域的高阶交互
2. 可解释性目前限于**拓扑层面**，与深层教育理论的连接需专家验证
3. 离线搜索计算量大（144 GPU 小时）

### 个人思考
- MIS 本质上是一种**结构正则化**，类似于机器学习中的 L1/L2，但放在了架构层面
- 实验发现 **square(sub(...))** 模式在有监督对比下效果突出，说明减法后的非线性变换可能是 CD 的关键结构模因
- 如果结合 LLM 做算子推荐，可能进一步拓展搜索空间

---

## 六、关键术语

| 缩写 | 全称 |
|------|------|
| CD | Cognitive Diagnosis / 认知诊断 |
| CDM | Cognitive Diagnosis Model / 认知诊断模型 |
| NAS | Neural Architecture Search / 神经架构搜索 |
| MONAS | Multi-Objective NAS / 多目标神经架构搜索 |
| EDA | Estimation of Distribution Algorithm / 估计分布算法 |
| MIS | Model Interpretability Score / 模型可解释性评分 |
| MOOP | Multi-Objective Optimization Problem / 多目标优化问题 |
| AUC | Area Under the Curve / ROC曲线下面积 |
| IRT | Item Response Theory / 项目反应理论 |
| DINA | Deterministic Inputs, Noisy "And" Gate |
| NCDM | Neural Cognitive Diagnosis Model |

---

*阅读日期：2026-05-31*

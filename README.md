# CET6 听力预测系统

<div align="center">

![Version](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-orange)
![License](https://img.shields.io/badge/协议-MIT-red)

**基于多模型融合的大学英语六级听力答案预测系统**

[功能特性](#功能特性) • [技术架构](#技术架构) • [快速开始](#快速开始) • [模型说明](#模型说明) • [特别声明](#特别声明)

</div>

---

## 📖 系统介绍

CET6 听力预测系统是一个基于历史数据分析和多模型融合的智能预测平台，通过对 **9年（2016-2024）** 六级听力真题答案的深度挖掘，运用统计学方法和机器学习算法，为用户提供未来考试的答案趋势预测参考。

### 核心理念

本系统的核心是**数据驱动**：通过分析历史答案的分布规律、位置特征、时间趋势等多维度信息，结合多种预测模型的集成投票，生成具有一定参考价值的预测结果。

---

## ✨ 功能特性

| 功能模块 | 描述 |
|----------|------|
| 📊 **数据概览** | 历史考试统计、ABCD选项分布饼图、25题位置频率堆叠图 |
| 🔮 **智能预测** | 多时间点预测（2025-2026年）、置信度分析、一键复制答案 |
| 📈 **年度分析** | 逐年答案分布对比、每套试卷分布、25题位置热力图 |
| 📉 **趋势变化** | 历年ABCD占比趋势折线图、数据洞察分析 |
| 🧪 **模型回测** | 11+模型逐年回测、最佳模型自动选择、预测vs真实对比 |
| 📚 **原题练习** | 跳转外部真题平台、配套听力音频 |
| 📜 **历史数据** | 36套历史真题答案、年份搜索过滤 |
| 🌓 **主题切换** | 白天/黑夜自动切换（6:00-18:00）、手动切换 |
| 📱 **响应式设计** | 完美适配桌面端和移动端 |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Frontend)                        │
│  HTML5 + CSS3 + JavaScript + Chart.js + Lucide Icons    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   后端 (Backend)                          │
│              Flask RESTful API (Python)                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  预测引擎 (Model Engine)                   │
│    统计模型 + 集成模型 + 机器学习模型 (11+ Models)           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   数据层 (Data Layer)                     │
│           CSV 历史数据 (2016-2024, 36套试卷)               │
└─────────────────────────────────────────────────────────┘
```

### 技术栈

- **前端**: HTML5, CSS3 (响应式), JavaScript (ES6+), Chart.js 4.x, Lucide Icons
- **后端**: Python 3.8+, Flask 2.x
- **数据分析**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装步骤

```bash
# 1. 进入项目目录
cd 六级听力预测分析

# 2. 安装依赖
pip install flask pandas numpy scikit-learn xgboost

# 3. 启动服务
python app.py

# 4. 访问系统
# 浏览器打开 http://127.0.0.1:5001
```

### 目录结构

```
六级听力预测分析/
├── app.py                          # Flask 主程序
├── 六级听力预测模型.py               # 预测模型核心
├── 六级听力答案_2025-12-04.csv      # 历史数据
├── README.md                        # 说明文档
├── static/
│   ├── css/
│   │   └── style.css               # 样式文件
│   └── js/
│       └── app.js                  # 前端逻辑
└── templates/
    └── index.html                  # 页面模板
```

---

## 🤖 模型说明

### 统计模型

| 模型名称 | 原理说明 |
|----------|----------|
| **趋势模型** | 分析最近10套试卷的答案变化趋势 |
| **加权频率** | 近年数据权重更高，指数衰减 |
| **滑动窗口** | 只使用最近8套试卷的统计数据 |
| **指数加权** | 时间衰减因子 α=0.9 的加权统计 |
| **马尔可夫链** | 基于前一题答案的转移概率预测 |
| **贝叶斯推断** | 使用后验概率进行预测 |
| **周期模型** | 分析6月/12月考试的周期差异 |
| **反模式** | 假设出题者避免与最近试卷相同 |
| **N-gram** | 基于前2题答案的序列模式 |

### 集成模型

| 模型名称 | 原理说明 |
|----------|----------|
| **投票集成** | 多模型等权投票 |
| **加权投票** | 根据历史表现分配模型权重 |

### 机器学习模型

| 模型名称 | 原理说明 |
|----------|----------|
| **随机森林** | 100棵决策树集成 |
| **XGBoost** | 梯度提升树 |
| **MLP神经网络** | 4层全连接网络 (256-128-64-32) |
| **逻辑回归** | 多分类逻辑回归 |
| **梯度提升** | Scikit-learn 梯度提升分类器 |

---

## 📊 数据来源

### 数据集

- **时间跨度**: 2016年6月 - 2024年12月
- **试卷数量**: 36套
- **题目总数**: 900题 (36套 × 25题)
- **数据格式**: CSV (UTF-8编码)

### 数据字段

| 字段名 | 说明 |
|--------|------|
| 年份 | 考试年份 (2016-2024) |
| 月份 | 考试月份 (6月/12月) |
| 套数 | 试卷编号 (1-3) |
| T1-T25 | 第1-25题答案 (A/B/C/D) |
| 完整答案 | 25题答案连续字符串 |

### 数据来源说明

历史答案数据来源于公开发布的六级听力真题标准答案，经人工校验整理。

---

## 💡 创新点

### 1. 多模型融合架构
- 整合 **11+ 种预测模型**，包括统计模型、集成模型和机器学习模型
- 自动选择每年表现最佳的模型进行预测
- 加权投票机制综合多模型优势

### 2. 动态回测验证
- 逐年滚动回测（留一法验证）
- 实时展示预测 vs 真实答案对比
- 自动计算各模型历史准确率

### 3. 多维度可视化
- 25题位置热力图（稳定/波动标记）
- 年度答案分布饼图
- 历年趋势折线图
- 逐年回测详情展开

### 4. 智能主题适配
- 根据时间自动切换白天/黑夜模式
- 图表颜色随主题同步更新

### 5. 移动端优先设计
- 响应式布局，完美适配手机/平板
- 底部TabBar导航，操作便捷
- 触摸友好的交互设计

---

## ⚠️ 特别声明

### 关于预测准确率

> **重要提示**: 六级听力答案本质上具有**高度随机性**，不存在可被精确预测的规律。

- 理论随机概率: **25%** (4选1)
- 本系统最佳模型准确率: **约30%** (略高于随机)
- 准确率提升主要来自于答案分布的**统计偏差**，而非出题规律

### 为什么准确率有限？

1. **答案设计随机性**: 标准化考试的答案分布经过专业设计，刻意避免可预测的规律
2. **样本量限制**: 9年36套试卷的数据量不足以发现强统计规律
3. **题目内容差异**: 每次考试内容完全不同，答案与题目内容强相关
4. **防作弊机制**: 命题方会刻意打破历史规律

### 正确使用姿势

✅ **推荐用途**:
- 作为学习数据分析和机器学习的案例项目
- 了解预测模型的构建和评估方法
- 分析历史答案的统计特征

❌ **不推荐用途**:
- 依赖预测结果参加考试
- 将预测答案作为复习依据

---

## 👨‍💻 作者信息

<div align="center">

**开发者: 传康 kk**

| 联系方式 | 信息 |
|----------|------|
| 微信 | 1837620622 |
| 邮箱 | 2040168455@qq.com |
| 咸鱼 | 万能程序员 |
| B站 | 万能程序员 |

</div>

---

## 📄 开源协议

本项目采用 **MIT 协议** 开源。

```
MIT License

Copyright (c) 2024 传康 kk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 使用须知

1. ✅ 本项目**仅供学习研究使用**
2. ❌ 禁止用于任何商业用途
3. ❌ 禁止用于考试作弊或其他非法用途
4. ⚠️ 使用本系统造成的任何后果由使用者自行承担

---

<div align="center">

**如果觉得有帮助，欢迎 ⭐ Star 支持！**

Made with ❤️ by 传康 kk

</div>

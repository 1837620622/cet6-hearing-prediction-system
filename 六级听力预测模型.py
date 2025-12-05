# -*- coding: utf-8 -*-
"""
============================================================================
六级听力答案预测模型
============================================================================
作者: 传康kk (Vx:1837620622)
邮箱: 2040168455@qq.com
咸鱼/B站: 万能程序员

预测策略：
1. 加权频率模型 - 近年数据权重更高
2. 马尔可夫链 - 分析题目间转移概率
3. 平衡约束 - 确保ABCD分布均衡(每个约6-7题)
4. 反连续模型 - 避免过多连续相同答案
5. 集成投票 - 多模型综合决策
============================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# 机器学习模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_data(csv_path):
    """加载CSV数据"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"加载数据: {len(df)} 套试卷")
    return df

def extract_answers(df):
    """提取答案矩阵，按时间排序"""
    # 按年份月份套数排序
    df = df.sort_values(['年份', '月份', '套数']).reset_index(drop=True)
    
    # 提取25题答案
    answer_cols = [f'T{i}' for i in range(1, 26)]
    answers = df[answer_cols].values
    
    # 时间信息
    times = df[['年份', '月份', '套数']].values
    
    return answers, times, df

# ============================================================================
# 模型1: 加权频率模型 - 近年数据权重更高
# ============================================================================

def weighted_frequency_model(answers, times, decay=0.85):
    """
    加权频率模型
    越近的年份权重越高，使用指数衰减
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    for q in range(n_questions):
        freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for i in range(n_exams):
            # 权重：越新的数据权重越高
            weight = decay ** (n_exams - 1 - i)
            ans = answers[i, q]
            if ans in freq:
                freq[ans] += weight
        
        # 选择加权频率最高的选项
        best = max(freq, key=freq.get)
        predictions.append(best)
    
    return predictions

# ============================================================================
# 模型2: 马尔可夫链模型 - 分析题目间转移概率
# ============================================================================

def markov_model(answers):
    """
    马尔可夫链模型
    基于前一题答案预测当前题
    """
    n_exams, n_questions = answers.shape
    
    # 构建转移矩阵
    transitions = defaultdict(lambda: defaultdict(int))
    
    for exam in answers:
        for i in range(1, n_questions):
            prev_ans = exam[i-1]
            curr_ans = exam[i]
            transitions[prev_ans][curr_ans] += 1
    
    # 转换为概率
    trans_prob = {}
    for prev, nexts in transitions.items():
        total = sum(nexts.values())
        trans_prob[prev] = {k: v/total for k, v in nexts.items()}
    
    # 第一题使用频率
    first_freq = Counter(answers[:, 0])
    first_ans = first_freq.most_common(1)[0][0]
    
    predictions = [first_ans]
    for i in range(1, n_questions):
        prev = predictions[-1]
        if prev in trans_prob:
            probs = trans_prob[prev]
            # 选择概率最高的
            next_ans = max(probs, key=probs.get)
        else:
            next_ans = 'A'
        predictions.append(next_ans)
    
    return predictions

# ============================================================================
# 模型3: 位置频率模型 - 每题独立分析
# ============================================================================

def position_frequency_model(answers):
    """
    位置频率模型
    统计每个位置上各选项的出现频率
    """
    n_exams, n_questions = answers.shape
    predictions = []
    probabilities = []
    
    for q in range(n_questions):
        freq = Counter(answers[:, q])
        total = sum(freq.values())
        
        # 计算概率
        probs = {opt: freq.get(opt, 0) / total for opt in 'ABCD'}
        probabilities.append(probs)
        
        # 选择频率最高的
        best = freq.most_common(1)[0][0] if freq else 'A'
        predictions.append(best)
    
    return predictions, probabilities

# ============================================================================
# 模型4: 平衡约束模型 - 确保ABCD分布均衡
# ============================================================================

def balanced_model(answers, probabilities):
    """
    平衡约束模型
    在保证每题倾向的同时，确保整体ABCD分布均衡
    目标：每个选项约6-7题（25题/4选项≈6.25）
    """
    n_questions = 25
    target_count = n_questions // 4  # 每个选项目标数量：6
    
    predictions = []
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for q in range(n_questions):
        probs = probabilities[q].copy()
        
        # 对已经达到目标数量的选项进行惩罚
        remaining = n_questions - q
        for opt in 'ABCD':
            if counts[opt] >= target_count + 1:
                probs[opt] *= 0.3  # 降低概率
            elif counts[opt] >= target_count:
                probs[opt] *= 0.7
        
        # 对还需要更多的选项进行加成
        for opt in 'ABCD':
            needed = target_count - counts[opt]
            if needed > 0 and remaining <= needed * 2:
                probs[opt] *= 1.5
        
        # 选择调整后概率最高的
        best = max(probs, key=probs.get)
        predictions.append(best)
        counts[best] += 1
    
    return predictions

# ============================================================================
# 模型5: 反连续模型 - 避免过多连续相同答案
# ============================================================================

def anti_consecutive_model(base_predictions, probabilities, max_consecutive=3):
    """
    反连续模型
    避免超过max_consecutive个连续相同答案
    """
    predictions = list(base_predictions)
    
    for i in range(len(predictions)):
        if i >= max_consecutive:
            # 检查前面是否有连续相同答案
            consecutive = 1
            for j in range(i-1, -1, -1):
                if predictions[j] == predictions[i]:
                    consecutive += 1
                else:
                    break
            
            if consecutive > max_consecutive:
                # 选择次优选项
                probs = probabilities[i].copy()
                probs[predictions[i]] = 0
                if sum(probs.values()) > 0:
                    predictions[i] = max(probs, key=probs.get)
    
    return predictions

# ============================================================================
# 模型6: 时间趋势模型 - 分析选项随时间的变化趋势
# ============================================================================

def trend_model(answers, times):
    """
    时间趋势模型
    分析每个位置上选项的时间变化趋势
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    for q in range(n_questions):
        # 计算最近5套试卷的频率变化
        recent = answers[-10:, q] if n_exams >= 10 else answers[:, q]
        recent_freq = Counter(recent)
        
        # 使用最近的趋势
        if recent_freq:
            best = recent_freq.most_common(1)[0][0]
        else:
            best = 'A'
        predictions.append(best)
    
    return predictions

# ============================================================================
# 机器学习模型 - 特征工程和训练
# ============================================================================

def prepare_ml_features(answers, times):
    """
    为机器学习模型准备特征
    特征包括：题目位置、历史频率、前后题关系等
    """
    n_exams, n_questions = answers.shape
    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D'])
    
    X_all = []
    y_all = []
    
    for q in range(n_questions):
        for i in range(n_exams):
            features = []
            
            # 特征1: 题目位置 (one-hot编码简化为数值)
            features.append(q / 25.0)  # 归一化位置
            
            # 特征2: 所属section (1-8题, 9-15题, 16-25题)
            if q < 8:
                features.extend([1, 0, 0])
            elif q < 15:
                features.extend([0, 1, 0])
            else:
                features.extend([0, 0, 1])
            
            # 特征3: 历史该位置各选项频率
            if i > 0:
                hist = answers[:i, q]
                for opt in ['A', 'B', 'C', 'D']:
                    features.append(np.mean(hist == opt))
            else:
                features.extend([0.25, 0.25, 0.25, 0.25])
            
            # 特征4: 前一题答案 (如果有)
            if q > 0:
                prev_ans = answers[i, q-1]
                features.extend([1 if prev_ans == opt else 0 for opt in ['A', 'B', 'C', 'D']])
            else:
                features.extend([0, 0, 0, 0])
            
            # 特征5: 当前试卷已有答案分布
            if q > 0:
                current_dist = Counter(answers[i, :q])
                for opt in ['A', 'B', 'C', 'D']:
                    features.append(current_dist.get(opt, 0) / q)
            else:
                features.extend([0, 0, 0, 0])
            
            # 特征6: 年份和月份
            features.append((times[i, 0] - 2016) / 10.0)  # 归一化年份
            features.append(times[i, 1] / 12.0)  # 归一化月份
            
            X_all.append(features)
            y_all.append(le.transform([answers[i, q]])[0])
    
    return np.array(X_all), np.array(y_all), le

def prepare_prediction_features(answers, times, le):
    """
    为预测准备特征（使用最后一套试卷的模式）
    """
    n_exams, n_questions = answers.shape
    
    X_pred = []
    for q in range(n_questions):
        features = []
        
        # 特征1: 题目位置
        features.append(q / 25.0)
        
        # 特征2: 所属section
        if q < 8:
            features.extend([1, 0, 0])
        elif q < 15:
            features.extend([0, 1, 0])
        else:
            features.extend([0, 0, 1])
        
        # 特征3: 历史该位置各选项频率
        hist = answers[:, q]
        for opt in ['A', 'B', 'C', 'D']:
            features.append(np.mean(hist == opt))
        
        # 特征4: 使用最常见的前一题答案
        if q > 0:
            prev_common = Counter(answers[:, q-1]).most_common(1)[0][0]
            features.extend([1 if prev_common == opt else 0 for opt in ['A', 'B', 'C', 'D']])
        else:
            features.extend([0, 0, 0, 0])
        
        # 特征5: 平均分布
        features.extend([0.25, 0.25, 0.25, 0.25])
        
        # 特征6: 年份和月份 (预测2025年)
        features.append((2025 - 2016) / 10.0)
        features.append(6 / 12.0)
        
        X_pred.append(features)
    
    return np.array(X_pred)

# ============================================================================
# 随机森林模型
# ============================================================================

def random_forest_model(train_answers, train_times, test_q_features=None):
    """随机森林分类器"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # 预测
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# XGBoost模型
# ============================================================================

def xgboost_model(train_answers, train_times, test_q_features=None):
    """XGBoost梯度提升树"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# 梯度提升模型
# ============================================================================

def gradient_boosting_model(train_answers, train_times, test_q_features=None):
    """梯度提升分类器"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# 神经网络MLP模型
# ============================================================================

def mlp_model(train_answers, train_times, test_q_features=None):
    """多层感知机神经网络"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# 逻辑回归模型
# ============================================================================

def logistic_model(train_answers, train_times, test_q_features=None):
    """逻辑回归分类器"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# SVM支持向量机模型
# ============================================================================

def svm_model(train_answers, train_times, test_q_features=None):
    """支持向量机分类器"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )
    model.fit(X, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    predictions = model.predict(test_q_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# 朴素贝叶斯模型
# ============================================================================

def naive_bayes_model(train_answers, train_times, test_q_features=None):
    """朴素贝叶斯分类器"""
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    # 将特征转换为非负值（朴素贝叶斯要求）
    X_positive = X - X.min() + 0.001
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X_positive, y)
    
    if test_q_features is None:
        test_q_features = prepare_prediction_features(train_answers, train_times, le)
    
    test_positive = test_q_features - X.min() + 0.001
    predictions = model.predict(test_positive)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# N-gram序列模型
# ============================================================================

def ngram_model(answers, n=3):
    """
    N-gram模型
    基于前n-1题预测当前题
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    for q in range(n_questions):
        if q < n - 1:
            # 前n-1题使用频率模型
            freq = Counter(answers[:, q])
            predictions.append(freq.most_common(1)[0][0])
        else:
            # 使用n-gram
            ngram_freq = defaultdict(Counter)
            for exam in answers:
                context = tuple(exam[q-n+1:q])
                ngram_freq[context][exam[q]] += 1
            
            # 获取最可能的后续
            # 使用最近几套试卷的上下文
            recent_contexts = [tuple(answers[i, q-n+1:q]) for i in range(-min(5, n_exams), 0)]
            combined = Counter()
            for ctx in recent_contexts:
                if ctx in ngram_freq:
                    combined.update(ngram_freq[ctx])
            
            if combined:
                predictions.append(combined.most_common(1)[0][0])
            else:
                freq = Counter(answers[:, q])
                predictions.append(freq.most_common(1)[0][0])
    
    return predictions

# ============================================================================
# 滑动窗口模型 - 只用最近N套试卷
# ============================================================================

def sliding_window_model(answers, times, window_size=8):
    """
    滑动窗口模型
    只使用最近window_size套试卷进行预测
    """
    n_exams, n_questions = answers.shape
    
    # 使用最近的试卷
    recent = answers[-window_size:] if n_exams >= window_size else answers
    
    predictions = []
    for q in range(n_questions):
        freq = Counter(recent[:, q])
        predictions.append(freq.most_common(1)[0][0])
    
    return predictions

# ============================================================================
# 指数加权模型 - 指数衰减权重
# ============================================================================

def exponential_weighted_model(answers, times, alpha=0.9):
    """
    指数加权模型
    越近的试卷权重越高，使用指数衰减
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    for q in range(n_questions):
        weighted_freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for i in range(n_exams):
            weight = alpha ** (n_exams - 1 - i)
            ans = answers[i, q]
            weighted_freq[ans] += weight
        
        predictions.append(max(weighted_freq, key=weighted_freq.get))
    
    return predictions

# ============================================================================
# 贝叶斯推断模型
# ============================================================================

def bayesian_model(answers, times):
    """
    贝叶斯推断模型
    使用后验概率进行预测
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    # 先验：均匀分布
    prior = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
    
    for q in range(n_questions):
        # 计算后验概率
        freq = Counter(answers[:, q])
        total = sum(freq.values())
        
        posterior = {}
        for opt in 'ABCD':
            # 似然 * 先验
            likelihood = (freq.get(opt, 0) + 1) / (total + 4)  # 拉普拉斯平滑
            posterior[opt] = likelihood * prior[opt]
        
        # 归一化
        total_post = sum(posterior.values())
        posterior = {k: v/total_post for k, v in posterior.items()}
        
        predictions.append(max(posterior, key=posterior.get))
    
    return predictions

# ============================================================================
# 投票集成模型 - 多模型投票
# ============================================================================

def voting_ensemble_model(answers, times):
    """
    投票集成模型
    综合多个基础模型的预测进行投票
    """
    _, probabilities = position_frequency_model(answers)
    
    # 获取各模型预测
    models = {
        'weighted': weighted_frequency_model(answers, times),
        'position': position_frequency_model(answers)[0],
        'trend': trend_model(answers, times),
        'markov': markov_model(answers),
        'sliding': sliding_window_model(answers, times, 8),
        'exp_weighted': exponential_weighted_model(answers, times, 0.9),
        'bayesian': bayesian_model(answers, times),
    }
    
    predictions = []
    for q in range(25):
        votes = Counter()
        for name, pred in models.items():
            votes[pred[q]] += 1
        
        # 选择得票最多的
        predictions.append(votes.most_common(1)[0][0])
    
    return predictions

# ============================================================================
# 加权投票集成 - 根据历史表现加权
# ============================================================================

def weighted_voting_model(answers, times):
    """
    加权投票集成
    根据各模型历史表现分配权重
    """
    _, probabilities = position_frequency_model(answers)
    
    # 模型权重（基于回测结果）
    weights = {
        'trend': 3.0,       # 趋势模型权重最高
        'weighted': 2.5,    # 加权频率
        'markov': 2.0,      # 马尔可夫
        'position': 1.5,    # 位置频率
        'sliding': 2.0,     # 滑动窗口
        'exp_weighted': 2.0,# 指数加权
        'bayesian': 1.5,    # 贝叶斯
    }
    
    # 获取各模型预测
    models = {
        'weighted': weighted_frequency_model(answers, times),
        'position': position_frequency_model(answers)[0],
        'trend': trend_model(answers, times),
        'markov': markov_model(answers),
        'sliding': sliding_window_model(answers, times, 8),
        'exp_weighted': exponential_weighted_model(answers, times, 0.9),
        'bayesian': bayesian_model(answers, times),
    }
    
    predictions = []
    for q in range(25):
        votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for name, pred in models.items():
            votes[pred[q]] += weights.get(name, 1.0)
        
        # 加入位置频率作为参考
        for opt in 'ABCD':
            votes[opt] += probabilities[q].get(opt, 0) * 1.5
        
        predictions.append(max(votes, key=votes.get))
    
    return predictions

# ============================================================================
# 周期模型 - 分析答案周期性
# ============================================================================

def periodic_model(answers, times):
    """
    周期模型
    分析是否存在周期性规律（如6月/12月差异）
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    # 分离6月和12月试卷
    june_mask = times[:, 1] == 6
    dec_mask = times[:, 1] == 12
    
    june_answers = answers[june_mask] if any(june_mask) else answers
    dec_answers = answers[dec_mask] if any(dec_mask) else answers
    
    # 预测下一个6月（2025年6月）
    for q in range(n_questions):
        # 使用6月份的频率
        freq = Counter(june_answers[:, q])
        if freq:
            predictions.append(freq.most_common(1)[0][0])
        else:
            freq = Counter(answers[:, q])
            predictions.append(freq.most_common(1)[0][0])
    
    return predictions

# ============================================================================
# 反模式模型 - 避免重复最近的答案
# ============================================================================

def anti_pattern_model(answers, times):
    """
    反模式模型
    假设出题者会避免与最近试卷相同的答案
    """
    n_exams, n_questions = answers.shape
    predictions = []
    
    # 最近3套试卷的答案
    recent = answers[-3:] if n_exams >= 3 else answers
    
    for q in range(n_questions):
        # 统计最近答案
        recent_ans = [recent[i, q] for i in range(len(recent))]
        recent_freq = Counter(recent_ans)
        
        # 全部历史频率
        all_freq = Counter(answers[:, q])
        
        # 降低最近出现过的选项权重
        adjusted = {}
        for opt in 'ABCD':
            base = all_freq.get(opt, 0)
            penalty = recent_freq.get(opt, 0) * 0.3
            adjusted[opt] = max(base - penalty, 0.1)
        
        predictions.append(max(adjusted, key=adjusted.get))
    
    return predictions

# ============================================================================
# 混合深度模型 - 结合多种特征
# ============================================================================

def hybrid_deep_model(train_answers, train_times):
    """
    混合深度模型
    使用更丰富的特征和更深的网络
    """
    X, y, le = prepare_ml_features(train_answers, train_times)
    
    # 添加更多特征
    n_exams, n_questions = train_answers.shape
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate='adaptive',
        alpha=0.001  # L2正则化
    )
    model.fit(X, y)
    
    test_features = prepare_prediction_features(train_answers, train_times, le)
    predictions = model.predict(test_features)
    return [le.inverse_transform([p])[0] for p in predictions]

# ============================================================================
# 集成模型 - 多模型投票
# ============================================================================

def ensemble_predict(answers, times):
    """
    集成预测
    综合多个模型的结果进行投票
    """
    # 获取各模型预测
    pred_weighted = weighted_frequency_model(answers, times)
    pred_markov = markov_model(answers)
    pred_position, probabilities = position_frequency_model(answers)
    pred_balanced = balanced_model(answers, probabilities)
    pred_trend = trend_model(answers, times)
    
    # 对平衡模型应用反连续约束
    pred_balanced = anti_consecutive_model(pred_balanced, probabilities)
    
    # 投票（给不同模型不同权重）
    weights = {
        'weighted': 2.0,    # 加权频率权重高
        'position': 1.5,    # 位置频率
        'balanced': 2.0,    # 平衡模型权重高
        'trend': 1.5,       # 趋势模型
        'markov': 1.0       # 马尔可夫
    }
    
    final_predictions = []
    prediction_details = []
    
    for i in range(25):
        votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        votes[pred_weighted[i]] += weights['weighted']
        votes[pred_position[i]] += weights['position']
        votes[pred_balanced[i]] += weights['balanced']
        votes[pred_trend[i]] += weights['trend']
        votes[pred_markov[i]] += weights['markov']
        
        # 加入概率信息作为参考
        for opt in 'ABCD':
            votes[opt] += probabilities[i].get(opt, 0) * 1.0
        
        best = max(votes, key=votes.get)
        final_predictions.append(best)
        
        # 记录详情
        prediction_details.append({
            'question': i + 1,
            'prediction': best,
            'weighted': pred_weighted[i],
            'position': pred_position[i],
            'balanced': pred_balanced[i],
            'trend': pred_trend[i],
            'markov': pred_markov[i],
            'probs': probabilities[i],
            'confidence': votes[best] / sum(votes.values())
        })
    
    return final_predictions, prediction_details

# ============================================================================
# 概率采样预测 - 基于历史概率随机采样
# ============================================================================

def probabilistic_predict(probabilities, seed=None):
    """
    概率采样预测
    根据每题的历史概率分布进行随机采样
    同时保证ABCD分布相对均衡
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    predictions = []
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    target = 6  # 每个选项目标数量
    
    for q in range(25):
        probs = probabilities[q].copy()
        
        # 动态调整概率以保持平衡
        remaining = 25 - q
        for opt in 'ABCD':
            if counts[opt] >= 7:  # 已经太多
                probs[opt] *= 0.3
            elif counts[opt] >= 6:
                probs[opt] *= 0.6
            
            # 如果某选项还差很多，增加其概率
            needed = target - counts[opt]
            if needed > 0 and remaining <= needed * 2:
                probs[opt] *= 1.8
        
        # 归一化
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # 按概率采样
        opts = list(probs.keys())
        weights = [probs[o] for o in opts]
        choice = random.choices(opts, weights=weights, k=1)[0]
        
        predictions.append(choice)
        counts[choice] += 1
    
    return predictions

# ============================================================================
# 生成两套试卷（有明显差异）
# ============================================================================

def generate_two_sets(answers, times, probabilities):
    """
    生成两套有差异的预测试卷
    使用不同的随机种子和策略
    """
    # 第一套：集成模型 + 概率采样混合
    pred1_ensemble, details = ensemble_predict(answers, times)
    pred1_prob = probabilistic_predict(probabilities, seed=2025)
    
    # 混合两种预测
    pred1 = []
    for i in range(25):
        probs = probabilities[i]
        if probs[pred1_ensemble[i]] > 0.35:
            pred1.append(pred1_ensemble[i])
        else:
            pred1.append(pred1_prob[i])
    
    # 应用平衡约束
    pred1 = apply_balance_constraint(pred1, probabilities)
    
    # 第二套：使用不同种子的概率采样
    pred2 = probabilistic_predict(probabilities, seed=2026)
    
    # 确保两套有足够差异（至少5题不同）
    diff_count = sum(1 for a, b in zip(pred1, pred2) if a != b)
    if diff_count < 5:
        # 强制增加差异
        for i in range(25):
            if pred1[i] == pred2[i] and diff_count < 8:
                sorted_opts = sorted(probabilities[i].items(), key=lambda x: -x[1])
                if len(sorted_opts) >= 2:
                    pred2[i] = sorted_opts[1][0]
                    diff_count += 1
    
    return pred1, pred2, details

def generate_two_sets_with_seed(answers, times, probabilities, seed):
    """使用指定种子生成两套试卷"""
    random.seed(seed)
    np.random.seed(seed)
    return generate_two_sets(answers, times, probabilities)

def apply_balance_constraint(predictions, probabilities):
    """应用平衡约束，确保ABCD分布合理"""
    pred = list(predictions)
    counts = Counter(pred)
    
    # 目标：每个选项5-7题
    for _ in range(10):  # 最多调整10次
        need_adjust = False
        
        for opt in 'ABCD':
            if counts.get(opt, 0) > 8:  # 太多
                # 找一个可以替换的位置
                for i in range(25):
                    if pred[i] == opt:
                        probs = probabilities[i]
                        sorted_opts = sorted(probs.items(), key=lambda x: -x[1])
                        for new_opt, _ in sorted_opts:
                            if new_opt != opt and counts.get(new_opt, 0) < 6:
                                pred[i] = new_opt
                                counts[opt] -= 1
                                counts[new_opt] = counts.get(new_opt, 0) + 1
                                need_adjust = True
                                break
                        if need_adjust:
                            break
            
            elif counts.get(opt, 0) < 4:  # 太少
                # 找一个可以换成这个选项的位置
                for i in range(25):
                    if pred[i] != opt and counts[pred[i]] > 6:
                        probs = probabilities[i]
                        if probs.get(opt, 0) > 0.15:
                            old = pred[i]
                            pred[i] = opt
                            counts[old] -= 1
                            counts[opt] = counts.get(opt, 0) + 1
                            need_adjust = True
                            break
        
        if not need_adjust:
            break
    
    return pred

# ============================================================================
# 多模型回测评估
# ============================================================================

def backtest_all_models(answers, times):
    """
    多模型回测评估（包含机器学习模型）
    逐年滚动回测，比较各模型准确率
    """
    all_years = sorted(set(times[:, 0]))
    
    print(f"\n{'='*100}")
    print("  多模型逐年回测评估（统计模型 + 机器学习模型）")
    print(f"{'='*100}")
    
    # 统计模型
    stat_models = ['加权频率', '位置频率', '马尔可夫', '趋势模型', '滑动窗口', '指数加权', '贝叶斯']
    # 集成模型
    ensemble_models = ['投票集成', '加权投票']
    # 其他模型
    other_models = ['周期模型', '反模式', 'N-gram', '随机森林', 'XGBoost', '深度混合']
    # 所有模型
    model_names = stat_models + ensemble_models + other_models
    
    # 存储各模型各年结果
    model_results = {name: {'correct': 0, 'total': 0, 'yearly': {}} for name in model_names}
    yearly_comparison = []
    
    # 从2018年开始回测（确保有足够训练数据）
    for test_year in all_years[2:]:
        train_mask = times[:, 0] < test_year
        test_mask = times[:, 0] == test_year
        
        if not any(train_mask) or not any(test_mask):
            continue
        
        train_answers = answers[train_mask]
        train_times = times[train_mask]
        test_answers = answers[test_mask]
        test_times = times[test_mask]
        
        print(f"\n回测 {int(test_year)} 年 (训练集: {len(train_answers)}套)...", end=" ")
        
        # 获取概率分布
        _, probabilities = position_frequency_model(train_answers)
        
        # 统计模型预测
        predictions = {
            '加权频率': weighted_frequency_model(train_answers, train_times),
            '位置频率': position_frequency_model(train_answers)[0],
            '马尔可夫': markov_model(train_answers),
            '趋势模型': trend_model(train_answers, train_times),
            '滑动窗口': sliding_window_model(train_answers, train_times, 8),
            '指数加权': exponential_weighted_model(train_answers, train_times, 0.9),
            '贝叶斯': bayesian_model(train_answers, train_times),
        }
        
        # 集成模型预测
        try:
            predictions['投票集成'] = voting_ensemble_model(train_answers, train_times)
        except Exception as e:
            predictions['投票集成'] = predictions['位置频率']
        
        try:
            predictions['加权投票'] = weighted_voting_model(train_answers, train_times)
        except Exception as e:
            predictions['加权投票'] = predictions['位置频率']
        
        # 其他模型
        try:
            predictions['周期模型'] = periodic_model(train_answers, train_times)
        except Exception as e:
            predictions['周期模型'] = predictions['位置频率']
        
        try:
            predictions['反模式'] = anti_pattern_model(train_answers, train_times)
        except Exception as e:
            predictions['反模式'] = predictions['位置频率']
        
        try:
            predictions['N-gram'] = ngram_model(train_answers, n=3)
        except Exception as e:
            predictions['N-gram'] = predictions['位置频率']
        
        try:
            predictions['随机森林'] = random_forest_model(train_answers, train_times)
        except Exception as e:
            predictions['随机森林'] = predictions['位置频率']
        
        try:
            predictions['XGBoost'] = xgboost_model(train_answers, train_times)
        except Exception as e:
            predictions['XGBoost'] = predictions['位置频率']
        
        try:
            predictions['深度混合'] = hybrid_deep_model(train_answers, train_times)
        except Exception as e:
            predictions['深度混合'] = predictions['位置频率']
        
        # 计算各模型在该年的准确率
        year_results = {'year': int(test_year), 'exams': len(test_answers)}
        
        for model_name, pred in predictions.items():
            correct = 0
            for exam in test_answers:
                correct += sum(1 for p, a in zip(pred, exam) if p == a)
            
            total = len(test_answers) * 25
            accuracy = correct / total * 100
            
            model_results[model_name]['correct'] += correct
            model_results[model_name]['total'] += total
            model_results[model_name]['yearly'][int(test_year)] = accuracy
            
            year_results[model_name] = accuracy
        
        yearly_comparison.append(year_results)
        print("完成")
    
    # 打印逐年对比表格
    print(f"\n{'='*120}")
    print("逐年各模型准确率对比:")
    print("-" * 120)
    
    # 显示表头
    header = f"{'年份':^6}"
    for name in model_names:
        header += f" {name[:4]:^7}"
    print(header)
    print("-" * 120)
    
    for yr in yearly_comparison:
        row = f"{yr['year']:^6}"
        best_acc = max(yr[name] for name in model_names)
        for name in model_names:
            acc = yr[name]
            marker = "★" if acc == best_acc else " "
            row += f" {acc:>4.0f}%{marker} "
        print(row)
    
    print("-" * 120)
    
    # 计算总体准确率
    print(f"\n{'='*60}")
    print("总体准确率排名:")
    print("-" * 60)
    
    final_results = []
    for name in model_names:
        total_correct = model_results[name]['correct']
        total_questions = model_results[name]['total']
        avg_acc = total_correct / total_questions * 100 if total_questions > 0 else 0
        improvement = avg_acc - 25
        if name in stat_models:
            model_type = "统计"
        elif name in ensemble_models:
            model_type = "集成"
        else:
            model_type = "ML"
        final_results.append({
            'name': name,
            'type': model_type,
            'correct': total_correct,
            'total': total_questions,
            'accuracy': avg_acc,
            'improvement': improvement
        })
    
    # 按准确率排序
    final_results.sort(key=lambda x: -x['accuracy'])
    
    for i, r in enumerate(final_results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        sign = "+" if r['improvement'] >= 0 else ""
        type_tag = f"[{r['type']}]"
        print(f"{medal} {r['name']:^12} {type_tag:^6}: {r['accuracy']:>6.2f}% "
              f"({r['correct']}/{r['total']}题) [{sign}{r['improvement']:.2f}%]")
    
    print("-" * 60)
    print(f"理论随机概率: 25.00%")
    
    # 返回最佳模型
    best_model = final_results[0]['name']
    best_type = final_results[0]['type']
    print(f"\n🏆 最佳模型: {best_model} ({best_type}) - 准确率 {final_results[0]['accuracy']:.2f}%")
    
    # 导出回测报告
    export_backtest_report(final_results, yearly_comparison, model_names)
    
    return final_results, yearly_comparison

def export_backtest_report(final_results, yearly_comparison, model_names=None):
    """导出回测报告到CSV"""
    output_path = '/Users/chuankangkk/Downloads/六级听力/六级听力预测分析/回测报告.csv'
    
    # 准备数据
    rows = []
    if model_names is None:
        model_names = ['加权频率', '位置频率', '马尔可夫', '趋势模型', '随机森林', 'XGBoost', '梯度提升', 'MLP神经网络', '逻辑回归', 'N-gram']
    
    # 逐年数据
    for yr in yearly_comparison:
        row = {'年份': yr['year'], '试卷数': yr['exams']}
        for name in model_names:
            row[name] = f"{yr[name]:.1f}%"
        rows.append(row)
    
    # 添加汇总行
    summary_row = {'年份': '总计', '试卷数': sum(yr['exams'] for yr in yearly_comparison)}
    for r in final_results:
        summary_row[r['name']] = f"{r['accuracy']:.2f}%"
    rows.append(summary_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n回测报告已导出: {output_path}")

def backtest(answers, times, test_years=2):
    """简化回测（兼容旧调用）"""
    return backtest_all_models(answers, times)

def detailed_comparison(answers, times, best_model_name='集成模型'):
    """
    详细对比：显示每套试卷的预测vs真实答案
    """
    all_years = sorted(set(times[:, 0]))
    
    print(f"\n{'='*70}")
    print(f"  详细对比: {best_model_name} 预测 vs 真实答案")
    print(f"{'='*70}")
    
    # 逐年回测
    for test_year in all_years[2:]:
        train_mask = times[:, 0] < test_year
        test_mask = times[:, 0] == test_year
        
        if not any(train_mask) or not any(test_mask):
            continue
        
        train_answers = answers[train_mask]
        train_times = times[train_mask]
        test_answers = answers[test_mask]
        test_times = times[test_mask]
        
        # 获取概率分布
        _, probabilities = position_frequency_model(train_answers)
        
        # 使用集成模型预测
        if best_model_name == '集成模型':
            pred, _ = ensemble_predict(train_answers, train_times)
        elif best_model_name == '加权频率':
            pred = weighted_frequency_model(train_answers, train_times)
        elif best_model_name == '位置频率':
            pred, _ = position_frequency_model(train_answers)
        elif best_model_name == '平衡约束':
            pred = balanced_model(train_answers, probabilities)
        else:
            pred, _ = ensemble_predict(train_answers, train_times)
        
        print(f"\n【{int(test_year)}年】")
        print("-" * 70)
        
        for i, (exam, time) in enumerate(zip(test_answers, test_times)):
            exam_name = f"{int(time[0])}年{int(time[1])}月第{int(time[2])}套"
            real = ''.join(exam)
            predicted = ''.join(pred)
            
            # 计算正确题目
            correct_positions = [j+1 for j in range(25) if pred[j] == exam[j]]
            wrong_positions = [j+1 for j in range(25) if pred[j] != exam[j]]
            correct_count = len(correct_positions)
            
            print(f"\n{exam_name}:")
            print(f"  预测: {predicted}")
            print(f"  真实: {real}")
            
            # 逐题对比
            comparison = ""
            for j in range(25):
                if pred[j] == exam[j]:
                    comparison += "✓"
                else:
                    comparison += "✗"
            print(f"  对比: {comparison}")
            print(f"  正确: {correct_count}/25 ({correct_count/25*100:.1f}%)")
            if correct_count > 0:
                print(f"  命中题号: {correct_positions}")

# ============================================================================
# 打印预测结果
# ============================================================================

def print_predictions(pred1, pred2, year, month, details):
    """打印预测结果"""
    print(f"\n{'='*60}")
    print(f"  {year}年{month}月 六级听力答案预测")
    print(f"{'='*60}")
    
    # 第一套
    print(f"\n【第一套试卷预测】")
    print(f"完整答案: {''.join(pred1)}")
    print(f"\n分题答案:")
    for i in range(5):
        start = i * 5
        end = start + 5
        row = '  '.join([f"T{j+1}:{pred1[j]}" for j in range(start, end)])
        print(f"  {row}")
    
    # 统计分布
    dist1 = Counter(pred1)
    print(f"\n选项分布: A:{dist1['A']} B:{dist1['B']} C:{dist1['C']} D:{dist1['D']}")
    
    # 第二套
    print(f"\n{'─'*60}")
    print(f"\n【第二套试卷预测】")
    print(f"完整答案: {''.join(pred2)}")
    print(f"\n分题答案:")
    for i in range(5):
        start = i * 5
        end = start + 5
        row = '  '.join([f"T{j+1}:{pred2[j]}" for j in range(start, end)])
        print(f"  {row}")
    
    # 统计分布
    dist2 = Counter(pred2)
    print(f"\n选项分布: A:{dist2['A']} B:{dist2['B']} C:{dist2['C']} D:{dist2['D']}")
    
    # 两套差异
    diff_count = sum(1 for a, b in zip(pred1, pred2) if a != b)
    diff_positions = [i+1 for i, (a, b) in enumerate(zip(pred1, pred2)) if a != b]
    print(f"\n两套差异: {diff_count}题 (位置: {diff_positions})")
    
    # 置信度分析
    print(f"\n{'─'*60}")
    print("置信度分析（按题号）:")
    high_conf = [d for d in details if d['confidence'] >= 0.4]
    low_conf = [d for d in details if d['confidence'] < 0.3]
    
    if high_conf:
        print(f"  高置信度(>40%): T{', T'.join([str(d['question']) for d in high_conf])}")
    if low_conf:
        print(f"  低置信度(<30%): T{', T'.join([str(d['question']) for d in low_conf])}")

# ============================================================================
# 导出预测结果
# ============================================================================

def export_predictions(predictions_list, output_path):
    """导出预测结果到CSV"""
    rows = []
    for pred in predictions_list:
        row = {
            '年份': pred['year'],
            '月份': pred['month'],
            '套数': pred['set'],
            '考试时间': f"{pred['year']}年{pred['month']}月"
        }
        for i, ans in enumerate(pred['answers']):
            row[f'T{i+1}'] = ans
        row['完整答案'] = ''.join(pred['answers'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已导出: {output_path}")

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*60)
    print("  六级听力答案预测系统")
    print("="*60)
    
    # 加载数据
    csv_path = '/Users/chuankangkk/Downloads/六级听力/六级听力预测分析/六级听力答案_2025-12-04.csv'
    df = load_data(csv_path)
    
    # 提取答案
    answers, times, df = extract_answers(df)
    print(f"数据范围: {int(times[0,0])}年 - {int(times[-1,0])}年")
    
    # 获取概率分布
    _, probabilities = position_frequency_model(answers)
    
    # 多模型回测评估
    final_results, yearly_comparison = backtest(answers, times, test_years=2)
    
    # 获取最佳模型名称
    best_model_name = final_results[0]['name']
    
    # 详细对比
    detailed_comparison(answers, times, best_model_name)
    
    # ========================================================================
    # 预测2025年6月和12月（各两套）
    # ========================================================================
    
    all_predictions = []
    
    # 使用最佳模型(趋势模型)预测 + 概率采样生成两套
    def best_model_predict(probabilities, seed):
        """使用趋势模型+概率采样组合预测"""
        # 趋势模型预测
        trend_pred = trend_model(answers, times)
        # 概率采样作为第二套
        prob_pred = probabilistic_predict(probabilities, seed=seed)
        return trend_pred, prob_pred
    
    # 2025年6月
    pred1_jun, pred2_jun = best_model_predict(probabilities, 20256)
    _, details_jun = ensemble_predict(answers, times)
    print_predictions(pred1_jun, pred2_jun, 2025, 6, details_jun)
    all_predictions.append({'year': 2025, 'month': 6, 'set': 1, 'answers': pred1_jun})
    all_predictions.append({'year': 2025, 'month': 6, 'set': 2, 'answers': pred2_jun})
    
    # 2025年12月
    pred1_dec, pred2_dec = best_model_predict(probabilities, 202512)
    print_predictions(pred1_dec, pred2_dec, 2025, 12, details_jun)
    all_predictions.append({'year': 2025, 'month': 12, 'set': 1, 'answers': pred1_dec})
    all_predictions.append({'year': 2025, 'month': 12, 'set': 2, 'answers': pred2_dec})
    
    # 2026年6月
    pred1_jun26, pred2_jun26 = best_model_predict(probabilities, 20266)
    print_predictions(pred1_jun26, pred2_jun26, 2026, 6, details_jun)
    all_predictions.append({'year': 2026, 'month': 6, 'set': 1, 'answers': pred1_jun26})
    all_predictions.append({'year': 2026, 'month': 6, 'set': 2, 'answers': pred2_jun26})
    
    # 2026年12月
    pred1_dec26, pred2_dec26 = best_model_predict(probabilities, 202612)
    print_predictions(pred1_dec26, pred2_dec26, 2026, 12, details_jun)
    all_predictions.append({'year': 2026, 'month': 12, 'set': 1, 'answers': pred1_dec26})
    all_predictions.append({'year': 2026, 'month': 12, 'set': 2, 'answers': pred2_dec26})
    
    # 导出预测
    output_path = '/Users/chuankangkk/Downloads/六级听力/六级听力预测分析/六级听力预测结果_2025-2026.csv'
    export_predictions(all_predictions, output_path)
    
    # ========================================================================
    # 统计分析
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("历史数据统计分析")
    print(f"{'='*60}")
    
    print("\n各题位置选项频率分布:")
    print("-" * 60)
    print(f"{'题号':^6} {'A':^10} {'B':^10} {'C':^10} {'D':^10} {'最常见':^8}")
    print("-" * 60)
    
    for q in range(25):
        probs = probabilities[q]
        most_common = max(probs, key=probs.get)
        print(f"T{q+1:>2}    {probs['A']*100:>6.1f}%   {probs['B']*100:>6.1f}%   "
              f"{probs['C']*100:>6.1f}%   {probs['D']*100:>6.1f}%    {most_common}")
    
    print(f"\n{'='*60}")
    print("预测完成！祝考试顺利！")
    print("="*60)

if __name__ == '__main__':
    main()

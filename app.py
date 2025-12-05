# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify
from collections import Counter
import pandas as pd
import numpy as np
import os
from six_level_prediction_model import (
    load_data, extract_answers, position_frequency_model,
    weighted_frequency_model, markov_model, trend_model,
    sliding_window_model, exponential_weighted_model, bayesian_model,
    voting_ensemble_model, weighted_voting_model,
    periodic_model, anti_pattern_model, ngram_model,
    probabilistic_predict, apply_balance_constraint,
    backtest_all_models, ensemble_predict,
    random_forest_model, xgboost_model, gradient_boosting_model,
    mlp_model, logistic_model, hybrid_deep_model
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# 配置
DATA_PATH = os.path.join(os.path.dirname(__file__), '六级听力答案_2025-12-04.csv')

# 全局缓存
cache = {
    'df': None,
    'answers': None,
    'times': None,
    'probabilities': None,
    'backtest_results': None
}

def get_data():
    """获取或加载数据"""
    if cache['df'] is None:
        df = load_data(DATA_PATH)
        answers, times, df_sorted = extract_answers(df)
        _, probabilities = position_frequency_model(answers)
        
        cache['df'] = df_sorted
        cache['answers'] = answers
        cache['times'] = times
        cache['probabilities'] = probabilities
    return cache

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """获取基础统计数据"""
    data = get_data()
    df = data['df']
    probabilities = data['probabilities']
    
    stats = {
        'total_exams': len(df),
        'years_range': f"{int(data['times'][0,0])}-{int(data['times'][-1,0])}",
        'position_freq': []
    }
    
    for i, prob in enumerate(probabilities):
        stats['position_freq'].append({
            'question': i + 1,
            'probs': prob,
            'most_common': max(prob, key=prob.get)
        })
        
    return jsonify(stats)

@app.route('/api/predict')
def get_prediction():
    """获取预测结果"""
    data = get_data()
    answers = data['answers']
    times = data['times']
    probs = data['probabilities']
    
    # 使用最佳策略：趋势模型 + 概率采样
    predictions = []
    
    # 预测目标
    targets = [
        {'year': 2025, 'month': 6},
        {'year': 2025, 'month': 12},
        {'year': 2026, 'month': 6},
        {'year': 2026, 'month': 12}
    ]
    
    for target in targets:
        # 生成第一套 (趋势模型 + 调整)
        trend_pred = trend_model(answers, times)
        
        # 生成第二套 (概率采样)
        seed = int(f"{target['year']}{target['month']}")
        prob_pred = probabilistic_predict(probs, seed=seed)
        
        # 格式化
        predictions.append({
            'title': f"{target['year']}年{target['month']}月 第1套",
            'answers': trend_pred,
            'answers_str': ''.join(trend_pred),
            'analysis': analyze_prediction(trend_pred, probs)
        })
        predictions.append({
            'title': f"{target['year']}年{target['month']}月 第2套",
            'answers': prob_pred,
            'answers_str': ''.join(prob_pred),
            'analysis': analyze_prediction(prob_pred, probs)
        })
        
    return jsonify(predictions)

def analyze_prediction(pred, probs):
    """分析预测结果的置信度"""
    analysis = []
    for i, ans in enumerate(pred):
        confidence = probs[i].get(ans, 0)
        analysis.append({
            'q': i + 1,
            'ans': ans,
            'confidence': confidence,
            'is_high_conf': confidence > 0.35,
            'is_low_conf': confidence < 0.2
        })
    return analysis

@app.route('/api/backtest')
def get_backtest():
    """获取回测数据"""
    data = get_data()
    
    # 如果已有缓存，直接返回
    if cache['backtest_results']:
        return jsonify(cache['backtest_results'])
    
    # 运行回测 (简化版，只跑主要模型以加快速度)
    # 注意：为了前端响应速度，这里最好是预先跑好或者异步
    # 这里演示实时跑几个主要模型
    
    answers = data['answers']
    times = data['times']
    
    # 选择表现最好的几个模型（统计模型 + 机器学习模型）
    models = {
        '趋势模型': trend_model,
        '加权频率': weighted_frequency_model,
        '加权投票': weighted_voting_model,
        '滑动窗口': lambda a, t: sliding_window_model(a, t, 8),
        '马尔可夫': lambda a, t: markov_model(a),
        '贝叶斯': bayesian_model,
        '周期模型': periodic_model,
    }
    
    results = []
    years = sorted(set(times[:, 0]))[2:] # 从第3年开始
    
    model_scores = {name: [] for name in models}
    yearly_scores = []
    
    for year in years:
        train_mask = times[:, 0] < year
        test_mask = times[:, 0] == year
        
        if not any(train_mask) or not any(test_mask):
            continue
            
        train_a = answers[train_mask]
        train_t = times[train_mask]
        test_a = answers[test_mask]
        
        year_data = {'year': int(year)}
        
        for name, func in models.items():
            try:
                pred = func(train_a, train_t)
                correct = sum(sum(1 for p, a in zip(pred, exam) if p == a) for exam in test_a)
                total = len(test_a) * 25
                acc = correct / total * 100
                year_data[name] = round(acc, 1)
                model_scores[name].append(acc)
            except:
                year_data[name] = 0
        
        yearly_scores.append(year_data)
    
    # 总体平均
    summary = []
    for name, scores in model_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        summary.append({'name': name, 'accuracy': round(avg, 2)})
    
    summary.sort(key=lambda x: -x['accuracy'])
    
    response_data = {
        'yearly': yearly_scores,
        'summary': summary
    }
    
    cache['backtest_results'] = response_data
    return jsonify(response_data)

@app.route('/api/history')
def get_history():
    """获取历史数据"""
    data = get_data()
    df = data['df']
    
    history = []
    for _, row in df.iterrows():
        history.append({
            'year': int(row['年份']),
            'month': int(row['月份']),
            'set': int(row['套数']),
            'answers': row['完整答案']
        })
    
    return jsonify(history[::-1]) # 倒序

@app.route('/api/trends')
def get_trends():
    """获取选项趋势数据"""
    data = get_data()
    df = data['df']
    
    # 按年份分组计算ABCD比例
    years = sorted(df['年份'].unique())
    trends = {'years': [int(y) for y in years], 'A': [], 'B': [], 'C': [], 'D': []}
    
    for year in years:
        year_df = df[df['年份'] == year]
        # 拼接该年所有答案
        all_ans = ''.join(year_df['完整答案'].tolist())
        total = len(all_ans)
        if total > 0:
            trends['A'].append(round(all_ans.count('A') / total * 100, 1))
            trends['B'].append(round(all_ans.count('B') / total * 100, 1))
            trends['C'].append(round(all_ans.count('C') / total * 100, 1))
            trends['D'].append(round(all_ans.count('D') / total * 100, 1))
        else:
            for t in ['A','B','C','D']: trends[t].append(0)
            
    return jsonify(trends)

@app.route('/api/yearly_distribution')
def get_yearly_distribution():
    """获取每年详细的答案分布数据"""
    data = get_data()
    df = data['df']
    
    years = sorted(df['年份'].unique())
    result = []
    
    for year in years:
        year_df = df[df['年份'] == year]
        all_ans = ''.join(year_df['完整答案'].tolist())
        total = len(all_ans)
        
        # 计算分布
        dist = {
            'year': int(year),
            'exams': len(year_df),
            'total_questions': total,
            'A': {'count': all_ans.count('A'), 'percent': round(all_ans.count('A') / total * 100, 1) if total > 0 else 0},
            'B': {'count': all_ans.count('B'), 'percent': round(all_ans.count('B') / total * 100, 1) if total > 0 else 0},
            'C': {'count': all_ans.count('C'), 'percent': round(all_ans.count('C') / total * 100, 1) if total > 0 else 0},
            'D': {'count': all_ans.count('D'), 'percent': round(all_ans.count('D') / total * 100, 1) if total > 0 else 0},
        }
        
        # 每套试卷的分布
        dist['papers'] = []
        for _, row in year_df.iterrows():
            ans = row['完整答案']
            dist['papers'].append({
                'month': int(row['月份']),
                'set': int(row['套数']),
                'A': ans.count('A'),
                'B': ans.count('B'),
                'C': ans.count('C'),
                'D': ans.count('D'),
            })
        
        result.append(dist)
    
    return jsonify(result)

@app.route('/api/backtest_detail')
def get_backtest_detail():
    """获取详细的回测对比数据：每年预测 vs 真实答案"""
    data = get_data()
    answers = data['answers']
    times = data['times']
    
    # 全部模型（统计模型 + 集成模型 + 机器学习模型）
    models = {
        # 统计模型
        '趋势模型': trend_model,
        '加权频率': weighted_frequency_model,
        '滑动窗口': lambda a, t: sliding_window_model(a, t, 8),
        '指数加权': exponential_weighted_model,
        '马尔可夫': lambda a, t: markov_model(a),
        '贝叶斯': bayesian_model,
        '周期模型': periodic_model,
        '反模式': anti_pattern_model,
        'N-gram': lambda a, t: ngram_model(a, n=3),
        # 集成模型
        '投票集成': voting_ensemble_model,
        '加权投票': weighted_voting_model,
    }
    
    years = sorted(set(times[:, 0]))[2:]  # 从第3年开始
    detailed_results = []
    
    for year in years:
        train_mask = times[:, 0] < year
        test_mask = times[:, 0] == year
        
        if not any(train_mask) or not any(test_mask):
            continue
        
        train_a = answers[train_mask]
        train_t = times[train_mask]
        test_a = answers[test_mask]
        test_t = times[test_mask]
        
        year_detail = {
            'year': int(year),
            'train_count': int(np.sum(train_mask)),
            'test_count': int(np.sum(test_mask)),
            'models': [],
            'best_model': None,
            'best_accuracy': 0,
            'papers': []
        }
        
        # 每个模型的预测结果
        predictions = {}
        for name, func in models.items():
            try:
                pred = func(train_a, train_t)
                correct_total = 0
                for exam in test_a:
                    correct_total += sum(1 for p, a in zip(pred, exam) if p == a)
                total = len(test_a) * 25
                acc = correct_total / total * 100
                predictions[name] = {'pred': pred, 'accuracy': round(acc, 1)}
                year_detail['models'].append({'name': name, 'accuracy': round(acc, 1)})
                
                if acc > year_detail['best_accuracy']:
                    year_detail['best_accuracy'] = round(acc, 1)
                    year_detail['best_model'] = name
            except Exception as e:
                continue
        
        # 每套试卷的详细对比
        if year_detail['best_model'] and year_detail['best_model'] in predictions:
            best_pred = predictions[year_detail['best_model']]['pred']
            for i, (exam, t) in enumerate(zip(test_a, test_t)):
                paper = {
                    'month': int(t[1]),
                    'set': int(t[2]),
                    'real': ''.join(exam),
                    'predicted': ''.join(best_pred),
                    'correct': sum(1 for p, a in zip(best_pred, exam) if p == a),
                    'match_detail': [1 if p == a else 0 for p, a in zip(best_pred, exam)]
                }
                year_detail['papers'].append(paper)
        
        detailed_results.append(year_detail)
    
    # 计算总体最佳模型
    model_total = {}
    for yr in detailed_results:
        for m in yr['models']:
            if m['name'] not in model_total:
                model_total[m['name']] = []
            model_total[m['name']].append(m['accuracy'])
    
    overall_best = {'name': None, 'avg_accuracy': 0}
    for name, accs in model_total.items():
        avg = sum(accs) / len(accs) if accs else 0
        if avg > overall_best['avg_accuracy']:
            overall_best = {'name': name, 'avg_accuracy': round(avg, 2)}
    
    return jsonify({
        'yearly_detail': detailed_results,
        'overall_best_model': overall_best,
        'model_comparison': [{'name': k, 'avg_accuracy': round(sum(v)/len(v), 2)} 
                            for k, v in model_total.items()]
    })

@app.route('/api/position_analysis')
def get_position_analysis():
    """获取每个题目位置的详细分析"""
    data = get_data()
    answers = data['answers']
    probabilities = data['probabilities']
    
    result = []
    for i in range(25):
        # 计算该位置的历史数据
        pos_answers = answers[:, i]
        freq = Counter(pos_answers)
        
        # 最近5套的趋势
        recent = answers[-5:, i] if len(answers) >= 5 else answers[:, i]
        recent_freq = Counter(recent)
        
        result.append({
            'question': i + 1,
            'probabilities': probabilities[i],
            'most_common': max(probabilities[i], key=probabilities[i].get),
            'historical_counts': dict(freq),
            'recent_trend': dict(recent_freq),
            'is_stable': max(probabilities[i].values()) > 0.35,  # 高置信度
            'is_volatile': max(probabilities[i].values()) < 0.28  # 波动大
        })
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

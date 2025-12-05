/* ================================================================
   六级听力预测系统 - 响应式版本
   桌面端：侧边栏导航 | 移动端：底部TabBar
================================================================ */

// ================================================================
// 全局状态
// ================================================================
const App = {
    page: 'dashboard',
    data: { stats: null, predictions: null, trends: null, backtest: null, history: null }
};

// ================================================================
// 图表配色 (极光风格)
// ================================================================
const COLORS = {
    A: { bg: 'rgba(124, 58, 237, 0.7)', border: '#7c3aed' },
    B: { bg: 'rgba(16, 185, 129, 0.7)', border: '#10b981' },
    C: { bg: 'rgba(249, 115, 22, 0.7)', border: '#f97316' },
    D: { bg: 'rgba(236, 72, 153, 0.7)', border: '#ec4899' }
};

Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ================================================================
// 初始化
// ================================================================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    lucide.createIcons();
    initWelcomeModal();
    initNavigation();
    initMobileMenu();
    initThemeToggle();
    updateTime();
    setInterval(updateTime, 1000);
});

// ================================================================
// 欢迎弹窗 - 条款确认
// ================================================================
function initWelcomeModal() {
    const modal = document.getElementById('welcomeModal');
    const agreeBtn = document.getElementById('agreeBtn');
    
    // 检查是否已同意条款（当天有效）
    const agreed = localStorage.getItem('termsAgreed');
    const today = new Date().toDateString();
    
    if (agreed === today) {
        // 已同意，隐藏弹窗，加载数据
        modal.classList.add('hidden');
        loadDashboard();
    } else {
        // 未同意，显示弹窗
        modal.classList.remove('hidden');
    }
    
    // 同意按钮点击
    agreeBtn.addEventListener('click', () => {
        // 保存同意状态（当天有效）
        localStorage.setItem('termsAgreed', today);
        
        // 关闭弹窗动画
        modal.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => {
            modal.classList.add('hidden');
            modal.style.animation = '';
            // 加载数据
            loadDashboard();
        }, 300);
    });
}

// ================================================================
// 主题管理 (白天 6:00-18:00 / 黑夜)
// ================================================================
function initTheme() {
    // 检查本地存储是否有手动设置的主题
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // 根据时间自动设置
        autoSetTheme();
    }
}

function autoSetTheme() {
    const hour = new Date().getHours();
    // 6:00 - 18:00 为白天模式
    const theme = (hour >= 6 && hour < 18) ? 'light' : 'dark';
    setTheme(theme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
    // 更新 Chart.js 全局颜色
    if (theme === 'light') {
        Chart.defaults.color = '#475569';
        Chart.defaults.borderColor = 'rgba(0, 0, 0, 0.08)';
    } else {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    }
    // 刷新当前页面图表以应用新主题颜色
    refreshCurrentPageCharts();
}

function refreshCurrentPageCharts() {
    // 根据当前页面重新渲染图表
    switch (App.page) {
        case 'dashboard':
            if (App.data.stats) renderDashboard(App.data.stats);
            break;
        case 'trends':
            if (App.data.trends) renderTrends(App.data.trends);
            break;
        case 'backtest':
            if (App.data.backtest) renderBacktestChart(App.data.backtest.yearly);
            break;
        case 'analysis':
            if (App.data.yearlyDist) {
                const year = document.querySelector('.year-btn.active')?.dataset?.year;
                if (year) selectYear(parseInt(year));
            }
            break;
    }
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('themeIcon');
    if (icon) {
        // 白天显示太阳图标，点击切换到黑夜；黑夜显示月亮图标
        icon.setAttribute('data-lucide', theme === 'light' ? 'moon' : 'sun');
        lucide.createIcons();
    }
}

function initThemeToggle() {
    const btn = document.getElementById('themeBtn');
    if (btn) {
        btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'light' ? 'dark' : 'light';
            setTheme(next);
            localStorage.setItem('theme', next);
        });
    }
}

// ================================================================
// 时间更新
// ================================================================
function updateTime() {
    const now = new Date();
    const h = now.getHours().toString().padStart(2, '0');
    const m = now.getMinutes().toString().padStart(2, '0');
    document.getElementById('currentTime').textContent = `${h}:${m}`;
}

// ================================================================
// 导航初始化 (桌面侧边栏 + 移动底部栏)
// ================================================================
function initNavigation() {
    // 桌面端侧边栏导航
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchPage(link.dataset.page);
        });
    });
    
    // 移动端底部导航
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            switchPage(tab.dataset.page);
        });
    });
}

// ================================================================
// 移动端菜单
// ================================================================
function initMobileMenu() {
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    
    if (menuBtn) {
        menuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('open');
        });
    }
    
    if (overlay) {
        overlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('open');
        });
    }
}

function switchPage(pageName) {
    // 更新桌面侧边栏状态
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });
    
    // 更新移动端底部栏状态
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.page === pageName);
    });
    
    // 切换页面
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${pageName}`);
    });
    
    // 更新标题
    const titles = {
        dashboard: '数据概览',
        prediction: '智能预测',
        analysis: '年度分析',
        trends: '趋势变化',
        practice: '原题练习',
        backtest: '模型回测',
        history: '历史数据'
    };
    document.getElementById('pageTitle').textContent = titles[pageName] || '概览';
    
    // 关闭移动端侧边栏
    document.querySelector('.sidebar')?.classList.remove('open');
    document.getElementById('sidebarOverlay')?.classList.remove('open');
    
    // 加载数据
    App.page = pageName;
    loadPageData(pageName);
    
    // 重新渲染图标
    lucide.createIcons();
}

function loadPageData(page) {
    switch (page) {
        case 'dashboard': loadDashboard(); break;
        case 'prediction': loadPredictions(); break;
        case 'trends': loadTrends(); break;
        case 'backtest': loadBacktestDetail(); break;
        case 'history': loadHistory(); break;
        case 'analysis': loadYearlyAnalysis(); break;
        // practice 页面是外部链接，不需要加载数据
    }
}

// ================================================================
// Dashboard
// ================================================================
async function loadDashboard() {
    if (App.data.stats) {
        renderDashboard(App.data.stats);
        return;
    }
    try {
        const res = await fetch('/api/stats');
        App.data.stats = await res.json();
        renderDashboard(App.data.stats);
        loadBacktestAccuracy();
    } catch (e) { console.error('加载统计失败:', e); }
}

function renderDashboard(stats) {
    document.getElementById('totalExams').textContent = stats.total_exams;
    document.getElementById('yearsRange').textContent = stats.years_range;
    
    // 计算总体分布
    const dist = { A: 0, B: 0, C: 0, D: 0 };
    stats.position_freq.forEach(p => {
        Object.keys(dist).forEach(k => dist[k] += p.probs[k] || 0);
    });
    
    renderPieChart(dist);
    renderBarChart(stats.position_freq);
}

function renderPieChart(dist) {
    const ctx = document.getElementById('optionDistChart');
    if (!ctx) return;
    if (window.pieChart && typeof window.pieChart.destroy === 'function') window.pieChart.destroy();
    
    const total = Object.values(dist).reduce((a, b) => a + b, 0);
    const data = Object.keys(dist).map(k => (dist[k] / total * 100).toFixed(1));
    
    window.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['A', 'B', 'C', 'D'],
            datasets: [{
                data,
                backgroundColor: [COLORS.A.bg, COLORS.B.bg, COLORS.C.bg, COLORS.D.bg],
                borderColor: [COLORS.A.border, COLORS.B.border, COLORS.C.border, COLORS.D.border],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, cutout: '60%',
            plugins: {
                legend: { position: 'right', labels: { usePointStyle: true, padding: 15, font: { size: 12 } } }
            }
        }
    });
}

function renderBarChart(posFreq) {
    const ctx = document.getElementById('positionFreqChart');
    if (!ctx) return;
    if (window.barChart && typeof window.barChart.destroy === 'function') window.barChart.destroy();
    
    const labels = posFreq.map((_, i) => `Q${i + 1}`);
    const datasets = ['A', 'B', 'C', 'D'].map(opt => ({
        label: opt,
        data: posFreq.map(p => ((p.probs[opt] || 0) * 100).toFixed(1)),
        backgroundColor: COLORS[opt].bg,
        borderColor: COLORS[opt].border,
        borderWidth: 1, borderRadius: 4
    }));
    
    window.barChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: { stacked: true, max: 100, ticks: { callback: v => v + '%' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function loadBacktestAccuracy() {
    try {
        const res = await fetch('/api/backtest');
        const data = await res.json();
        if (data.summary?.length > 0) {
            document.getElementById('bestAccuracy').textContent = data.summary[0].accuracy.toFixed(1) + '%';
        }
    } catch (e) {}
}

// ================================================================
// 预测结果
// ================================================================
async function loadPredictions() {
    const container = document.getElementById('predictionsContainer');
    if (App.data.predictions) {
        renderPredictions(App.data.predictions);
        return;
    }
    
    container.innerHTML = '<div class="loading-box"><div class="spinner"></div><p>正在计算预测...</p></div>';
    
    try {
        const res = await fetch('/api/predict');
        App.data.predictions = await res.json();
        renderPredictions(App.data.predictions);
    } catch (e) {
        container.innerHTML = '<div class="loading-box"><p>加载失败，请重试</p></div>';
    }
}

function renderPredictions(preds) {
    const container = document.getElementById('predictionsContainer');
    container.innerHTML = preds.map((pred, idx) => {
        const highConf = pred.analysis.filter(a => a.is_high_conf).length;
        const avgConf = (pred.analysis.reduce((s, a) => s + a.confidence, 0) / 25 * 100).toFixed(1);
        
        const cells = pred.answers.map((ans, i) => {
            const cls = ans.toLowerCase();
            return `<div class="answer-cell ${cls}" title="Q${i+1}">${ans}</div>`;
        }).join('');
        
        return `
            <div class="prediction-card">
                <div class="pred-title">${pred.title}</div>
                <div class="answers-grid">${cells}</div>
                <div class="pred-stats">
                    <span>平均置信度: ${avgConf}%</span>
                    <span>高置信: ${highConf}/25</span>
                </div>
                <div class="pred-actions">
                    <button class="btn-copy" onclick="copyAnswer('${pred.answers_str}', this)">
                        <i data-lucide="copy"></i> 复制答案
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    // 刷新预测按钮
    document.getElementById('refreshPrediction').onclick = () => {
        App.data.predictions = null;
        loadPredictions();
    };
    
    lucide.createIcons();
}

function copyAnswer(answerStr, btn) {
    navigator.clipboard.writeText(answerStr).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i data-lucide="check"></i> 已复制';
        btn.classList.add('copied');
        lucide.createIcons();
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('copied');
            lucide.createIcons();
        }, 2000);
    }).catch(err => {
        console.error('复制失败:', err);
    });
}

// ================================================================
// 趋势分析
// ================================================================
async function loadTrends() {
    if (App.data.trends) {
        renderTrends(App.data.trends);
        return;
    }
    try {
        const res = await fetch('/api/trends');
        App.data.trends = await res.json();
        renderTrends(App.data.trends);
    } catch (e) { console.error('趋势加载失败:', e); }
}

function renderTrends(trends) {
    const ctx = document.getElementById('trendsLineChart');
    if (!ctx) return;
    if (window.lineChart && typeof window.lineChart.destroy === 'function') window.lineChart.destroy();
    
    const datasets = ['A', 'B', 'C', 'D'].map(opt => ({
        label: opt,
        data: trends[opt],
        borderColor: COLORS[opt].border,
        backgroundColor: COLORS[opt].bg.replace('0.7', '0.1'),
        borderWidth: 3, fill: true, tension: 0.4,
        pointRadius: 4, pointBackgroundColor: COLORS[opt].border
    }));
    
    window.lineChart = new Chart(ctx, {
        type: 'line',
        data: { labels: trends.years, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: { y: { min: 15, max: 35, ticks: { callback: v => v + '%' } } },
            plugins: { legend: { display: false } }
        }
    });
    
    // 洞察
    const insights = ['A', 'B', 'C', 'D'].map(opt => {
        const data = trends[opt];
        const avg = (data.reduce((a, b) => a + b, 0) / data.length).toFixed(1);
        const latest = data[data.length - 1];
        const trend = latest > data[data.length - 2] ? '↑' : (latest < data[data.length - 2] ? '↓' : '→');
        return { opt, latest, trend };
    });
    
    document.getElementById('trendsInsights').innerHTML = insights.map(i => `
        <div class="insight-item">
            <div class="insight-label">选项 ${i.opt}</div>
            <div class="insight-value ${i.opt.toLowerCase()}">${i.latest}% ${i.trend}</div>
        </div>
    `).join('');
}

// ================================================================
// 模型回测
// ================================================================
async function loadBacktest() {
    const ranking = document.getElementById('modelRanking');
    if (App.data.backtest) {
        renderBacktest(App.data.backtest);
        return;
    }
    
    ranking.innerHTML = '<div class="loading-box"><div class="spinner"></div><p>正在回测...</p></div>';
    
    try {
        const res = await fetch('/api/backtest');
        App.data.backtest = await res.json();
        renderBacktest(App.data.backtest);
    } catch (e) { console.error('回测失败:', e); }
}

function renderBacktest(data) {
    const ranking = document.getElementById('modelRanking');
    ranking.innerHTML = data.summary.map((m, i) => {
        let badge = 'normal';
        if (i === 0) badge = 'gold';
        else if (i === 1) badge = 'silver';
        else if (i === 2) badge = 'bronze';
        
        return `
            <div class="rank-item">
                <div class="rank-badge ${badge}">${i + 1}</div>
                <div class="rank-info">
                    <div class="rank-name">${m.name}</div>
                    <div class="rank-desc">历史数据验证</div>
                </div>
                <div class="rank-score">${m.accuracy.toFixed(1)}%</div>
            </div>
        `;
    }).join('');
    
    // 图表
    renderBacktestChart(data.yearly);
}

function renderBacktestChart(yearly) {
    const ctx = document.getElementById('backtestChart');
    if (!ctx) return;
    if (window.btChart && typeof window.btChart.destroy === 'function') window.btChart.destroy();
    
    const labels = yearly.map(y => y.year);
    const modelNames = Object.keys(yearly[0]).filter(k => k !== 'year');
    const colors = [COLORS.A, COLORS.B, COLORS.C, COLORS.D, { bg: 'rgba(6, 182, 212, 0.7)', border: '#06b6d4' }];
    
    const datasets = modelNames.map((name, i) => ({
        label: name,
        data: yearly.map(y => y[name] || 0),
        backgroundColor: colors[i % colors.length].bg,
        borderColor: colors[i % colors.length].border,
        borderWidth: 2, borderRadius: 6
    }));
    
    window.btChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 50, ticks: { callback: v => v + '%' } } },
            plugins: { legend: { position: 'top', labels: { usePointStyle: true, padding: 10, font: { size: 11 } } } }
        }
    });
}

// ================================================================
// 历史数据
// ================================================================
async function loadHistory() {
    if (App.data.history) {
        renderHistory(App.data.history);
        return;
    }
    try {
        const res = await fetch('/api/history');
        App.data.history = await res.json();
        renderHistory(App.data.history);
    } catch (e) { console.error('历史加载失败:', e); }
}

function renderHistory(history) {
    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = history.map(h => `
        <tr>
            <td>${h.year}</td>
            <td>${h.month}月</td>
            <td>第${h.set}套</td>
            <td><span class="answer-str">${h.answers}</span></td>
        </tr>
    `).join('');
    
    // 搜索
    document.getElementById('historySearch').oninput = (e) => {
        const kw = e.target.value.trim();
        const filtered = kw ? history.filter(h => String(h.year).includes(kw)) : history;
        renderHistory(filtered);
    };
}

// ================================================================
// 详细回测 (新版)
// ================================================================
async function loadBacktestDetail() {
    try {
        // 同时获取简单回测和详细回测数据
        const [btRes, detailRes] = await Promise.all([
            fetch('/api/backtest'),
            fetch('/api/backtest_detail')
        ]);
        const btData = await btRes.json();
        const detailData = await detailRes.json();
        
        // 渲染最佳模型卡片
        if (detailData.overall_best_model) {
            document.getElementById('bestModelName').textContent = detailData.overall_best_model.name;
            document.getElementById('bestModelAcc').textContent = detailData.overall_best_model.avg_accuracy;
        }
        
        // 渲染排行榜
        renderModelRanking(btData.summary);
        
        // 渲染年度图表
        renderBacktestChart(btData.yearly);
        
        // 渲染详细回测对比
        renderYearlyBacktest(detailData.yearly_detail);
        
    } catch (e) {
        console.error('回测加载失败:', e);
        loadBacktest(); // 降级到旧版
    }
}

function renderModelRanking(summary) {
    const ranking = document.getElementById('modelRanking');
    ranking.innerHTML = summary.map((m, i) => {
        let badge = 'normal';
        if (i === 0) badge = 'gold';
        else if (i === 1) badge = 'silver';
        else if (i === 2) badge = 'bronze';
        
        return `
            <div class="rank-item">
                <div class="rank-badge ${badge}">${i + 1}</div>
                <div class="rank-info">
                    <div class="rank-name">${m.name}</div>
                    <div class="rank-desc">历史回测验证</div>
                </div>
                <div class="rank-score">${m.accuracy.toFixed(1)}%</div>
            </div>
        `;
    }).join('');
}

function renderYearlyBacktest(yearlyDetail) {
    const container = document.getElementById('yearlyBacktest');
    
    container.innerHTML = yearlyDetail.map(yr => {
        const papersHtml = yr.papers.map(p => {
            const matchHtml = p.match_detail.map((m, i) => 
                `<div class="match-dot ${m ? 'correct' : 'wrong'}">${i+1}</div>`
            ).join('');
            
            return `
                <div class="paper-compare">
                    <div class="paper-compare-header">
                        <span class="paper-name">${p.month}月 第${p.set}套</span>
                        <span class="paper-score">${p.correct}/25 正确</span>
                    </div>
                    <div class="compare-row">
                        <span class="label">预测:</span>
                        <span class="answers">${p.predicted}</span>
                    </div>
                    <div class="compare-row">
                        <span class="label">真实:</span>
                        <span class="answers">${p.real}</span>
                    </div>
                    <div class="compare-row">
                        <span class="label">对比:</span>
                        <div class="match-indicators">${matchHtml}</div>
                    </div>
                </div>
            `;
        }).join('');
        
        return `
            <div class="backtest-year-item">
                <div class="backtest-year-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <div class="backtest-year-title">
                        <span class="year">${yr.year}年</span>
                        <span class="best-tag">${yr.best_model}</span>
                    </div>
                    <span class="backtest-year-acc">${yr.best_accuracy}%</span>
                </div>
                <div class="backtest-year-body">${papersHtml}</div>
            </div>
        `;
    }).join('');
}

// ================================================================
// 年度分析页面
// ================================================================
async function loadYearlyAnalysis() {
    try {
        const [distRes, posRes] = await Promise.all([
            fetch('/api/yearly_distribution'),
            fetch('/api/position_analysis')
        ]);
        const distData = await distRes.json();
        const posData = await posRes.json();
        
        App.data.yearlyDist = distData;
        App.data.positionAnalysis = posData;
        
        // 渲染年份选择器
        renderYearSelector(distData);
        
        // 默认选中最后一年
        if (distData.length > 0) {
            selectYear(distData[distData.length - 1].year);
        }
        
        // 渲染位置热力图
        renderPositionHeatmap(posData);
        
    } catch (e) {
        console.error('年度分析加载失败:', e);
    }
}

function renderYearSelector(distData) {
    const container = document.getElementById('yearSelector');
    container.innerHTML = distData.map(d => `
        <button class="year-btn" data-year="${d.year}">${d.year}年</button>
    `).join('');
    
    // 绑定点击事件
    container.querySelectorAll('.year-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            selectYear(parseInt(btn.dataset.year));
        });
    });
}

function selectYear(year) {
    // 更新按钮状态
    document.querySelectorAll('.year-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.year) === year);
    });
    
    // 更新标题
    document.getElementById('selectedYearTitle').textContent = `${year}年`;
    
    // 获取该年数据
    const yearData = App.data.yearlyDist.find(d => d.year === year);
    if (!yearData) return;
    
    // 渲染饼图
    renderYearDistChart(yearData);
    
    // 渲染每套试卷分布
    renderPapersGrid(yearData.papers, year);
}

function renderYearDistChart(data) {
    const ctx = document.getElementById('yearDistChart');
    if (!ctx) return;
    if (window.yearChart && typeof window.yearChart.destroy === 'function') window.yearChart.destroy();
    
    window.yearChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['A', 'B', 'C', 'D'],
            datasets: [{
                data: [data.A.percent, data.B.percent, data.C.percent, data.D.percent],
                backgroundColor: [COLORS.A.bg, COLORS.B.bg, COLORS.C.bg, COLORS.D.bg],
                borderColor: [COLORS.A.border, COLORS.B.border, COLORS.C.border, COLORS.D.border],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, cutout: '55%',
            plugins: {
                legend: { position: 'right', labels: { usePointStyle: true, padding: 15 } },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.raw}% (${data[ctx.label].count}题)`
                    }
                }
            }
        }
    });
}

function renderPapersGrid(papers, year) {
    const container = document.getElementById('papersGrid');
    container.innerHTML = papers.map(p => `
        <div class="paper-item">
            <div class="paper-item-title">${year}年${p.month}月 第${p.set}套</div>
            <div class="paper-item-bars">
                <div class="mini-bar a"><div class="mini-bar-value">${p.A}</div><div class="mini-bar-label">A</div></div>
                <div class="mini-bar b"><div class="mini-bar-value">${p.B}</div><div class="mini-bar-label">B</div></div>
                <div class="mini-bar c"><div class="mini-bar-value">${p.C}</div><div class="mini-bar-label">C</div></div>
                <div class="mini-bar d"><div class="mini-bar-value">${p.D}</div><div class="mini-bar-label">D</div></div>
            </div>
        </div>
    `).join('');
}

function renderPositionHeatmap(posData) {
    const container = document.getElementById('positionHeatmap');
    container.innerHTML = posData.map(p => {
        const ans = p.most_common.toLowerCase();
        const prob = (p.probabilities[p.most_common] * 100).toFixed(0);
        let stateClass = '';
        if (p.is_stable) stateClass = 'stable';
        else if (p.is_volatile) stateClass = 'volatile';
        
        return `
            <div class="position-cell ${ans} ${stateClass}" title="Q${p.question}: ${p.most_common} (${prob}%)">
                <div class="q-num">Q${p.question}</div>
                <div class="q-answer">${p.most_common}</div>
                <div class="q-prob">${prob}%</div>
            </div>
        `;
    }).join('');
}

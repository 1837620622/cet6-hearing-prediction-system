// ==UserScript==
// @name         六级听力答案批量提取工具
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  批量提取历年六级听力答案(前25题)，支持导出JSON和Excel格式
// @author       传康kk (Vx:1837620622)
// @match        https://zhenti.burningvocabulary.cn/cet6*
// @grant        GM_download
// @grant        GM_setClipboard
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    // ========================================================================
    // 配置：所有试卷URL列表
    // ========================================================================
    const EXAM_LIST = [
        { url: '/cet6/2024-12/01', title: '2024年12月第1套' },
        { url: '/cet6/2024-12/02', title: '2024年12月第2套' },
        { url: '/cet6/2024-12/03', title: '2024年12月第3套' },
        { url: '/cet6/2024-06/01', title: '2024年6月第1套' },
        { url: '/cet6/2024-06/02', title: '2024年6月第2套' },
        { url: '/cet6/2024-06/03', title: '2024年6月第3套' },
        { url: '/cet6/2023-12/01', title: '2023年12月第1套' },
        { url: '/cet6/2023-12/02', title: '2023年12月第2套' },
        { url: '/cet6/2023-12/03', title: '2023年12月第3套' },
        { url: '/cet6/2023-06/01', title: '2023年6月第1套' },
        { url: '/cet6/2023-06/02', title: '2023年6月第2套' },
        { url: '/cet6/2023-06/03', title: '2023年6月第3套' },
        { url: '/cet6/2022-12/01', title: '2022年12月第1套' },
        { url: '/cet6/2022-12/02', title: '2022年12月第2套' },
        { url: '/cet6/2022-12/03', title: '2022年12月第3套' },
        { url: '/cet6/2022-06/01', title: '2022年6月第1套' },
        { url: '/cet6/2022-06/02', title: '2022年6月第2套' },
        { url: '/cet6/2022-06/03', title: '2022年6月第3套' },
        { url: '/cet6/2021-12/01', title: '2021年12月第1套' },
        { url: '/cet6/2021-12/02', title: '2021年12月第2套' },
        { url: '/cet6/2021-12/03', title: '2021年12月第3套' },
        { url: '/cet6/2021-06/01', title: '2021年6月第1套' },
        { url: '/cet6/2021-06/02', title: '2021年6月第2套' },
        { url: '/cet6/2021-06/03', title: '2021年6月第3套' },
        { url: '/cet6/2020-12/01', title: '2020年12月第1套' },
        { url: '/cet6/2020-12/02', title: '2020年12月第2套' },
        { url: '/cet6/2020-12/03', title: '2020年12月第3套' },
        { url: '/cet6/2020-09/01', title: '2020年9月第1套' },
        { url: '/cet6/2020-09/02', title: '2020年9月第2套' },
        { url: '/cet6/2020-07/01', title: '2020年7月组合卷' },
        { url: '/cet6/2019-12/01', title: '2019年12月第1套' },
        { url: '/cet6/2019-12/02', title: '2019年12月第2套' },
        { url: '/cet6/2019-12/03', title: '2019年12月第3套' },
        { url: '/cet6/2019-06/01', title: '2019年6月第1套' },
        { url: '/cet6/2019-06/02', title: '2019年6月第2套' },
        { url: '/cet6/2019-06/03', title: '2019年6月第3套' },
        { url: '/cet6/2018-12/01', title: '2018年12月第1套' },
        { url: '/cet6/2018-12/02', title: '2018年12月第2套' },
        { url: '/cet6/2018-12/03', title: '2018年12月第3套' },
        { url: '/cet6/2018-06/01', title: '2018年6月第1套' },
        { url: '/cet6/2018-06/02', title: '2018年6月第2套' },
        { url: '/cet6/2018-06/03', title: '2018年6月第3套' },
        { url: '/cet6/2017-12/01', title: '2017年12月第1套' },
        { url: '/cet6/2017-12/02', title: '2017年12月第2套' },
        { url: '/cet6/2017-12/03', title: '2017年12月第3套' },
        { url: '/cet6/2017-06/01', title: '2017年6月第1套' },
        { url: '/cet6/2017-06/02', title: '2017年6月第2套' },
        { url: '/cet6/2017-06/03', title: '2017年6月第3套' },
        { url: '/cet6/2016-12/01', title: '2016年12月第1套' },
        { url: '/cet6/2016-12/02', title: '2016年12月第2套' },
        { url: '/cet6/2016-12/03', title: '2016年12月第3套' },
        { url: '/cet6/2016-06/01', title: '2016年6月第1套' },
        { url: '/cet6/2016-06/02', title: '2016年6月第2套' },
        { url: '/cet6/2016-06/03', title: '2016年6月第3套' }
    ];

    const BASE_URL = 'https://zhenti.burningvocabulary.cn';

    // 存储结果
    let collectedResults = [];
    let isRunning = false;

    // ========================================================================
    // 创建控制面板UI
    // ========================================================================
    function createPanel() {
        const panel = document.createElement('div');
        panel.id = 'cet6-extractor-panel';
        panel.innerHTML = `
            <style>
                #cet6-extractor-panel {
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    width: 320px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    z-index: 999999;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    color: white;
                    overflow: hidden;
                }
                #cet6-extractor-panel .header {
                    padding: 15px;
                    background: rgba(0,0,0,0.2);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: move;
                }
                #cet6-extractor-panel .header h3 {
                    margin: 0;
                    font-size: 16px;
                }
                #cet6-extractor-panel .header .close-btn {
                    background: rgba(255,255,255,0.2);
                    border: none;
                    color: white;
                    width: 28px;
                    height: 28px;
                    border-radius: 50%;
                    cursor: pointer;
                    font-size: 18px;
                }
                #cet6-extractor-panel .content {
                    padding: 15px;
                }
                #cet6-extractor-panel .btn {
                    width: 100%;
                    padding: 12px;
                    margin: 5px 0;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                #cet6-extractor-panel .btn-primary {
                    background: white;
                    color: #667eea;
                }
                #cet6-extractor-panel .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
                }
                #cet6-extractor-panel .btn-secondary {
                    background: rgba(255,255,255,0.2);
                    color: white;
                }
                #cet6-extractor-panel .btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                #cet6-extractor-panel .progress {
                    margin: 10px 0;
                    padding: 10px;
                    background: rgba(0,0,0,0.2);
                    border-radius: 8px;
                    font-size: 13px;
                }
                #cet6-extractor-panel .progress-bar {
                    height: 6px;
                    background: rgba(255,255,255,0.3);
                    border-radius: 3px;
                    margin-top: 8px;
                    overflow: hidden;
                }
                #cet6-extractor-panel .progress-bar-inner {
                    height: 100%;
                    background: white;
                    border-radius: 3px;
                    transition: width 0.3s;
                }
                #cet6-extractor-panel .log {
                    max-height: 150px;
                    overflow-y: auto;
                    font-size: 12px;
                    background: rgba(0,0,0,0.2);
                    border-radius: 8px;
                    padding: 10px;
                    margin-top: 10px;
                }
                #cet6-extractor-panel .log-item {
                    padding: 3px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                #cet6-extractor-panel .log-item:last-child {
                    border-bottom: none;
                }
                #cet6-extractor-panel .footer {
                    padding: 10px 15px;
                    background: rgba(0,0,0,0.2);
                    font-size: 11px;
                    text-align: center;
                    opacity: 0.8;
                }
            </style>
            <div class="header">
                <h3>📚 六级听力答案提取</h3>
                <button class="close-btn" id="close-panel">×</button>
            </div>
            <div class="content">
                <button class="btn btn-primary" id="btn-extract-current">提取当前页答案</button>
                <button class="btn btn-primary" id="btn-extract-all">批量提取所有答案</button>
                <button class="btn btn-secondary" id="btn-stop" style="background:#e74c3c;display:none;">停止提取</button>
                <button class="btn btn-secondary" id="btn-export-json" disabled>导出JSON</button>
                <button class="btn btn-secondary" id="btn-export-csv" disabled>导出CSV</button>
                <div class="progress" id="progress-area" style="display:none;">
                    <div id="progress-text">准备中...</div>
                    <div class="progress-bar">
                        <div class="progress-bar-inner" id="progress-bar" style="width:0%"></div>
                    </div>
                </div>
                <div class="log" id="log-area"></div>
            </div>
            <div class="footer">
                Vx: 1837620622 | 咸鱼/B站: 万能程序员
            </div>
        `;
        document.body.appendChild(panel);

        // 绑定事件
        document.getElementById('close-panel').onclick = () => panel.style.display = 'none';
        document.getElementById('btn-extract-current').onclick = extractCurrentPage;
        document.getElementById('btn-extract-all').onclick = extractAllPages;
        document.getElementById('btn-stop').onclick = stopAutoExtract;
        document.getElementById('btn-export-json').onclick = exportJSON;
        document.getElementById('btn-export-csv').onclick = exportCSV;

        // 检查是否在自动模式
        const autoMode = localStorage.getItem('cet6_extract_mode') === 'auto';
        if (autoMode) {
            document.getElementById('btn-stop').style.display = 'block';
            const idx = parseInt(localStorage.getItem('cet6_extract_index') || '0');
            updateProgress(idx, EXAM_LIST.length, `继续提取中...`);
        }

        // 拖拽功能
        makeDraggable(panel);
    }

    // ========================================================================
    // 拖拽功能
    // ========================================================================
    function makeDraggable(element) {
        const header = element.querySelector('.header');
        let isDragging = false;
        let offsetX, offsetY;

        header.onmousedown = (e) => {
            isDragging = true;
            offsetX = e.clientX - element.offsetLeft;
            offsetY = e.clientY - element.offsetTop;
        };

        document.onmousemove = (e) => {
            if (isDragging) {
                element.style.left = (e.clientX - offsetX) + 'px';
                element.style.top = (e.clientY - offsetY) + 'px';
                element.style.right = 'auto';
            }
        };

        document.onmouseup = () => isDragging = false;
    }

    // ========================================================================
    // 日志功能
    // ========================================================================
    function log(message) {
        const logArea = document.getElementById('log-area');
        const item = document.createElement('div');
        item.className = 'log-item';
        item.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logArea.insertBefore(item, logArea.firstChild);
        console.log('[CET6提取]', message);
    }

    // ========================================================================
    // 更新进度
    // ========================================================================
    function updateProgress(current, total, text) {
        const progressArea = document.getElementById('progress-area');
        const progressText = document.getElementById('progress-text');
        const progressBar = document.getElementById('progress-bar');

        progressArea.style.display = 'block';
        progressText.textContent = text || `进度: ${current}/${total}`;
        progressBar.style.width = `${(current / total) * 100}%`;
    }

    // ========================================================================
    // 从页面提取听力答案(前25题)
    // ========================================================================
    function extractAnswersFromPage() {
        const tables = document.querySelectorAll('table');
        let answers = [];

        // 前5个表格是听力部分（每个表格5题，共25题）
        for (let i = 0; i < 5 && i < tables.length; i++) {
            const cells = tables[i].querySelectorAll('tr:first-child td');
            cells.forEach(cell => {
                const text = cell.textContent.trim();
                if (text && text.length === 1 && 'ABCD'.includes(text)) {
                    answers.push(text);
                }
            });
        }

        return answers.slice(0, 25);
    }

    // ========================================================================
    // 提取当前页面答案
    // ========================================================================
    async function extractCurrentPage() {
        log('开始提取当前页面答案...');

        // 检查是否有答案面板
        let answerPanel = document.querySelector('h4');
        if (!answerPanel || !answerPanel.textContent.includes('参考答案')) {
            // 尝试点击查答案按钮
            const answerBtn = [...document.querySelectorAll('div')].find(d => d.textContent === '查答案');
            if (answerBtn) {
                answerBtn.click();
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        const answers = extractAnswersFromPage();

        if (answers.length === 25) {
            const title = document.title.split('、')[0] || '当前试卷';
            const result = {
                title: title,
                url: window.location.pathname,
                answers: answers,
                answersStr: answers.join('')
            };

            // 检查是否已存在
            const exists = collectedResults.find(r => r.url === result.url);
            if (!exists) {
                collectedResults.push(result);
            }

            log(`✅ 成功提取: ${title}`);
            log(`答案: ${answers.join('')}`);

            updateExportButtons();
        } else {
            log(`❌ 提取失败，只找到 ${answers.length} 题`);
        }
    }

    // ========================================================================
    // 批量提取 - 自动逐页导航模式
    // ========================================================================
    async function extractAllPages() {
        // 检查是否在详情页
        const currentPath = window.location.pathname;
        const isDetailPage = /\/cet6\/\d{4}-\d{2}\/\d{2}/.test(currentPath);

        if (!isDetailPage) {
            // 在列表页，开始自动导航
            log('开始批量提取模式...');
            log('将自动跳转到每个试卷页面提取答案');
            
            // 保存任务状态到localStorage
            localStorage.setItem('cet6_extract_mode', 'auto');
            localStorage.setItem('cet6_extract_index', '0');
            localStorage.setItem('cet6_extract_results', '[]');
            
            // 跳转到第一个试卷
            window.location.href = BASE_URL + EXAM_LIST[0].url;
            return;
        }

        // 在详情页，检查是否是自动模式
        const autoMode = localStorage.getItem('cet6_extract_mode') === 'auto';
        if (!autoMode) {
            log('点击"提取当前页答案"获取本页答案');
            log('或在列表页点击"批量提取"启动自动模式');
            return;
        }

        // 自动模式：提取当前页并跳转下一页
        await autoExtractAndNext();
    }

    // ========================================================================
    // 自动提取并跳转下一页
    // ========================================================================
    async function autoExtractAndNext() {
        const currentIndex = parseInt(localStorage.getItem('cet6_extract_index') || '0');
        const results = JSON.parse(localStorage.getItem('cet6_extract_results') || '[]');
        const exam = EXAM_LIST[currentIndex];

        if (!exam) {
            // 全部完成
            finishAutoExtract(results);
            return;
        }

        updateProgress(currentIndex + 1, EXAM_LIST.length, `正在提取: ${exam.title}`);
        log(`[${currentIndex + 1}/${EXAM_LIST.length}] 提取: ${exam.title}`);

        // 等待页面完全加载
        await new Promise(r => setTimeout(r, 2000));

        // 多次尝试点击查答案按钮
        let retryCount = 0;
        let answers = [];
        
        while (retryCount < 3 && answers.length < 25) {
            // 点击查答案按钮
            const answerBtn = [...document.querySelectorAll('div')].find(d => 
                d.textContent.trim() === '查答案'
            );
            if (answerBtn) {
                answerBtn.click();
                await new Promise(r => setTimeout(r, 2000));
            }

            // 提取答案
            answers = extractAnswersFromPage();
            
            if (answers.length < 25) {
                retryCount++;
                log(`重试 ${retryCount}/3...`);
                await new Promise(r => setTimeout(r, 1500));
            }
        }

        // 解析年份和月份
        const urlMatch = exam.url.match(/\/cet6\/(\d{4})-(\d{2})\/(\d{2})/);
        const year = urlMatch ? parseInt(urlMatch[1]) : 0;
        const month = urlMatch ? parseInt(urlMatch[2]) : 0;
        const set = urlMatch ? parseInt(urlMatch[3]) : 0;

        if (answers.length === 25) {
            results.push({
                title: exam.title,
                year: year,
                month: month,
                set: set,
                url: exam.url,
                answers: answers,
                answersStr: answers.join(''),
                T1: answers[0], T2: answers[1], T3: answers[2], T4: answers[3], T5: answers[4],
                T6: answers[5], T7: answers[6], T8: answers[7], T9: answers[8], T10: answers[9],
                T11: answers[10], T12: answers[11], T13: answers[12], T14: answers[13], T15: answers[14],
                T16: answers[15], T17: answers[16], T18: answers[17], T19: answers[18], T20: answers[19],
                T21: answers[20], T22: answers[21], T23: answers[22], T24: answers[23], T25: answers[24]
            });
            log(`✅ ${exam.title}: ${answers.join('')}`);
        } else {
            log(`⚠️ ${exam.title}: 只找到 ${answers.length} 题，跳过`);
        }

        // 保存进度
        localStorage.setItem('cet6_extract_results', JSON.stringify(results));
        localStorage.setItem('cet6_extract_index', String(currentIndex + 1));

        // 跳转下一页
        if (currentIndex + 1 < EXAM_LIST.length) {
            await new Promise(r => setTimeout(r, 800));
            window.location.href = BASE_URL + EXAM_LIST[currentIndex + 1].url;
        } else {
            finishAutoExtract(results);
        }
    }

    // ========================================================================
    // 完成自动提取
    // ========================================================================
    function finishAutoExtract(results) {
        localStorage.removeItem('cet6_extract_mode');
        localStorage.removeItem('cet6_extract_index');
        localStorage.removeItem('cet6_extract_results');

        collectedResults = results;
        log(`🎉 批量提取完成！共 ${results.length} 套`);
        updateProgress(EXAM_LIST.length, EXAM_LIST.length, '提取完成！');
        updateExportButtons();

        // 自动下载结果
        if (results.length > 0) {
            setTimeout(() => {
                exportJSON();
                exportCSV();
            }, 1000);
        }
    }

    // ========================================================================
    // 页面加载时检查自动模式
    // ========================================================================
    function checkAutoMode() {
        const autoMode = localStorage.getItem('cet6_extract_mode') === 'auto';
        const isDetailPage = /\/cet6\/\d{4}-\d{2}\/\d{2}/.test(window.location.pathname);

        if (autoMode && isDetailPage) {
            // 延迟执行自动提取
            setTimeout(() => {
                autoExtractAndNext();
            }, 2000);
        }
    }

    // ========================================================================
    // 停止自动提取
    // ========================================================================
    function stopAutoExtract() {
        const results = JSON.parse(localStorage.getItem('cet6_extract_results') || '[]');
        localStorage.removeItem('cet6_extract_mode');
        localStorage.removeItem('cet6_extract_index');
        
        if (results.length > 0) {
            collectedResults = results;
            log(`已停止，共提取 ${results.length} 套`);
            updateExportButtons();
        }
    }

    // ========================================================================
    // 更新导出按钮状态
    // ========================================================================
    function updateExportButtons() {
        const hasData = collectedResults.length > 0;
        document.getElementById('btn-export-json').disabled = !hasData;
        document.getElementById('btn-export-csv').disabled = !hasData;
    }

    // ========================================================================
    // 导出JSON
    // ========================================================================
    function exportJSON() {
        if (collectedResults.length === 0) {
            log('没有数据可导出');
            return;
        }

        const data = JSON.stringify(collectedResults, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `六级听力答案_${new Date().toISOString().slice(0,10)}.json`;
        a.click();

        URL.revokeObjectURL(url);
        log(`✅ 已导出JSON文件，共 ${collectedResults.length} 套`);
    }

    // ========================================================================
    // 导出CSV - 包含详细年份信息，方便训练
    // ========================================================================
    function exportCSV() {
        if (collectedResults.length === 0) {
            log('没有数据可导出');
            return;
        }

        // 按年份月份排序（从旧到新）
        const sorted = [...collectedResults].sort((a, b) => {
            if (a.year !== b.year) return a.year - b.year;
            if (a.month !== b.month) return a.month - b.month;
            return a.set - b.set;
        });

        // 构建CSV内容 - 包含年份、月份、套数等详细信息
        let csv = '序号,年份,月份,套数,考试时间,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,完整答案\n';

        sorted.forEach((result, index) => {
            const examTime = `${result.year}年${result.month}月`;
            const row = [
                index + 1,
                result.year,
                result.month,
                result.set,
                examTime,
                ...result.answers,
                result.answersStr
            ];
            csv += row.join(',') + '\n';
        });

        // 添加BOM以支持中文
        const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `六级听力答案_${new Date().toISOString().slice(0,10)}.csv`;
        a.click();

        URL.revokeObjectURL(url);
        log(`✅ 已导出CSV文件，共 ${collectedResults.length} 套，按时间排序`);
    }

    // ========================================================================
    // 初始化
    // ========================================================================
    function init() {
        // 等待页面加载完成
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                createPanel();
                checkAutoMode();
            });
        } else {
            createPanel();
            checkAutoMode();
        }
        console.log('[CET6提取] 六级听力答案提取工具已加载');
    }

    init();
})();

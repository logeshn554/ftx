/**
 * Self-Evolving Crypto Trading Command Center Frontend
 * $100 Virtual Account Live Trade Lifecycle & Step-by-Step Neural Telemetry
 */

const host = window.location.hostname || '127.0.0.1';
const port = window.location.port || '8080';
const WS_URL = `ws://${host}:${port}/ws`;
const API_BASE = `http://${host}:${port}`;

// State
const state = {
    connected: false,
    equityCurve: [1000.00],
    labels: [new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })],
};

// Clock
function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });

    if (state.lastEvolutionTs) {
        const sec = Math.max(0, Math.floor(Date.now() / 1000 - state.lastEvolutionTs));
        const el = document.getElementById('evolve-time-ago');
        if (el) el.textContent = sec < 60 ? `(${sec}s ago)` : `(${Math.floor(sec/60)}m ago)`;
    }
}
setInterval(updateClock, 1000);
updateClock();

// Chart Initialization ($100 base)
const ctx = document.getElementById('equity-chart').getContext('2d');
const gradient = ctx.createLinearGradient(0, 0, 0, 260);
gradient.addColorStop(0, 'rgba(16, 185, 129, 0.35)');
gradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

const equityChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: state.labels,
        datasets: [{
            label: 'Virtual Equity ($)',
            data: state.equityCurve,
            borderColor: '#10b981',
            backgroundColor: gradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#10b981',
        }],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                borderColor: 'rgba(16, 185, 129, 0.4)',
                borderWidth: 1,
                callbacks: {
                    label: (ctx) => `Equity: $${ctx.parsed.y.toFixed(2)}`,
                },
            },
        },
        scales: {
            x: {
                grid: { color: 'rgba(51, 65, 85, 0.2)' },
                ticks: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 10 }, maxTicksLimit: 8 },
            },
            y: {
                grid: { color: 'rgba(51, 65, 85, 0.2)' },
                ticks: {
                    color: '#94a3b8',
                    font: { family: "'JetBrains Mono', monospace", size: 10 },
                    callback: (v) => `$${v.toFixed(1)}`,
                },
            },
        },
    },
});

// WebSocket Connection & Dual-Mode Telemetry Stream
let ws = null;
let reconnectTimer = null;
let fallbackPollingTimer = null;

function connectWebSocket() {
    try {
        if (ws) {
            try { ws.close(); } catch (e) {}
        }
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            state.connected = true;
            stopFallbackPolling();
            addEvent('success', '⚡ Real-time Bitcoin trading stream connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'pipeline_telemetry') {
                    handleTelemetry(msg.data);
                }
            } catch (e) {
                console.warn('Invalid WS payload:', event.data);
            }
        };

        ws.onclose = () => {
            state.connected = false;
            startFallbackPolling();
            scheduleReconnect();
        };

        ws.onerror = () => {
            state.connected = false;
            startFallbackPolling();
        };
    } catch (e) {
        startFallbackPolling();
        scheduleReconnect();
    }
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWebSocket();
    }, 2500);
}

function startFallbackPolling() {
    if (fallbackPollingTimer) return;
    fallbackPollingTimer = setInterval(async () => {
        if (state.connected) {
            stopFallbackPolling();
            return;
        }
        try {
            const res = await fetch(`${API_BASE}/api/state`);
            if (res.ok) {
                const data = await res.json();
                handleTelemetry(data);
            }
        } catch (e) {}
    }, 1000);
}

function stopFallbackPolling() {
    if (fallbackPollingTimer) {
        clearInterval(fallbackPollingTimer);
        fallbackPollingTimer = null;
    }
}

// Telemetry Handler
function handleTelemetry(data) {
    // 1. Bitcoin Price & Live Header
    if (data.btc_price !== undefined) {
        document.getElementById('btc-ticker').textContent = `$${data.btc_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    }
    if (data.current_step) {
        document.getElementById('pipeline-step-text').textContent = data.current_step;
    }
    if (data.regime) {
        document.getElementById('regime-status').textContent = data.regime;
    }

    // 2. Account KPIs ($100 virtual balance)
    if (data.equity !== undefined) {
        document.getElementById('equity-value').textContent = `$${data.equity.toFixed(2)}`;
        
        state.equityCurve.push(data.equity);
        state.labels.push(new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        if (state.equityCurve.length > 60) {
            state.equityCurve.shift();
            state.labels.shift();
        }
        equityChart.data.labels = state.labels;
        equityChart.data.datasets[0].data = state.equityCurve;
        equityChart.update('none');
    }

    if (data.cash_balance !== undefined) {
        document.getElementById('cash-value').textContent = `$${data.cash_balance.toFixed(2)}`;
    }

    if (data.daily_pnl !== undefined) {
        const pnlEl = document.getElementById('pnl-value');
        const pnlDelta = document.getElementById('pnl-delta');
        const sign = data.daily_pnl >= 0 ? '+' : '-';
        pnlEl.textContent = `${sign}$${Math.abs(data.daily_pnl).toFixed(2)}`;
        pnlEl.className = data.daily_pnl >= 0 ? 'kpi-value pnl-positive' : 'kpi-value pnl-negative';

        if (data.daily_return !== undefined) {
            pnlDelta.textContent = `${(data.daily_return * 100).toFixed(2)}%`;
            pnlDelta.className = data.daily_return >= 0 ? 'kpi-delta kpi-delta--positive' : 'kpi-delta kpi-delta--negative';
        }
    }

    if (data.total_trades !== undefined) {
        document.getElementById('trades-count-val').textContent = data.total_trades;
        document.getElementById('win-rate-val').textContent = `Win Rate: ${data.win_rate || 0}%`;
    }

    if (data.total_fees_paid !== undefined) {
        const feesEl = document.getElementById('fees-total-val');
        if (feesEl) feesEl.textContent = `$${data.total_fees_paid.toFixed(2)}`;
    }

    if (data.drawdown !== undefined) {
        document.getElementById('drawdown-value').textContent = `${(data.drawdown * 100).toFixed(2)}%`;
        document.getElementById('drawdown-fill').style.width = `${Math.min(data.drawdown * 500, 100)}%`;
    }

    // 3. Neural Decision Pipeline Boxes
    if (data.detected_pattern) {
        document.getElementById('pipe-pattern-name').textContent = data.detected_pattern;
    }
    if (data.rsi !== undefined) {
        document.getElementById('pipe-rsi').textContent = data.rsi.toFixed(1);
    }
    if (data.macd !== undefined) {
        document.getElementById('pipe-macd').textContent = data.macd;
    }
    if (data.regime) {
        document.getElementById('pipe-regime-name').textContent = `${data.regime} (${Math.round((data.regime_confidence || 0.85)*100)}% conf)`;
    }
    if (data.faiss_recall) {
        document.getElementById('pipe-faiss-match').textContent = data.faiss_recall;
    }
    if (data.risk_status) {
        document.getElementById('pipe-xgb-status').textContent = data.risk_status;
    }
    if (data.risk_score !== undefined) {
        document.getElementById('pipe-risk-score').textContent = `${data.risk_score.toFixed(2)} / 0.35 max`;
    }
    if (data.last_decision) {
        document.getElementById('pipe-decision-text').textContent = data.last_decision;
    }

    // 4. Live Open Position Table & Progress
    const posTable = document.getElementById('positions-body');
    const badge = document.getElementById('pos-status-badge');
    const pnlFill = document.getElementById('pos-pnl-fill');
    const pnlText = document.getElementById('pos-pnl-text');
    const capText = document.getElementById('capital-util-text');

    if (data.open_position) {
        const pos = data.open_position;
        badge.textContent = `ACTIVE (${pos.side})`;
        badge.className = 'badge badge-active ' + (pos.side === 'BUY' ? 'badge-buy' : 'badge-sell');

        const pnlSign = pos.pnl >= 0 ? '+' : '';
        const pnlCls = pos.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        posTable.innerHTML = `
            <tr>
                <td><span class="badge ${pos.side === 'BUY' ? 'badge-buy' : 'badge-sell'}">${pos.side}</span></td>
                <td>$${pos.allocated_capital.toFixed(2)} (${pos.quantity.toFixed(5)} BTC)</td>
                <td>$${pos.entry_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
                <td>$${pos.current_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
                <td class="${pnlCls}"><strong>${pnlSign}$${pos.pnl.toFixed(2)} (${pos.pnl_pct >= 0 ? '+' : ''}${pos.pnl_pct.toFixed(2)}%)</strong></td>
            </tr>
        `;

        pnlText.textContent = `${pnlSign}$${pos.pnl.toFixed(2)} (${pos.pnl_pct.toFixed(2)}%)`;
        pnlText.className = 'meter-value ' + pnlCls;
        pnlFill.style.width = `${Math.min(Math.max((pos.pnl_pct + 2) * 25, 5), 100)}%`;
        pnlFill.className = pos.pnl >= 0 ? 'meter-fill meter-fill--green' : 'meter-fill meter-fill--red';

        capText.textContent = `$${pos.allocated_capital.toFixed(2)} / $1,000.00`;
    } else {
        badge.textContent = 'FLAT';
        badge.className = 'badge';
        posTable.innerHTML = `<tr class="empty-row"><td colspan="5">Scanning for next high-probability pattern...</td></tr>`;
        pnlText.textContent = '$0.00 (0.00%)';
        pnlText.className = 'meter-value';
        pnlFill.style.width = '0%';
        capText.textContent = `$0.00 / $1,000.00`;
    }

    // === 5. Closed Trades History (Live Stream) ===
    if (data.recent_trades && data.recent_trades.length) {
        renderClosedTrades(data.recent_trades);
    }

    // === 6. Live Events Synchronizer ===
    if (data.events && data.events.length) {
        syncEvents(data.events);
    }

    // === 7. WebRL Self-Evolving Telemetry ===
    if (data.webrl) {
        updateWebRLPanel(data.webrl);
    }
}

function syncEvents(eventsList) {
    const log = document.getElementById('event-log');
    if (!log || !eventsList || !eventsList.length) return;
    
    // Render top event if not already at the top
    const topMsg = eventsList[0].message;
    const currentFirst = log.firstChild ? log.firstChild.querySelector('.log-msg')?.textContent : '';
    if (topMsg !== currentFirst) {
        log.innerHTML = '';
        eventsList.forEach(e => {
            const entry = document.createElement('div');
            entry.className = `log-entry log-entry--${e.level || 'info'}`;
            entry.innerHTML = `
                <span class="log-time">${e.time || new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
                <span class="log-msg">${e.message}</span>
            `;
            log.appendChild(entry);
        });
    }
}

// Learning Curve Chart (separate from equity chart)
let learningChart = null;
function initLearningChart() {
    const canvas = document.getElementById('learning-curve-canvas');
    if (!canvas) return;
    const lctx = canvas.getContext('2d');
    const grad = lctx.createLinearGradient(0, 0, 0, 160);
    grad.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
    grad.addColorStop(1, 'rgba(139, 92, 246, 0.0)');

    learningChart = new Chart(lctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Win Rate %',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: grad,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#a78bfa',
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: 'rgba(51, 65, 85, 0.2)' },
                    ticks: { color: '#94a3b8', font: { size: 9 }, maxTicksLimit: 10 },
                },
                y: {
                    min: 0, max: 100,
                    grid: { color: 'rgba(51, 65, 85, 0.2)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        callback: (v) => `${v}%`,
                        stepSize: 25,
                    },
                },
            },
        },
    });
}
initLearningChart();

function updateWebRLPanel(webrl) {
    // Status badge
    const statusBadge = document.getElementById('webrl-status-badge');
    if (statusBadge) {
        statusBadge.textContent = webrl.total_losses > 0 ? `EVOLVING (${webrl.total_trades} trades)` : 'LEARNING';
    }

    // --- Advanced RL Suite: MuZero MCTS + GRPO + PDRL ---
    if (webrl.pdrl) {
        const pd = webrl.pdrl;
        const pdrlBadge = document.getElementById('pdrl-mode-badge');
        const pdrlState = document.getElementById('pdrl-state-text');
        const pdrlStreak = document.getElementById('pdrl-streak-badge');
        const pdrlRule = document.getElementById('pdrl-rule-text');

        if (pdrlBadge) {
            if (pd.exploitation_mode_active) {
                pdrlBadge.textContent = '🎯 PDRL DEEP EXPLOITATION ACTIVE';
                pdrlBadge.style.background = 'rgba(239, 68, 68, 0.25)';
                pdrlBadge.style.color = '#f87171';
                pdrlBadge.style.borderColor = 'rgba(239, 68, 68, 0.6)';
            } else {
                pdrlBadge.textContent = '⚖️ BALANCED DYNAMICS';
                pdrlBadge.style.background = 'rgba(16, 185, 129, 0.2)';
                pdrlBadge.style.color = '#10b981';
                pdrlBadge.style.borderColor = 'rgba(16, 185, 129, 0.5)';
            }
        }
        if (pdrlState) {
            pdrlState.textContent = pd.exploitation_mode_active ? '🎯 DEEP EXPLOITATION (LOSS RECOVERY)' : '⚖️ BALANCED';
            pdrlState.style.color = pd.exploitation_mode_active ? '#f87171' : '#10b981';
        }
        if (pdrlStreak) {
            pdrlStreak.textContent = `Streak: ${pd.consecutive_losses} Loss`;
        }
        if (pdrlRule) {
            pdrlRule.textContent = pd.exploitation_mode_active
                ? `🚨 Loss Streak active: Exploration disabled (ε=0.0). PDRL enforcing strict historical pattern memory & positive MuZero MCTS lookahead EV.`
                : `Balanced mode: Dynamic curriculum learning active with fractional Kelly risk sizing.`;
        }
    }

    if (webrl.muzero) {
        const mz = webrl.muzero;
        const evEl = document.getElementById('muzero-ev-val');
        const prunedEl = document.getElementById('muzero-pruned-val');
        const planEl = document.getElementById('muzero-plan-text');
        const depthEl = document.getElementById('muzero-depth-badge');

        if (depthEl) depthEl.textContent = `Depth: ${mz.lookahead_depth} Bars (${mz.simulations_per_search} Sims)`;
        if (prunedEl) prunedEl.textContent = `${mz.pruned_branches_count} Paths`;

        if (mz.last_plan && mz.last_plan.expected_value_pct !== undefined) {
            const ev = mz.last_plan.expected_value_pct;
            if (evEl) {
                evEl.textContent = `${ev >= 0 ? '+' : ''}${ev.toFixed(2)}% EV`;
                evEl.style.color = ev >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
            }
            if (planEl) {
                planEl.textContent = `MCTS Action: ${mz.last_plan.best_action} (Win Prob: ${((mz.last_plan.win_probability || 0.5)*100).toFixed(0)}%, Status: ${mz.last_plan.mcts_status})`;
            }
        }
    }

    if (webrl.grpo) {
        const gr = webrl.grpo;
        const advEl = document.getElementById('grpo-adv-val');
        const winnerEl = document.getElementById('grpo-winner-val');
        const descEl = document.getElementById('grpo-desc-text');

        if (gr.last_group_result && gr.last_group_result.top_advantage !== undefined) {
            const res = gr.last_group_result;
            if (advEl) advEl.textContent = `+${res.top_advantage.toFixed(2)} A*`;
            if (winnerEl) winnerEl.textContent = `${res.top_candidate}`;
            if (descEl) descEl.textContent = `Advantage Spread: ${res.advantage_spread} | Mean Reward: ${res.mean_reward}`;
        }
    }

    // --- Autonomous RL Goal & Ruin Survival Tracker ($0.00 <-> $1,050.00) ---
    if (webrl.goal_survival) {
        const gs = webrl.goal_survival;
        const curEqEl = document.getElementById('goal-current-equity');
        const tgtValEl = document.getElementById('goal-target-val');
        const distEl = document.getElementById('goal-dist-text');
        const deltaEl = document.getElementById('goal-delta-text');
        const ruinEl = document.getElementById('ruin-safety-text');
        const pctEl = document.getElementById('goal-progress-pct-text');
        const barEl = document.getElementById('goal-progress-bar');
        const genEl = document.getElementById('rl-generation-badge');
        const wonEl = document.getElementById('goals-won-badge');
        const bannerEl = document.getElementById('goal-status-banner');

        if (curEqEl) curEqEl.textContent = `$${gs.current_equity.toFixed(2)}`;
        if (tgtValEl) tgtValEl.textContent = `$${gs.profit_target.toFixed(2)}`;
        if (distEl) distEl.textContent = `Distance: $${gs.distance_to_goal.toFixed(2)} to Goal`;
        if (deltaEl) {
            const diff = gs.current_equity - gs.start_capital;
            deltaEl.textContent = `${diff >= 0 ? '+' : ''}$${diff.toFixed(2)} (${((diff/gs.start_capital)*100).toFixed(2)}%)`;
            deltaEl.style.color = diff >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
        }
        if (ruinEl) ruinEl.textContent = `Buffer: $${gs.safety_buffer_to_ruin.toFixed(2)}`;
        if (pctEl) pctEl.textContent = `${gs.progress_to_target_pct.toFixed(1)}%`;
        if (barEl) barEl.style.width = `${Math.min(gs.progress_to_target_pct, 100)}%`;
        if (genEl) genEl.textContent = `Gen #${gs.generation}`;
        if (wonEl) wonEl.textContent = `${gs.goals_achieved} Goals Won`;

        if (gs.last_event && bannerEl) {
            const ev = gs.last_event;
            const color = ev.type === 'PROFIT_GOAL_ACHIEVED' ? 'var(--accent-green)' : '#f59e0b';
            bannerEl.innerHTML = `<strong style="color: ${color};">[${ev.timestamp}] ${ev.type.replace(/_/g, ' ')}:</strong> ${ev.message} <br><em>Action: ${ev.actions}</em>`;
        }
    }

    // --- Loss Analyzer ---
    const la = webrl.loss_analyzer;
    if (la) {
        document.getElementById('total-failures-badge').textContent = `${la.total_failures} Failures`;

        if (la.last_failure) {
            const f = la.last_failure;
            document.getElementById('failure-cause').textContent = f.failure_cause.replace(/_/g, ' ');
            document.getElementById('failure-explanation').textContent = f.explanation || '';
            document.getElementById('failure-fix').textContent = f.suggested_fix ? `Fix: ${f.suggested_fix}` : '';
        }

        // Cause distribution chips
        const distEl = document.getElementById('cause-distribution');
        if (distEl && la.cause_distribution) {
            distEl.innerHTML = Object.entries(la.cause_distribution)
                .filter(([, v]) => v > 0)
                .map(([cause, count]) =>
                    `<span class="cause-chip">${cause.replace(/_/g, ' ')} <span class="cause-chip-count">${count}</span></span>`
                ).join('');
        }
    }

    // --- Win Analyzer & Pattern Memory Bank ---
    const wa = webrl.win_analyzer;
    if (wa) {
        const winBadge = document.getElementById('total-wins-analyzed-badge');
        if (winBadge) winBadge.textContent = `${wa.total_analyzed_wins} Wins Stored`;

        if (wa.last_win) {
            const w = wa.last_win;
            const driverEl = document.getElementById('win-driver');
            const expEl = document.getElementById('win-explanation');
            const ruleEl = document.getElementById('win-rule');
            if (driverEl) driverEl.textContent = `${w.profit_driver.replace(/_/g, ' ')}: +$${w.pnl.toFixed(2)}`;
            if (expEl) expEl.textContent = `${w.pattern} (${w.side}) | ${w.explanation}`;
            if (ruleEl) ruleEl.textContent = w.extracted_rule ? `Rule: ${w.extracted_rule}` : '';
        }
    }

    const pm = webrl.pattern_memory;
    if (pm && pm.patterns_list) {
        const chipsEl = document.getElementById('pattern-memory-chips');
        if (chipsEl) {
            chipsEl.innerHTML = pm.patterns_list.map(p => {
                const isDisc = p.status.includes('DISCOVERED');
                const badgeColor = isDisc ? '#8b5cf6' : '#10b981';
                return `
                    <span style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 6px; font-size: 10px; font-family: var(--font-mono); background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3);">
                        <strong style="color: var(--text-primary);">${p.name.split(' ')[0]}</strong>
                        <span style="color: ${badgeColor}; font-weight: 700;">${p.wins}W/${p.losses}L</span>
                        <span style="color: var(--accent-cyan);">${(p.weight*100).toFixed(0)}%</span>
                    </span>
                `;
            }).join('');
        }
    }

    // --- ORM ---
    const orm = webrl.orm;
    if (orm) {
        document.getElementById('orm-updates-badge').textContent = `${orm.total_updates} updates`;
        document.getElementById('orm-buffer-size').textContent = orm.buffer_size;
        document.getElementById('orm-avg-score').textContent = orm.avg_recent_score.toFixed(3);
        document.getElementById('orm-weights-norm').textContent = orm.weights_norm.toFixed(3);

        // Position the gauge indicator: score -1..+1 maps to 0%..100%
        const gaugePos = ((orm.avg_recent_score + 1) / 2) * 100;
        document.getElementById('orm-gauge-fill').style.left = `${gaugePos}%`;
        document.getElementById('orm-score-display').textContent = orm.avg_recent_score >= 0
            ? `+${orm.avg_recent_score.toFixed(3)}`
            : orm.avg_recent_score.toFixed(3);
    }

    // --- Curriculum ---
    const cur = webrl.curriculum;
    if (cur) {
        document.getElementById('curriculum-size-badge').textContent = `${cur.curriculum_size} items`;

        const listEl = document.getElementById('curriculum-list');
        if (listEl && cur.top_lessons && cur.top_lessons.length > 0) {
            listEl.innerHTML = cur.top_lessons.map(item => `
                <div class="curriculum-item">
                    <div class="curriculum-item-header">
                        <span class="curriculum-item-source">${item.source.replace(/_/g, ' ')}</span>
                        <span class="curriculum-item-priority">Priority: ${item.priority.toFixed(2)}</span>
                    </div>
                    <div class="curriculum-item-scenario">${item.scenario}</div>
                </div>
            `).join('');
        } else if (listEl) {
            listEl.innerHTML = '<div class="curriculum-empty">No curriculum items yet. Losses will generate training scenarios.</div>';
        }
    }

    // --- Policy Adapter ---
    const pol = webrl.policy;
    if (pol) {
        document.getElementById('policy-adaptations-badge').textContent = `${pol.total_adaptations} adaptations`;

        // KL meter
        const klPct = Math.min((pol.kl_distance / pol.kl_budget) * 100, 100);
        document.getElementById('kl-fill').style.width = `${klPct}%`;
        document.getElementById('kl-value').textContent = pol.kl_distance.toFixed(4);
        document.getElementById('kl-budget-value').textContent = `/ ${pol.kl_budget.toFixed(3)} budget`;

        // Adapted params
        if (pol.current_params) {
            const p = pol.current_params;
            document.getElementById('param-pos-size').textContent = `${(p.position_size_pct * 100).toFixed(1)}%`;
            document.getElementById('param-stop-loss').textContent = `${p.stop_loss_pct.toFixed(3)}%`;
            document.getElementById('param-take-profit').textContent = `${p.take_profit_pct.toFixed(3)}%`;
            document.getElementById('param-risk-thresh').textContent = p.xgb_risk_threshold.toFixed(4);

            // Pattern confidences
            const confEl = document.getElementById('pattern-confidences');
            if (confEl && p.pattern_confidences) {
                confEl.innerHTML = Object.entries(p.pattern_confidences).map(([name, conf]) => {
                    const shortName = name.length > 25 ? name.substring(0, 25) + '…' : name;
                    return `
                        <div class="pattern-conf-item">
                            <span class="pattern-conf-name" title="${name}">${shortName}</span>
                            <div class="pattern-conf-bar">
                                <div class="pattern-conf-fill" style="width: ${conf * 100}%"></div>
                            </div>
                            <span class="pattern-conf-value">${(conf * 100).toFixed(1)}%</span>
                        </div>
                    `;
                }).join('');
            }
        }
    }

    // --- Learning Curve ---
    document.getElementById('webrl-total-trades').textContent = webrl.total_trades;
    document.getElementById('webrl-total-wins').textContent = webrl.total_wins;
    document.getElementById('webrl-total-losses').textContent = webrl.total_losses;
    document.getElementById('webrl-win-rate').textContent = `${webrl.win_rate}%`;

    // --- 100-Attempt Macro Deep Weight Optimizer ---
    if (webrl.attempts_progress) {
        const ap = webrl.attempts_progress;
        const counterEl = document.getElementById('attempt-counter-text');
        const countdnEl = document.getElementById('attempt-countdown-text');
        const barEl = document.getElementById('attempt-progress-bar');
        const badgeEl = document.getElementById('milestone-badge');
        
        if (counterEl) counterEl.textContent = `${ap.current_cycle_progress} / 100 Attempts (Total: ${ap.current_attempt})`;
        if (countdnEl) countdnEl.textContent = `Next Deep Weight Recalibration in ${ap.remaining_to_milestone} attempt${ap.remaining_to_milestone === 1 ? '' : 's'}`;
        if (barEl) barEl.style.width = `${Math.min(ap.current_cycle_progress, 100)}%`;
        if (badgeEl) badgeEl.textContent = `Milestone #${ap.milestones_completed}`;
    }

    if (webrl.last_milestone_report) {
        const mr = webrl.last_milestone_report;
        const summaryEl = document.getElementById('last-milestone-summary');
        if (summaryEl) {
            summaryEl.innerHTML = `
                <div style="color: #f59e0b; font-weight: 700; margin-bottom: 4px;">🏆 Milestone #${mr.milestone_number} Optimization Executed (${mr.timestamp})</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-top: 4px;">
                    <div>• <strong>Top Loss Driver Neutralized:</strong> <span style="color: var(--accent-red);">${mr.top_loss_driver.replace(/_/g, ' ')}</span></div>
                    <div>• <strong>Est. Loss Reduction:</strong> <span style="color: var(--accent-green); font-weight: 700;">${mr.estimated_loss_reduction}</span></div>
                    <div>• <strong>Win Rate over 100 attempts:</strong> <span>${mr.win_rate}%</span></div>
                </div>
            `;
        }
    }

    // --- Evolution Timestamp Tracker & History ---
    if (webrl.last_evolution_timestamp) {
        state.lastEvolutionTs = webrl.last_evolution_timestamp;
    }
    if (webrl.last_evolution_time) {
        const timeBadge = document.getElementById('last-evolve-time-badge');
        const timeAgoEl = document.getElementById('evolve-time-ago');
        const countBadge = document.getElementById('total-evolutions-badge');
        if (timeBadge) timeBadge.textContent = webrl.last_evolution_time;
        if (countBadge && webrl.total_evolutions_count !== undefined) {
            countBadge.textContent = `${webrl.total_evolutions_count} Cycles`;
        }

        if (webrl.last_evolution_timestamp && timeAgoEl) {
            const sec = Math.max(0, Math.floor(Date.now() / 1000 - webrl.last_evolution_timestamp));
            timeAgoEl.textContent = sec < 60 ? `(${sec}s ago)` : `(${Math.floor(sec/60)}m ago)`;
        }
    }

    if (webrl.evolution_history && webrl.evolution_history.length > 0) {
        const listEl = document.getElementById('evolution-timeline-list');
        if (listEl) {
            listEl.innerHTML = webrl.evolution_history.slice().reverse().map(ev => {
                const isWin = ev.type.includes('WIN');
                const isMilestone = ev.type.includes('MILESTONE');
                const isGoal = ev.type.includes('GOAL');
                const badgeColor = isGoal ? '#10b981' : (isMilestone ? '#f59e0b' : (isWin ? '#3b82f6' : '#ef4444'));
                return `
                    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.25); padding: 4px 8px; border-radius: 6px; font-size: 11px; font-family: var(--font-mono);">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: var(--accent-cyan); font-weight: 700;">${ev.time}</span>
                            <span style="color: ${badgeColor}; font-weight: 600; padding: 1px 4px; border-radius: 3px; background: rgba(255,255,255,0.05); font-size: 10px;">${ev.type.replace(/_/g, ' ')}</span>
                            <span style="color: var(--text-primary);">${ev.reason}</span>
                        </div>
                        <span style="color: var(--text-muted); font-size: 10px;">${ev.details}</span>
                    </div>
                `;
            }).join('');
        }
    }

    if (learningChart && webrl.learning_curve && webrl.learning_curve.length) {
        learningChart.data.labels = webrl.learning_curve.map(d => `#${d.trade_num}`);
        learningChart.data.datasets[0].data = webrl.learning_curve.map(d => d.win_rate);
        learningChart.update('none');
    }
}

// Handle Closed Trade History & Initial State with Auto-Retry
async function fetchInitialTrades(retryCount = 0) {
    try {
        const res = await fetch(`${API_BASE}/trading/status`);
        if (res.ok) {
            const data = await res.json();
            handleTelemetry(data);
            if (data.recent_trades && data.recent_trades.length) {
                renderClosedTrades(data.recent_trades);
            }
        }
    } catch (e) {
        if (retryCount < 5) {
            setTimeout(() => fetchInitialTrades(retryCount + 1), 2000);
        }
    }
}

function renderClosedTrades(trades) {
    const tbody = document.getElementById('trades-body');
    if (!tbody || !trades || !trades.length) return;
    tbody.innerHTML = '';
    trades.forEach(t => {
        const row = document.createElement('tr');
        const pnlCls = t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const entryPrice = t.entry_price ? `$${Number(t.entry_price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
        const exitPrice = t.exit_price ? `$${Number(t.exit_price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
        const feeFormatted = t.fee !== undefined ? `-$${Number(t.fee).toFixed(2)}` : '-$0.00';
        row.innerHTML = `
            <td>${t.time || '—'}</td>
            <td><span class="badge ${t.side === 'BUY' ? 'badge-buy' : 'badge-sell'}">${t.side}</span></td>
            <td>${t.pattern || 'Breakout'}</td>
            <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary);">${entryPrice}</td>
            <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-primary);">${exitPrice}</td>
            <td style="font-family: var(--font-mono); font-size: 11px; color: #f59e0b;">${feeFormatted}</td>
            <td class="${pnlCls}">${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}</td>
            <td class="${pnlCls}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%</td>
            <td><strong>$${(t.balance_after || 1000.0).toFixed(2)}</strong></td>
        `;
        tbody.appendChild(row);
    });
    document.getElementById('recent-trades-count').textContent = trades.length;
}

function addEvent(level, message) {
    const log = document.getElementById('event-log');
    if (!log) return;
    const entry = document.createElement('div');
    entry.className = `log-entry log-entry--${level}`;
    entry.innerHTML = `
        <span class="log-time">${new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
        <span class="log-msg">${message}</span>
    `;
    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 40) log.removeChild(log.lastChild);
}

// Reset Virtual Capital Button
document.getElementById('btn-reset-capital')?.addEventListener('click', async () => {
    try {
        const res = await fetch(`${API_BASE}/api/reset_capital`, { method: 'POST' });
        if (res.ok) {
            addEvent('success', '💵 Virtual Account Reset: Capital restored to $1,000.00');
            state.equityCurve = [1000.00];
            state.labels = [new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })];
            equityChart.data.labels = state.labels;
            equityChart.data.datasets[0].data = state.equityCurve;
            equityChart.update();
        }
    } catch (e) {
        addEvent('warning', 'Resetting local buffer');
    }
});

// Evolution Button
document.getElementById('btn-trigger-evolve')?.addEventListener('click', async () => {
    addEvent('info', '🔄 Triggering self-evolution cycle...');
    try {
        const res = await fetch(`${API_BASE}/api/evolve`, { method: 'POST' });
        if (res.ok) {
            addEvent('success', '✨ Retraining cycle dispatched to RL agent with FAISS experience bank');
        }
    } catch (e) {
        addEvent('warning', 'Evolution request buffered');
    }
});

// Circuit Breaker Button
document.getElementById('btn-toggle-circuit')?.addEventListener('click', async () => {
    try {
        const res = await fetch(`${API_BASE}/api/circuit_breaker`, { method: 'POST' });
        if (res.ok) {
            const data = await res.json();
            const dot = document.getElementById('circuit-dot');
            const status = document.getElementById('circuit-status');
            status.textContent = data.circuit_breaker;
            dot.className = data.circuit_breaker === 'OPEN' ? 'status-dot status-dot--danger' : 'status-dot status-dot--active';
            addEvent(data.circuit_breaker === 'OPEN' ? 'error' : 'success', `Circuit Breaker set to ${data.circuit_breaker}`);
        }
    } catch (e) {
        addEvent('warning', 'Circuit breaker toggled');
    }
});

// Initialize
fetchInitialTrades();
connectWebSocket();

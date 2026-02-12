// 修复后的阶段动画处理函数
// 将此代码替换 frontend/index.html 中的 addStage 和消息处理部分

/**
 * 修复后的阶段动画处理
 * 
 * 改进点：
 * 1. 更精确的正则匹配，避免残留标记
 * 2. 改进的动画触发时机
 * 3. 更好的技能节点高亮效果
 */

// ========== 1. 修复阶段解析 ==========
function parseStageData(buffer) {
    const stages = [];
    const stagePattern = /\|STAGE\|([^|]+)\|/g;
    let match;
    
    while ((match = stagePattern.exec(buffer)) !== null) {
        const fullMatch = match[0];
        const stageName = match[1];
        
        // 提取完整阶段块（包含所有参数）
        const startIdx = match.index;
        let endIdx = startIdx + fullMatch.length;
        
        // 查找结束标记（下一个|STAGE|或字符串结尾）
        const nextStage = buffer.indexOf('|STAGE|', endIdx);
        if (nextStage > 0) {
            endIdx = nextStage;
        } else {
            endIdx = buffer.length;
        }
        
        const stageBlock = buffer.substring(startIdx, endIdx);
        
        // 提取参数
        const data = { stage: stageName };
        
        // 提取skills
        const skillsMatch = stageBlock.match(/\|skills\|([^|]+)\|/);
        if (skillsMatch && skillsMatch[1] !== 'none') {
            data.skills = skillsMatch[1].split(',').filter(s => s && s !== 'processing');
        }
        
        // 提取entropy
        const entropyMatch = stageBlock.match(/\|entropy\|([\d.]+)\|/);
        if (entropyMatch) {
            data.entropy = parseFloat(entropyMatch[1]);
        }
        
        // 提取mode
        const modeMatch = stageBlock.match(/\|mode\|([^|]+)\|/);
        if (modeMatch) {
            data.mode = modeMatch[1];
        }
        
        stages.push(data);
    }
    
    return stages;
}

// ========== 2. 改进的阶段动画 ==========
function addStageEnhanced(stage, skills, metadata = {}) {
    const names = {
        resonance: { 
            color: 'var(--accent2)', 
            name: '共振', 
            icon: '◎',
            description: '技能选择中...'
        },
        interference: { 
            color: 'var(--accent)', 
            name: '干涉', 
            icon: '⇌',
            description: metadata.tools ? `处理 ${metadata.tools} 个工具` : '处理中...'
        },
        collapse: { 
            color: 'var(--accent3)', 
            name: '坍缩', 
            icon: '◎',
            description: skills && skills.length > 0 ? `激活: ${skills.join(', ')}` : '生成回复...'
        }
    };
    
    const info = names[stage] || { color: '#888', name: stage, icon: '•', description: '' };
    
    // 创建阶段项
    const item = document.createElement('div');
    item.className = 'stage-item stage-' + stage;
    item.style.animation = 'stageIn 0.5s ease forwards';
    item.innerHTML = `
        <span style="color:${info.color};font-size:14px;">${info.icon}</span>
        <span style="color:${info.color};font-weight:600;">${info.name}</span>
        <span style="color:var(--text2);margin-left:8px;font-size:11px;">${info.description}</span>
    `;
    
    // 添加技能标签
    if (skills && skills.length > 0) {
        const skillsSpan = document.createElement('span');
        skillsSpan.style.cssText = 'margin-left:auto;display:flex;gap:4px;';
        skills.forEach(skill => {
            const tag = document.createElement('span');
            tag.style.cssText = `
                background: ${info.color}20;
                color: ${info.color};
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                border: 1px solid ${info.color}40;
            `;
            tag.textContent = skill;
            skillsSpan.appendChild(tag);
        });
        item.appendChild(skillsSpan);
    }
    
    // 添加到时间线
    const timeline = document.querySelector('.stage-timeline');
    if (timeline) {
        // 检查是否已存在同阶段
        const existing = timeline.querySelector('.stage-' + stage);
        if (!existing) {
            timeline.appendChild(item);
        }
    }
    
    // ========== 3. 改进的技能节点动画 ==========
    if (skills) {
        skills.forEach((skill, index) => {
            const node = document.getElementById('skill-' + skill);
            if (node) {
                // 清除之前的状态
                node.classList.remove('active', 'resonating', 'processing');
                void node.offsetWidth; // 强制重绘
                
                // 根据阶段添加不同动画
                if (stage === 'resonance') {
                    // 共振阶段：蓝色脉冲，快速闪烁
                    node.classList.add('resonating');
                    node.style.animationDelay = (index * 0.1) + 's';
                } else if (stage === 'interference') {
                    // 干涉阶段：绿色脉冲，表示处理中
                    node.classList.add('processing');
                } else if (stage === 'collapse') {
                    // 坍缩阶段：稳定高亮
                    node.classList.add('active');
                }
            }
        });
    }
    
    // 更新状态指示器
    setStatus(stage, info.name);
}

// ========== 4. 改进的消息处理 ==========
async function sendMessageFixed() {
    const input = document.getElementById('msg-input');
    const btn = document.getElementById('send-btn');
    const msg = input.value.trim();
    
    if (!msg || isSending) return;
    
    isSending = true;
    input.value = '';
    btn.disabled = true;
    
    // 添加用户消息
    addMessage('user', msg);
    
    // 清除之前的技能状态
    document.querySelectorAll('.skill-node').forEach(n => {
        n.classList.remove('active', 'resonating', 'processing');
        n.style.animationDelay = '';
    });
    
    try {
        const res = await fetch(API + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                user_id: userId,
                session_id: curSession,
                domain_focus: curDomain
            })
        });
        
        // 创建AI消息容器
        const aiDiv = document.createElement('div');
        aiDiv.className = 'message ai';
        aiDiv.innerHTML = `
            <div class="message-label">⚛ Field</div>
            <span class="content"></span>
            <div class="stage-timeline"></div>
        `;
        document.getElementById('chat-container').appendChild(aiDiv);
        
        const contentSpan = aiDiv.querySelector('.content');
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        
        let buffer = '';
        let fullText = '';
        let shownStages = new Set();
        
        // 读取流
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // ========== 关键修复：更精确的阶段解析 ==========
            const stages = parseStageData(buffer);
            
            stages.forEach(stageData => {
                const { stage, skills, entropy, mode } = stageData;
                
                if (!shownStages.has(stage)) {
                    shownStages.add(stage);
                    
                    // 提取工具数量
                    const toolsMatch = buffer.match(/(\d+)\s*tools/i);
                    const metadata = {
                        tools: toolsMatch ? toolsMatch[1] : null,
                        entropy: entropy,
                        mode: mode
                    };
                    
                    // 触发阶段动画
                    addStageEnhanced(stage, skills, metadata);
                    
                    // 从buffer中移除已处理的阶段标记
                    buffer = buffer.replace(/\|STAGE\|[^|]+\|/g, '');
                }
            });
            
            // ========== 关键修复：更好的内容清理 ==========
            let cleanContent = buffer
                // 移除所有阶段标记及其参数
                .replace(/\|STAGE\|[^|]*(?:\|[^|]*)*\|/g, '')
                // 移除残留的关键词
                .replace(/\b(complete|processing|none)\b/gi, '')
                // 移除工具数量标记
                .replace(/\d+\s*tools/gi, '')
                // 移除field_density标记
                .replace(/field_density\|\d+/g, '')
                // 移除entropy标记
                .replace(/entropy\|[\d.]+/g, '')
                // 移除mode标记
                .replace(/mode\|[^|]*/g, '')
                // 移除skills标记
                .replace(/skills\|[^|]*/g, '')
                // 移除多余的管道符
                .replace(/\|+/g, ' ')
                // 规范化空白
                .replace(/\s+/g, ' ')
                .trim();
            
            if (cleanContent) {
                fullText += cleanContent;
                contentSpan.innerHTML = formatContent(fullText);
                document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;
                
                // 清空buffer中已处理的内容
                buffer = '';
            }
        }
        
        // 完成处理
        setTimeout(() => {
            setStatus('ready', '就绪');
            
            // 添加完成标记
            const timeline = aiDiv.querySelector('.stage-timeline');
            if (timeline) {
                const completeItem = document.createElement('div');
                completeItem.className = 'stage-item';
                completeItem.style.cssText = 'margin-top:8px;color:var(--accent);font-size:11px;opacity:0.7;';
                completeItem.innerHTML = '✓ 坍缩完成';
                timeline.appendChild(completeItem);
            }
            
            aiDiv.querySelector('.message-label').innerHTML = '⚛ Field <span style="opacity:0.5;font-size:10px;">· 完成</span>';
        }, 500);
        
    } catch(e) {
        addMessage('ai', '❌ 错误: ' + e.message);
        setStatus('ready', '就绪');
    }
    
    isSending = false;
    btn.disabled = false;
}

// ========== 5. 内容格式化 ==========
function formatContent(text) {
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code style="background:rgba(0,255,136,0.1);padding:2px 6px;border-radius:4px;font-family:monospace;">$1</code>');
}

// ========== 6. 新增CSS样式（需要添加到<style>标签中） ==========
const additionalStyles = `
/* 改进的阶段项样式 */
.stage-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    padding: 4px 0;
    font-size: 12px;
    border-left: 2px solid transparent;
    padding-left: 8px;
    transition: all 0.3s ease;
}

.stage-item.stage-resonance {
    border-left-color: var(--accent2);
    background: linear-gradient(90deg, rgba(0,170,255,0.05), transparent);
}

.stage-item.stage-interference {
    border-left-color: var(--accent);
    background: linear-gradient(90deg, rgba(0,255,136,0.05), transparent);
}

.stage-item.stage-collapse {
    border-left-color: var(--accent3);
    background: linear-gradient(90deg, rgba(255,102,170,0.05), transparent);
}

/* 改进的技能节点动画 */
.skill-node.processing {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,255,136,0.1) !important;
    animation: skillProcessing 1s ease infinite !important;
}

@keyframes skillProcessing {
    0%, 100% { 
        transform: scale(1); 
        box-shadow: 0 0 0 rgba(0,255,136,0);
    }
    50% { 
        transform: scale(1.1); 
        box-shadow: 0 0 20px rgba(0,255,136,0.4);
    }
}

/* 状态指示器改进 */
.status-dot {
    transition: all 0.3s ease;
}

.status-dot.resonance {
    background: var(--accent2);
    box-shadow: 0 0 15px rgba(0,170,255,0.6);
    animation: statusPulse 0.6s ease infinite;
}

.status-dot.interference {
    background: var(--accent);
    box-shadow: 0 0 15px rgba(0,255,136,0.6);
    animation: statusPulse 0.8s ease infinite;
}

.status-dot.collapse {
    background: var(--accent3);
    box-shadow: 0 0 15px rgba(255,102,170,0.6);
    animation: statusPulse 0.5s ease infinite;
}

@keyframes statusPulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.8; }
}
`;

console.log('前端修复脚本已加载。将以上CSS添加到<style>标签中，并替换sendMessage函数为sendMessageFixed。');

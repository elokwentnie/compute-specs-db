(function () {
    'use strict';

    const EXAMPLE_PROMPTS = [
        'Which CPUs have the lowest TDP per core?',
        'Compare EPYC 7763 and EPYC 9654',
        'Which GPU has the most FP8 tensor TFLOPS?',
        'What socket and memory does Genoa use?',
    ];

    // Only links into this site are made clickable in answers.
    const SITE_PATH = /^\/(cpu|gpu|cpus|gpus|compare|about|propose|visualizations|docs|api)(\/[^\s)]*)?$/;

    // Builds answer text as DOM nodes (never innerHTML): **bold**, [label](/path) links,
    // bare site paths such as /cpu/amd-epyc-9654, and line breaks.
    function renderRichText(container, text) {
        const pattern = /\*\*([^*]+)\*\*|\[([^\]]+)\]\(([^)\s]+)\)|(\/(?:cpu|gpu|cpus|gpus|compare)\/[a-z0-9-]+(?:\/[a-z0-9-]+)?)/g;
        String(text).split('\n').forEach(function (line, index) {
            if (index > 0) container.appendChild(document.createElement('br'));
            let last = 0;
            let match;
            while ((match = pattern.exec(line)) !== null) {
                if (match.index > last) container.appendChild(document.createTextNode(line.slice(last, match.index)));
                if (match[1]) {
                    const strong = document.createElement('strong');
                    strong.textContent = match[1];
                    container.appendChild(strong);
                } else {
                    const label = match[2] || match[4];
                    const href = (match[3] || match[4]).replace(/^https:\/\/computespecsdb\.com/, '');
                    if (SITE_PATH.test(href)) {
                        const link = document.createElement('a');
                        link.href = href;
                        link.textContent = label;
                        container.appendChild(link);
                    } else {
                        container.appendChild(document.createTextNode(label));
                    }
                }
                last = pattern.lastIndex;
            }
            if (last < line.length) container.appendChild(document.createTextNode(line.slice(last)));
        });
    }

    function appendSources(sources) {
        const linked = (sources || []).filter(function (s) { return s && s.url && SITE_PATH.test(s.url); }).slice(0, 5);
        if (!linked.length) return;
        const div = document.createElement('div');
        div.className = 'ask-sources';
        div.appendChild(document.createTextNode('Sources: '));
        linked.forEach(function (source, i) {
            if (i > 0) div.appendChild(document.createTextNode(', '));
            const link = document.createElement('a');
            link.href = source.url;
            link.textContent = source.name || source.cpu_model_name || source.gpu_model_name || source.url;
            div.appendChild(link);
        });
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    const toggleBtn = document.getElementById('askToggle');
    const panel = document.getElementById('askPanel');
    const closeBtn = document.getElementById('askClose');
    const messagesEl = document.getElementById('askMessages');
    const inputEl = document.getElementById('askInput');
    const sendBtn = document.getElementById('askSend');
    const examplesEl = document.getElementById('askExamples');

    if (!toggleBtn || !panel) return;

    let isOpen = false;
    let isLoading = false;
    let hasSentMessage = false;

    function initExamples() {
        if (!examplesEl) return;
        examplesEl.innerHTML = '';
        EXAMPLE_PROMPTS.forEach(function (prompt) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'ask-example-btn';
            btn.textContent = prompt;
            btn.addEventListener('click', function () {
                inputEl.value = prompt;
                sendQuestion();
            });
            examplesEl.appendChild(btn);
        });
    }

    function openPanel() {
        isOpen = true;
        panel.classList.remove('hidden');
        inputEl.focus();
        if (!hasSentMessage) initExamples();
    }

    function closePanel() {
        isOpen = false;
        panel.classList.add('hidden');
    }

    function appendMessage(text, role) {
        const div = document.createElement('div');
        div.className = 'ask-message ask-message-' + role;
        if (role === 'assistant') renderRichText(div, text);
        else div.textContent = text;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setTyping(show) {
        let typing = document.getElementById('askTyping');
        if (show) {
            if (!typing) {
                typing = document.createElement('div');
                typing.id = 'askTyping';
                typing.className = 'ask-typing';
                typing.textContent = 'Thinking...';
                messagesEl.appendChild(typing);
            }
            messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (typing) {
            typing.remove();
        }
    }

    function setLoading(loading) {
        isLoading = loading;
        sendBtn.disabled = loading;
        inputEl.disabled = loading;
        setTyping(loading);
    }

    function hideExamples() {
        if (examplesEl) examplesEl.classList.add('hidden');
    }

    async function sendQuestion() {
        const question = inputEl.value.trim();
        if (!question || isLoading) return;

        hasSentMessage = true;
        hideExamples();
        appendMessage(question, 'user');
        inputEl.value = '';
        setLoading(true);

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });

            const data = await response.json().catch(function () {
                return {};
            });

            if (!response.ok) {
                const detail = data.detail || 'Something went wrong. Please try again.';
                const message = typeof detail === 'string' ? detail : JSON.stringify(detail);
                appendMessage(message, 'error');
                return;
            }

            appendMessage(data.answer || 'No answer received.', 'assistant');
            appendSources(data.sources);
        } catch (err) {
            appendMessage('Could not reach the server. Check your connection and try again.', 'error');
        } finally {
            setLoading(false);
            inputEl.focus();
        }
    }

    toggleBtn.addEventListener('click', function () {
        if (isOpen) closePanel();
        else openPanel();
    });

    closeBtn.addEventListener('click', closePanel);
    sendBtn.addEventListener('click', sendQuestion);

    inputEl.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });
})();

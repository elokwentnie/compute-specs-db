(function () {
    'use strict';

    const EXAMPLE_PROMPTS = [
        'How many CPUs are in the database?',
        'Which CPUs have the lowest TDP per core?',
        'Compare EPYC 7763 and EPYC 9654',
        'Which GPU has the most memory?',
    ];

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
        div.textContent = text;
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

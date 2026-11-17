// "Report an error" form on CPU and GPU spec pages.
// Sends the report to /api/reports/<kind>/<id>, which files it as a GitHub
// issue on the server, so visitors don't need a GitHub account.
(function () {
    const box = document.getElementById('report');
    const form = box && box.querySelector('.report-form');
    if (!form) return;

    const fieldSelect = form.querySelector('[name="field"]');
    const valueInput = form.querySelector('[name="suggested_value"]');
    const sourceInput = form.querySelector('[name="source_url"]');
    const currentHint = form.querySelector('[data-current-hint]');
    const result = form.querySelector('.report-result');
    const button = form.querySelector('button[type="submit"]');

    function esc(value) {
        const div = document.createElement('div');
        div.textContent = value;
        return div.innerHTML;
    }

    function openBox(focus) {
        box.open = true;
        box.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (focus) fieldSelect.focus({ preventScroll: true });
    }

    function showResult(type, html) {
        result.hidden = false;
        result.className = `report-result result ${type}`;
        result.innerHTML = html;
    }

    function fallbackLink(text) {
        return `<a href="${esc(form.dataset.fallback)}" target="_blank" rel="noopener">${text}</a>`;
    }

    function describeError(status, body) {
        if (status === 429) return 'Too many reports from your network. Please wait a minute and try again.';
        if (status === 422 && Array.isArray(body.detail)) {
            return body.detail.map(err => {
                const field = (err.loc || []).slice(-1)[0];
                if (field === 'source_url') return 'Please enter a valid http(s) link for the source, or leave it empty.';
                if (field === 'suggested_value') return 'Please enter the correct value.';
                return esc(`${field}: ${err.msg}`);
            }).join('<br>');
        }
        if (status === 502 || status === 503) {
            return `Reports can't be sent right now. Please try again later, or ${fallbackLink('report it on GitHub')} instead.`;
        }
        return esc(body.detail || `Something went wrong (HTTP ${status}).`);
    }

    function syncCurrent() {
        const option = fieldSelect.selectedOptions[0];
        const current = option && option.dataset.current;
        currentHint.textContent = current ? `Currently listed: ${current}` : '';
    }

    document.querySelectorAll('[data-open-report]').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            history.replaceState(null, '', '#report');
            openBox(true);
        });
    });
    if (location.hash === '#report') openBox(false);

    fieldSelect.addEventListener('change', syncCurrent);

    form.addEventListener('submit', async event => {
        event.preventDefault();
        const payload = {
            field: fieldSelect.value,
            suggested_value: valueInput.value.trim(),
            source_url: sourceInput.value.trim() || null,
            notes: form.querySelector('[name="notes"]').value.trim() || null,
            website: form.querySelector('[name="website"]').value,
        };

        if (!payload.field) {
            showResult('error', 'Please choose which value is wrong.');
            fieldSelect.focus();
            return;
        }
        if (!payload.suggested_value) {
            showResult('error', 'Please enter the correct value.');
            valueInput.focus();
            return;
        }
        if (payload.source_url && !/^https?:\/\/\S+\.\S+/.test(payload.source_url)) {
            showResult('error', 'Please enter a valid http(s) link for the source, or leave it empty.');
            sourceInput.focus();
            return;
        }

        button.disabled = true;
        button.textContent = 'Sending…';
        try {
            const response = await fetch(form.dataset.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const body = await response.json().catch(() => ({}));
            if (!response.ok) {
                showResult('error', describeError(response.status, body));
                return;
            }
            form.reset();
            syncCurrent();
            showResult(
                'success',
                'Thanks! Your report was sent and we will review it. ' +
                `<a href="${esc(body.issue_url)}" target="_blank" rel="noopener">Track it (#${esc(String(body.issue_number))})</a>.`
            );
        } catch (err) {
            showResult('error', `Network error. Please check your connection and try again, or ${fallbackLink('report it on GitHub')}.`);
        } finally {
            button.disabled = false;
            button.textContent = 'Send report';
        }
    });
})();

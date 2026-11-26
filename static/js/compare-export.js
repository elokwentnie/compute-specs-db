// PNG and citation downloads for spec comparisons. Used by the comparison pages
// (/compare/<pair>) and by the interactive comparison tool on /visualizations.
// Both outputs carry the page URL, so people who reuse the numbers know where they came from.
(function () {
    const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

    function cellText(cell) {
        return cell.textContent.replace(/\s+/g, ' ').trim();
    }

    // Rows of cell texts from a <table>, header row first.
    function tableRows(table) {
        return Array.from(table.querySelectorAll('tr')).map(tr => Array.from(tr.children).map(cellText));
    }

    function drawPng(rows, title, source) {
        const scale = 2;
        const pad = 24, rowH = 34, titleH = 56, footH = 48;
        const font = `15px ${FONT}`;
        const bold = `600 15px ${FONT}`;
        const sourceText = `Source: Compute Specs DB, ${source}`;
        const measure = document.createElement('canvas').getContext('2d');
        const widths = rows[0].map(() => 0);
        rows.forEach((row, r) => row.forEach((text, c) => {
            measure.font = r === 0 || c === 0 ? bold : font;
            widths[c] = Math.max(widths[c] || 0, measure.measureText(text).width);
        }));
        const cols = widths.map(w => Math.ceil(w) + 28);
        measure.font = `600 20px ${FONT}`;
        const titleW = measure.measureText(title).width;
        measure.font = `13px ${FONT}`;
        const sourceW = measure.measureText(sourceText).width + 8;
        const width = Math.ceil(Math.max(cols.reduce((a, b) => a + b, 0), titleW, sourceW, 420)) + pad * 2;
        const height = titleH + rows.length * rowH + footH + pad;

        const canvas = document.createElement('canvas');
        canvas.width = width * scale;
        canvas.height = height * scale;
        const ctx = canvas.getContext('2d');
        ctx.scale(scale, scale);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
        ctx.textBaseline = 'middle';

        ctx.fillStyle = '#111827';
        ctx.font = `600 20px ${FONT}`;
        ctx.fillText(title, pad, pad + 14);

        rows.forEach((row, r) => {
            const y = titleH + r * rowH;
            if (r === 0 || r % 2 === 0) {
                ctx.fillStyle = r === 0 ? '#eef2f7' : '#f8fafc';
                ctx.fillRect(pad, y, width - pad * 2, rowH);
            }
            let x = pad;
            row.forEach((text, c) => {
                ctx.fillStyle = c === 0 ? '#4b5563' : '#111827';
                ctx.font = r === 0 || c === 0 ? bold : font;
                ctx.fillText(text, x + 12, y + rowH / 2);
                x += cols[c];
            });
        });

        ctx.fillStyle = '#4b5563';
        ctx.font = `13px ${FONT}`;
        ctx.fillText(sourceText, pad, titleH + rows.length * rowH + 28);
        return canvas;
    }

    function savePng(rows, title, source, filename) {
        return new Promise((resolve, reject) => {
            try {
                drawPng(rows, title, source).toBlob(blob => {
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = `${filename}.png`;
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
                    resolve();
                }, 'image/png');
            } catch (err) {
                reject(err);
            }
        });
    }

    function citation(title, source) {
        const today = new Date().toISOString().slice(0, 10);
        return `${title} specifications. Compute Specs DB. ${source} (accessed ${today}).`;
    }

    // Copies the citation; resolves to a status message for the page to show.
    async function copyCitation(title, source) {
        const text = citation(title, source);
        try {
            await navigator.clipboard.writeText(text);
            return 'Citation copied.';
        } catch (err) {
            return text;  // clipboard blocked: show the text so it can be copied by hand
        }
    }

    window.SpecExport = { tableRows, drawPng, savePng, citation, copyCitation };

    // Comparison pages: wire up the buttons rendered by templates/compare.html.
    const box = document.querySelector('.compare-export[data-source]');
    const table = document.querySelector('.compare-table');
    if (!box || !table) return;

    const status = box.querySelector('.export-status');
    const say = text => {
        status.textContent = text;
        setTimeout(() => { if (status.textContent === text) status.textContent = ''; }, 4000);
    };
    const pngButton = box.querySelector('[data-export="png"]');
    const citeButton = box.querySelector('[data-export="cite"]');

    pngButton.hidden = false;
    pngButton.addEventListener('click', () => {
        savePng(tableRows(table), box.dataset.title, box.dataset.source, box.dataset.filename)
            .catch(() => say('Could not create the image in this browser.'));
    });
    citeButton.hidden = false;
    citeButton.addEventListener('click', async () => say(await copyCitation(box.dataset.title, box.dataset.source)));
})();

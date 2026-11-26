// FAQ search: filters the questions as you type and opens the ones that match.
// Without JavaScript the search box stays hidden and every question still works.
(function () {
    document.querySelectorAll('.faq').forEach(section => {
        if (section.dataset.faqReady) return;  // the script may be included more than once
        section.dataset.faqReady = '1';
        const box = section.querySelector('.faq-search');
        const input = box && box.querySelector('input');
        const empty = box && box.querySelector('.faq-empty');
        const items = Array.from(section.querySelectorAll('.faq-item'));
        if (!input || items.length < 2) return;
        box.hidden = false;

        const normalise = text => text.toLowerCase().normalize('NFKD').replace(/\s+/g, ' ');
        const texts = items.map(item => normalise(item.textContent));

        input.addEventListener('input', () => {
            const words = normalise(input.value).trim().split(' ').filter(Boolean);
            let shown = 0;
            items.forEach((item, i) => {
                const match = words.every(word => texts[i].includes(word));
                item.hidden = !match;
                // Open matches while searching; close them again when the box is cleared.
                item.open = words.length > 0 && match;
                if (match) shown += 1;
            });
            empty.hidden = shown > 0;
        });
    });
})();

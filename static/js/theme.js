(function () {
    const STORAGE_KEY = 'compute-specs-theme';

    function getSystemTheme() {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function getStoredTheme() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored === 'light' || stored === 'dark') return stored;
        } catch (_) {}
        return null;
    }

    function getEffectiveTheme() {
        return getStoredTheme() || getSystemTheme();
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        document.querySelectorAll('.theme-toggle').forEach(btn => {
            const isDark = theme === 'dark';
            btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
            btn.setAttribute('title', isDark ? 'Light mode' : 'Dark mode');
            const icon = btn.querySelector('.theme-toggle-icon');
            if (icon) icon.textContent = isDark ? '☀️' : '🌙';
        });
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
    }

    function setTheme(theme) {
        try {
            localStorage.setItem(STORAGE_KEY, theme);
        } catch (_) {}
        applyTheme(theme);
    }

    function toggleTheme() {
        setTheme(getEffectiveTheme() === 'dark' ? 'light' : 'dark');
    }

    window.Theme = { get: getEffectiveTheme, set: setTheme, toggle: toggleTheme };

    document.addEventListener('DOMContentLoaded', () => {
        applyTheme(getEffectiveTheme());

        document.querySelectorAll('.theme-toggle').forEach(btn => {
            btn.addEventListener('click', toggleTheme);
        });
    });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!getStoredTheme()) applyTheme(e.matches ? 'dark' : 'light');
    });
})();

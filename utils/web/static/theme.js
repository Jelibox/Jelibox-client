// Runs before the first paint (no flash of the wrong theme). A separate file: the page's CSP forbids inline scripts.
try { document.documentElement.dataset.theme = localStorage.getItem('jelibox-theme') || 'light'; } catch (e) { /* private mode */ }

// Small helpers shared by the pages: element builder, API calls, toast, dialogs, theme.

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}

let onSignedOut = () => {};
export function whenSignedOut(fn) { onSignedOut = fn; }

export async function api(path, method = 'GET', body, options = {}) {
  const res = await fetch('/api/' + path, {
    method, credentials: 'same-origin', keepalive: !!options.keepalive,
    headers: { 'Content-Type': 'application/json', 'X-Jelibox': '1' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (res.status === 401) onSignedOut();
  if (!res.ok) { const err = new Error(data.error || res.statusText); err.status = res.status; err.data = data; throw err; }
  return data;
}

export function toast(text) {
  let box = document.getElementById('toasts');
  if (!box) { box = h('div', { id: 'toasts' }); document.body.append(box); }
  const el = h('div', { class: 'toast' }, text);
  box.append(el);
  setTimeout(() => el.remove(), 3200);
}

export function theme() { return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'; }
export function setTheme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('jelibox-theme', next); } catch (e) { /* private mode */ }
}

// sticky: a click outside does not close it (an upload or a choice in progress)
export function dialog(title, content, sticky = false) {
  const overlay = h('div', { class: 'overlay', onmousedown: (e) => { if (!sticky && e.target === overlay) overlay.remove(); } },
    h('div', { class: 'dialog' }, h('h3', {}, title), content));
  document.body.append(overlay);
  return overlay;
}

export function notice(title, message) {
  return new Promise((resolve) => {
    const body = h('div', {}, h('p', { style: 'white-space:pre-line' }, message));
    const overlay = dialog(title, body, true);
    const done = () => { overlay.remove(); resolve(); };
    body.append(h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: done }, 'OK')));
  });
}

export function confirmBox(title, message, okText = 'OK', danger = false) {
  return new Promise((resolve) => {
    const body = h('div', {}, h('p', { style: 'white-space:pre-line' }, message));
    const overlay = dialog(title, body, true);
    const done = (v) => { overlay.remove(); resolve(v); };
    body.append(h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => done(false) }, 'Cancel'),
      h('button', { class: 'btn ' + (danger ? 'danger' : 'primary'), onclick: () => done(true) }, okText)));
  });
}

export function fmtTime(s) { s = Math.round(s); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }

// A progress window for a job that is already started (the answer of a 202). Resolves with its final state.
// `stoppable`: a Stop button asks the server to end it (the job decides where it can stop).
export function watchJob(title, started, stoppable = false) {
  const text = h('p', { class: 'muted' }, started.text || 'Working ...');
  const bar = h('div', { class: 'bar' }, h('i', {}));
  const body = h('div', {}, text, bar);
  const overlay = dialog(title, body, true);
  const fill = bar.firstChild;
  let stopBtn = null;
  if (stoppable) {
    stopBtn = h('button', { class: 'btn', onclick: async () => {
      stopBtn.disabled = true; stopBtn.textContent = 'Stopping ...';
      try { await api('jobs/' + started.id + '/cancel', 'POST', {}); } catch (e) { /* it ended meanwhile */ }
    } }, 'Stop');
    body.append(h('div', { class: 'row' }, stopBtn));
  }
  return new Promise((resolve) => {
    async function tick() {
      let job;
      try { job = await api('jobs/' + started.id); } catch (err) { overlay.remove(); resolve({ state: 'error', error: err.message }); return; }
      text.textContent = job.text + (job.total ? `  (${job.done} / ${job.total}, ${fmtTime(job.seconds)})` : '');
      bar.classList.toggle('busy', !job.total);
      fill.style.width = job.total ? Math.min(100, 100 * job.done / job.total) + '%' : '';
      if (job.state === 'running') { setTimeout(tick, 400); return; }
      overlay.remove(); resolve(job);
    }
    tick();
  });
}

// Browser for the file system of the SERVER. Folder mode: "Use this folder". File mode (`ext`, e.g. '.pt'): click a file.
// Resolves with the chosen path, or null.
export function pickServer({ start = '', ext = '' } = {}) {
  return new Promise((resolve) => {
    const list = h('div', { class: 'folders' });
    const here = h('code', { class: 'here' });
    let current = start || '';
    const overlay = dialog(ext ? `Choose a ${ext} file on the server` : 'Folder on the server', h('div', {}, here, list), true);
    const close = (value) => { overlay.remove(); resolve(value); };
    const join = (base, name) => base.replace(/[\\/]+$/, '') + (base.includes('\\') ? '\\' : '/') + name;
    async function open(path) {
      let data;
      try { data = await api('folders?path=' + encodeURIComponent(path) + (ext ? '&ext=' + encodeURIComponent(ext) : '')); }
      catch (err) { toast(err.message); return; }
      current = data.path;
      here.textContent = data.path || 'Start places';
      list.replaceChildren(
        ...(data.path ? [h('div', { class: 'folder', onclick: () => open(data.parent) }, '⬑  ..')] : []),
        ...(data.path ? [] : data.roots.map((r) => h('div', { class: 'folder', onclick: () => open(r) }, '📁  ' + r))),
        ...data.folders.map((n) => h('div', { class: 'folder', onclick: () => open(join(data.path, n)) }, '📁  ' + n)),
        ...(data.files || []).map((n) => h('div', { class: 'folder file', onclick: () => close(join(data.path, n)) }, '📄  ' + n)),
        ...(data.path && !data.folders.length && !(data.files || []).length ? [h('p', { class: 'muted', style: 'margin:8px' }, ext ? 'No subfolders and no ' + ext + ' files.' : 'No subfolders.')] : []));
    }
    overlay.firstChild.append(h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => close(null) }, 'Cancel'),
      ext ? null : h('button', { class: 'btn primary', onclick: () => current ? close(current) : toast('Open a folder first.') }, 'Use this folder')));
    open(current);
  });
}


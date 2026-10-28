// Jelibox web page: sign in, workspace list, mode menu, users. Plain JavaScript, nothing to build.

const app = document.getElementById('app');
const state = { user: null, workspaces: [], selected: null, open: new Set(), mode: null, modes: [] };

import { h, api, toast, dialog, theme, setTheme, whenSignedOut, watchJob, pickServer } from './ui.js';
import { openEditor } from './editor.js';
import { openAnalyze } from './analyze.js';
import { monitorPanel } from './sysmon.js';

whenSignedOut(() => { if (state.user) { state.user = null; render(); } });

function toggleTheme() {
  setTheme(theme() === 'dark' ? 'light' : 'dark');
  render();
}

// ------------------------------------------------------------------ sign in
// Sign in and "choose a new password" share one layout: a brand panel on the left, the form on the right.
function authShell(form) {
  const ghost = (cls, text) => h('div', { class: 'ghost-box ' + cls }, h('span', {}, text));
  const points = ['Boxes and polygons in a fast, keyboard-friendly editor',
    'Label Assistant: YOLO-World, SAM 2, LocateAnything or your own model',
    'Train, analyze and export, straight to your device'];
  return h('div', { class: 'auth' },
    h('aside', { class: 'auth-art' },
      h('div', { class: 'auth-brand' }, h('img', { src: '/static/jelibox.svg', alt: '' }), h('b', {}, 'Jelibox')),
      h('div', { class: 'auth-scene', 'aria-hidden': 'true' }, ghost('g1', 'boots  0.94'), ghost('g2', 'helmet  0.91'), ghost('g3', 'person  0.97')),
      h('div', { class: 'auth-copy' },
        h('h1', {}, 'Soft to use. Sharp on every object.'),
        h('p', {}, 'Annotate, auto-label, train and export your computer-vision datasets from any browser on your network.'),
        h('ul', {}, ...points.map((t) => h('li', {}, t))))),
    h('main', { class: 'auth-form' },
      h('button', { class: 'btn ghost auth-theme', type: 'button', onclick: () => { setTheme(theme() === 'dark' ? 'light' : 'dark'); render(); } },
        theme() === 'dark' ? 'Light' : 'Dark'),
      form,
      h('p', { class: 'auth-foot' }, 'Jelibox runs on your own computer. Your pictures never leave it.')));
}

// A password box with a Show / Hide button
function passwordBox(autocomplete) {
  const input = h('input', { type: 'password', autocomplete });
  const toggle = h('button', { type: 'button', class: 'pw-toggle', tabindex: '-1', onclick: () => {
    const hidden = input.type === 'password';
    input.type = hidden ? 'text' : 'password';
    toggle.textContent = hidden ? 'Hide' : 'Show';
  } }, 'Show');
  return { input, node: h('div', { class: 'pw' }, input, toggle) };
}

function loginView() {
  const error = h('div', { class: 'error', hidden: true });
  const user = h('input', { type: 'text', autocomplete: 'username', autofocus: true, placeholder: 'admin' });
  const pass = passwordBox('current-password');
  async function submit(e) {
    e.preventDefault();
    error.hidden = true;
    try {
      const data = await api('login', 'POST', { username: user.value, password: pass.input.value });
      state.user = data.user;
      await start();
    } catch (err) { error.textContent = err.message; error.hidden = false; pass.input.value = ''; pass.input.focus(); }
  }
  return authShell(h('form', { class: 'auth-card', onsubmit: submit },
    h('h2', {}, 'Welcome back'), h('p', { class: 'lead' }, 'Sign in to your Jelibox workspace.'),
    h('label', {}, 'User name'), user, h('label', {}, 'Password'), pass.node, error,
    h('button', { class: 'btn primary wide', type: 'submit' }, 'Sign in')));
}

function passwordView(forced) {
  const error = h('div', { class: 'error', hidden: true });
  const old = passwordBox('current-password');
  const fresh = passwordBox('new-password');
  const again = passwordBox('new-password');
  async function submit(e) {
    e.preventDefault();
    error.hidden = true;
    if (fresh.input.value !== again.input.value) { error.textContent = 'The two new passwords are not the same.'; error.hidden = false; return; }
    try {
      const data = await api('password', 'POST', { old: old.input.value, new: fresh.input.value });
      state.user = data.user;
      toast('Password changed.');
      await start();
    } catch (err) { error.textContent = err.message; error.hidden = false; }
  }
  const form = h('form', { class: forced ? 'auth-card' : 'card', onsubmit: submit },
    h('h2', {}, 'Choose a new password'),
    h('p', { class: 'lead' }, forced ? `Welcome, ${state.user.name}. Replace the temporary password before you continue.` : ''),
    h('label', {}, 'Current password'), old.node, h('label', {}, 'New password (6 characters or more)'), fresh.node,
    h('label', {}, 'New password again'), again.node, error,
    h('div', { class: 'row' }, h('button', { class: 'btn primary', type: 'submit' }, 'Save password'),
      forced ? h('button', { class: 'btn', type: 'button', onclick: signOut }, 'Sign out') : null));
  return forced ? authShell(form) : h('div', {}, form);
}

async function signOut() {
  try { await api('logout', 'POST', {}); } catch (e) { /* already out */ }
  state.user = null; render();
}

// ------------------------------------------------------------------ workspaces
function modeTitle() { const m = state.modes.find((x) => x.id === state.mode); return m ? m.title : ''; }

function chooseMode() {
  const box = h('div', {});
  const overlay = dialog('Choose the Label Assistant mode', box);
  for (const m of state.modes) {
    box.append(h('button', {
      class: 'mode' + (m.id === state.mode ? ' on' : ''),
      onclick: async () => {
        try { const data = await api('mode', 'PUT', { mode: m.id }); state.mode = data.mode; overlay.remove(); render(); }
        catch (err) { toast(err.message); }
      },
    }, h('b', {}, m.title), h('span', {}, m.summary), h('em', {}, m.needs)));
  }
}

async function refresh() {
  try { state.workspaces = (await api('workspaces/recount', 'POST', {})).workspaces; toast('Counted.'); render(); }
  catch (err) { toast(err.message); }
}

function manageUsers() {
  const box = h('div', {});
  const overlay = dialog('Users', box);
  async function draw() {
    const { users } = await api('users');
    const name = h('input', { type: 'text', placeholder: 'user name' });
    const pass = h('input', { type: 'password', placeholder: 'password', autocomplete: 'new-password' });
    const error = h('div', { class: 'error', hidden: true });
    box.replaceChildren(
      h('table', { class: 'users' }, users.map((u) => h('tr', {},
        h('td', {}, u.name, u.admin ? h('span', { class: 'badge' }, 'admin') : null),
        h('td', { style: 'text-align:right' },
          h('button', { class: 'btn small', onclick: async () => {
            const p = prompt(`New temporary password for ${u.name} (they must change it at the next sign in):`);
            if (!p) return;
            try { await api('users/' + encodeURIComponent(u.name), 'POST', { password: p }); toast('Password reset.'); } catch (e) { toast(e.message); }
          } }, 'Reset password'), ' ',
          h('button', { class: 'btn small danger', disabled: u.name === state.user.name, onclick: async () => {
            if (!confirm(`Delete the user ${u.name}?`)) return;
            try { await api('users/' + encodeURIComponent(u.name), 'DELETE'); draw(); } catch (e) { toast(e.message); }
          } }, 'Delete')))))
      , h('label', {}, 'New user'), name, pass, error,
      h('div', { class: 'row' },
        h('button', { class: 'btn primary', onclick: async () => {
          try { await api('users', 'POST', { username: name.value, password: pass.value }); draw(); }
          catch (e) { error.textContent = e.message; error.hidden = false; }
        } }, 'Add user'),
        h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close')));
  }
  draw().catch((e) => { box.textContent = e.message; });
}

function workspaceTree() {
  const side = h('div', { class: 'side' }, h('h4', {}, 'WORKSPACES'));
  if (!state.workspaces.length) side.append(h('p', { class: 'muted', style: 'margin:10px' }, 'No workspace yet. Use "+ Add Workspace".'));
  const stop = (fn) => (e) => { e.stopPropagation(); fn(); };
  for (const ws of state.workspaces) {
    const open = state.open.has(ws.name);
    side.append(h('div', { class: 'ws' },
      h('div', {
        class: 'ws-row' + (state.selected === ws.name ? ' sel' : ''),
        onclick: () => {
          state.selected = ws.name;
          if (state.open.has(ws.name)) state.open.delete(ws.name); else state.open.add(ws.name);
          render();
        },
      }, h('span', { class: 'caret' }, open ? '▾' : '▸'), h('span', { class: 'name' }, ws.name),
        h('button', { class: 'icon', title: 'Add a dataset to this workspace', onclick: stop(() => addWorkspace(ws.name)) }, '+'),
        h('button', { class: 'icon danger', title: 'Delete this workspace', onclick: stop(() => confirmDelete(ws.name)) }, '🗑'),
        h('span', { class: 'count' }, ws.counted ? ws.images : '')),
      open ? ws.instances.map((i) => h('div', { class: 'inst-row', title: 'Open in the annotation editor', onclick: () => { location.hash = '#/ws/' + encodeURIComponent(i.name); } },
        h('span', { class: 'name' }, i.name),
        h('button', { class: 'icon danger', title: 'Delete this dataset', onclick: stop(() => confirmDelete(ws.name, i.name)) }, '🗑'),
        h('span', { class: 'count' }, i.images == null ? 'not counted' : i.images))) : null));
  }
  return side;
}

// ------------------------------------------------------------------ jobs (progress window)
async function reloadWorkspaces() { state.workspaces = (await api('workspaces')).workspaces; render(); }

async function confirmDelete(name, instance) {
  let info;
  try { info = await api('workspaces/' + encodeURIComponent(name) + '/stats'); } catch (err) { toast(err.message); return; }
  const what = instance ? `the dataset "${instance}"` : `the whole workspace "${name}"`;
  const detail = instance
    ? 'Its pictures and the annotations that belong only to them are deleted.'
    : `${info.instances.length} dataset folder(s), ${info.image_count} picture(s), ${info.xml_count} annotation file(s) and the class list are deleted. Trained models and exports are kept.`;
  const box = h('div', {}, h('p', {}, `Delete ${what}?`), h('p', { class: 'muted' }, detail),
    h('div', { class: 'note' }, 'This cannot be undone. Make a copy first if you may need it again.'));
  const overlay = dialog('Delete', box);
  box.append(h('div', { class: 'row' },
    h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Cancel'),
    h('button', { class: 'btn danger', onclick: async () => {
      overlay.remove();
      try {
        const path = 'workspaces/' + encodeURIComponent(name) + (instance ? '/instances/' + encodeURIComponent(instance) : '');
        const job = await watchJob('Deleting', await api(path, 'DELETE'));
        if (job.state === 'error') toast(job.error);
        await reloadWorkspaces();
      } catch (err) { toast(err.message); }
    } }, 'Delete')));
}

const pickServerFolder = (start) => pickServer({ start });

// ------------------------------------------------------------------ Add Workspace
const UPLOAD_TYPES = /\.(jpe?g|png|bmp|tiff?|webp|mp4|avi|mov|mkv|webm|m4v|wmv|flv|mpe?g|xml|txt|json|ya?ml|zip)$/i;

async function uploadFiles(files, progress) {
  const { id } = await api('uploads', 'POST', {});
  const total = files.reduce((n, f) => n + f.file.size, 0);
  let done = 0, bytes = 0;
  const queue = files.slice();
  async function worker() {
    while (queue.length) {
      const f = queue.shift();
      const res = await fetch(`/api/uploads/${id}/file?path=${encodeURIComponent(f.path)}`, {
        method: 'PUT', credentials: 'same-origin', body: f.file,
        headers: { 'X-Jelibox': '1', 'Content-Type': 'application/octet-stream' },
      });
      if (!res.ok) { let m = res.statusText; try { m = (await res.json()).error || m; } catch (e) { /* not JSON */ } throw new Error(`${f.path}: ${m}`); }
      done += 1; bytes += f.file.size;
      progress(done, files.length, bytes, total);
    }
  }
  try { await Promise.all([worker(), worker(), worker()]); }
  catch (err) { api('uploads/' + id, 'DELETE').catch(() => {}); throw err; }
  return id;
}

function mb(n) { return n > 1e9 ? (n / 1e9).toFixed(1) + ' GB' : (n / 1e6).toFixed(n > 1e7 ? 0 : 1) + ' MB'; }

function addWorkspace(existing) {
  const src = { kind: null, folder: null, upload: null, scan: null };    // what was chosen
  const name = h('input', { type: 'text', placeholder: 'e.g. weapon', value: existing || '', disabled: !!existing, maxlength: 40 });
  const status = h('div', { class: 'note', hidden: true });
  const classes = h('input', { type: 'text', placeholder: 'cat, dog, rabbit  (optional)' });
  const prefix = h('input', { type: 'text', placeholder: 'optional, e.g. test-  (renames the pictures test-1, test-2 ...)' });
  const skip = h('input', { type: 'number', min: 1, max: 10000, value: 10 });
  const skipNote = h('p', { class: 'muted' });
  const videoBox = h('div', { hidden: true }, h('label', {}, 'Skip frames (video files: keep every N-th frame)'), skip, skipNote);
  const classBox = h('div', {}, h('label', {}, 'Classes (separated by commas)'), classes);
  const folderInput = h('input', { type: 'file', webkitdirectory: true, multiple: true, hidden: true });
  const filesInput = h('input', { type: 'file', multiple: true, hidden: true, accept: '.zip,image/*,video/*,.xml,.txt,.json,.yaml,.yml' });
  const create = h('button', { class: 'btn primary', disabled: true }, 'Create');
  let overlay;

  function say(text, kind) { status.textContent = text; status.className = kind === 'error' ? 'error' : 'note'; status.hidden = !text; }

  function frameNote() {
    const sc = src.scan;
    if (!sc || !sc.videos) return;
    const n = parseInt(skip.value, 10);
    skipNote.textContent = n >= 1 && sc.video_frames ? `About ${Math.ceil(sc.video_frames / n)} pictures from ${sc.videos} video(s).` : '';
  }
  skip.addEventListener('input', frameNote);

  async function scan(body) {
    say('Reading the folder ...');
    create.disabled = true;
    try {
      const sc = await api('scan', 'POST', { ...body, workspace: existing || name.value.trim() || undefined });
      src.scan = sc;
      if (sc.error) { say(sc.error, 'error'); return; }
      const label = { voc: 'Pascal VOC', yolo: 'YOLO', coco: 'COCO' }[sc.format];
      const found = Object.keys(sc.classes);
      const parts = [];
      if (sc.images) parts.push(sc.format ? `${sc.images} picture(s), ${sc.annotation_files} ${label} annotation file(s), ${found.length} class(es): ${found.slice(0, 8).join(', ')}${found.length > 8 ? ' ...' : ''}` : `${sc.images} picture(s), no annotations: they are added unannotated.`);
      if (sc.videos) parts.push(`${sc.videos} video(s): their frames become pictures without annotations.`);
      if (sc.yolo_unnamed) parts.push('⚠ No class name file (classes.txt / obj.names / data.yaml): the classes import as class_0, class_1 ...');
      if (sc.duplicates) parts.push(`⚠ ${sc.duplicates} picture(s) share a file name with another one: only one of each name is imported.`);
      if (sc.locked_classes) parts.push(`This workspace keeps its classes (${sc.locked_classes.join(', ')}).` + (sc.skipped_classes.length ? ` ⚠ Annotations of ${sc.skipped_classes.join(', ')} are skipped.` : ''));
      say(parts.join('\n'));
      status.style.whiteSpace = 'pre-line';
      classBox.hidden = !!sc.locked_classes;
      if (!sc.locked_classes && found.length && !sc.yolo_unnamed && !classes.value.trim()) classes.value = found.join(', ');
      videoBox.hidden = !sc.videos;
      frameNote();
      create.disabled = false;
    } catch (err) { say(err.message, 'error'); }
  }

  async function chooseUpload(fileList) {
    const files = [...fileList].map((file) => ({ file, path: file.webkitRelativePath || file.name })).filter((f) => UPLOAD_TYPES.test(f.file.name));
    if (!files.length) { say('Nothing Jelibox can use in that selection (pictures, videos, annotation files or a zip).', 'error'); return; }
    await discard();
    create.disabled = true;
    try {
      const id = await uploadFiles(files, (n, all, b, total) => say(`Uploading ${n} / ${all} files  (${mb(b)} of ${mb(total)})`));
      src.kind = 'upload'; src.upload = id; src.folder = null;
      await scan({ upload_id: id });
    } catch (err) { say(err.message, 'error'); }
  }

  async function chooseServer() {
    const folder = await pickServerFolder('');
    if (!folder) return;
    await discard();
    src.kind = 'server'; src.folder = folder; src.upload = null;
    await scan({ folder });
  }

  async function discard() {
    if (src.upload) { api('uploads/' + src.upload, 'DELETE').catch(() => {}); src.upload = null; }
  }

  folderInput.addEventListener('change', () => folderInput.files.length && chooseUpload(folderInput.files));
  filesInput.addEventListener('change', () => filesInput.files.length && chooseUpload(filesInput.files));

  create.addEventListener('click', async () => {
    const body = {
      name: existing ? undefined : name.value.trim(), workspace: existing || undefined,
      folder: src.folder || undefined, upload_id: src.upload || undefined, prefix: prefix.value.trim(),
      skip_frames: parseInt(skip.value, 10) || 0,
      classes: classes.value.split(',').map((c) => c.trim()).filter(Boolean),
    };
    try {
      const started = await api('import', 'POST', body);
      src.upload = null;                                  // the server removes the staging folder itself
      overlay.remove();
      const job = await watchJob(started.title, started);
      if (job.state === 'error') { toast(job.error); } else {
        const r = job.result;
        toast(`Added ${r.image_count} picture(s) to ${r.instance_name}` + (r.annotated_count ? `, ${r.annotated_count} annotated` : '') + '.');
      }
      await reloadWorkspaces();
    } catch (err) { say(err.message, 'error'); }
  });

  const body = h('div', {},
    h('label', {}, existing ? 'Workspace' : 'Workspace name'), name,
    h('label', {}, 'Dataset'),
    h('div', { class: 'row', style: 'margin-top:0' },
      h('button', { class: 'btn', onclick: () => folderInput.click() }, 'Upload a folder'),
      h('button', { class: 'btn', onclick: () => filesInput.click() }, 'Upload files or a .zip'),
      h('button', { class: 'btn', onclick: chooseServer }, 'Folder on the server')),
    h('p', { class: 'muted', style: 'font-size:12px' }, 'Upload sends the pictures from this device to the server. "Folder on the server" reads a folder that is already on the computer running Jelibox. Pascal VOC, YOLO and COCO are detected by themselves.'),
    folderInput, filesInput, status, classBox, videoBox,
    h('label', {}, 'File name prefix'), prefix);
  overlay = dialog(existing ? `Add a dataset to "${existing}"` : 'Add Workspace', body, true);
  overlay.firstChild.append(h('div', { class: 'row' },
    h('button', { class: 'btn', onclick: async () => { await discard(); overlay.remove(); } }, 'Cancel'), create));
}

function workspacesView() {
  const analyzable = state.workspaces.some((w) => w.name === state.selected);
  return h('div', { style: 'height:100%' },
    h('div', { class: 'header' },
      h('div', { class: 'brand' }, h('img', { src: '/static/jelibox.svg', alt: '' }),
        h('div', {}, h('b', {}, 'Jelibox'), h('small', {}, 'WORKSPACES'))),
      h('button', { class: 'btn', onclick: chooseMode }, 'Mode: ' + modeTitle()),
      h('button', { class: 'btn', onclick: refresh }, 'Refresh'),
      h('button', { class: 'btn primary', onclick: () => addWorkspace(null) }, '+ Add Workspace'),
      h('button', {
        class: 'btn', disabled: !analyzable, title: analyzable ? `Charts and numbers of "${state.selected}"` : 'Select a workspace first',
        onclick: () => openAnalyze(state.selected),
      }, 'Analyze'),
      h('div', { class: 'spacer' }),
      h('span', { class: 'who' }, state.user.name),
      state.user.admin ? h('button', { class: 'btn', onclick: manageUsers }, 'Users') : null,
      h('button', { class: 'btn', onclick: () => passwordDialog() }, 'Password'),
      h('button', { class: 'btn', onclick: toggleTheme }, theme() === 'dark' ? 'Light' : 'Dark'),
      h('button', { class: 'btn ghost', onclick: signOut }, 'Sign out')),
    h('div', { class: 'shell' }, workspaceTree(),
      h('div', { class: 'stage' }, monitorPanel(), h('div', { class: 'welcome' },
        h('img', { src: '/static/jelibox.svg', alt: '' }), h('h1', {}, 'Jelibox'),
        h('p', {}, 'Soft to use. Sharp on every object.'),
        h('p', {}, 'Select a workspace on the left, then pick a dataset to start annotating.')))));
}

function passwordDialog() {
  const view = passwordView(false);
  const overlay = dialog('Change password', view.querySelector('form'));
  overlay.querySelector('form').classList.remove('card');
}

// ------------------------------------------------------------------ boot
async function start() {
  document.querySelectorAll('.overlay').forEach((o) => o.remove());
  if (state.user && state.user.must_change) { render(); return; }
  if (state.user) {
    try {
      state.workspaces = (await api('workspaces')).workspaces;
      const m = await api('mode');
      state.mode = m.mode; state.modes = m.modes;
    } catch (err) { toast(err.message); }
  }
  render();
}

let leaveEditor = null;

function render() {
  if (leaveEditor) { leaveEditor(); leaveEditor = null; }
  if (!state.user || state.user.must_change) {
    app.replaceChildren(!state.user ? loginView() : passwordView(true));
    return;
  }
  const match = /^#\/ws\/([^/]+)$/.exec(location.hash);
  if (match) {
    const instance = decodeURIComponent(match[1]);
    app.replaceChildren();
    leaveEditor = openEditor(app, instance, {
      home: () => { location.hash = ''; },
      toggleTheme: () => { setTheme(theme() === 'dark' ? 'light' : 'dark'); },
      user: state.user,
    });
    return;
  }
  app.replaceChildren(workspacesView());
}

window.addEventListener('hashchange', () => { if (state.user) render(); });

(async function boot() {
  try { state.user = (await api('me')).user; } catch (e) { state.user = null; }
  await start();
})();

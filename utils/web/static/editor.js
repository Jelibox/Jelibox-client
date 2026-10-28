// The annotation editor: the same tools, shortcuts and files as the desktop window, in a browser.
// Coordinates of boxes and polygons are ALWAYS original picture pixels; only drawing converts to the screen.

import { h, api, toast, dialog, confirmBox } from './ui.js';
import { runAssist, inferDialog, settingsDialog } from './assistant.js';
import { openExportDataset, openExportModel } from './export.js';
import { openTrain } from './train.js';

const MIN_BOX = 8;                 // pixels of the original picture (smaller boxes are ignored, like the desktop)
const HISTORY_LIMIT = 200;
const AUTOSAVE_MS = 2500;
const enc = encodeURIComponent;

export function openEditor(root, instance, env) {
  const S = {
    instance, images: [], annotated: new Set(), classes: [], idx: 0, name: null, img: null, imgVersion: 1,
    w: 0, h: 0, version: '0', bboxes: [], polygons: [], masks: [],
    curClass: null, mode: 'bbox', force: false, maskMode: false,
    selBox: null, selBoxes: new Set(), selPoly: null, selPt: null, preview: [], maskPts: [],
    hidden: new Set(), showText: true, undo: [], redo: [], prev: null, unreviewed: new Set(),
    view: { scale: 1, ox: 0, oy: 0 }, mouse: null, drag: null, space: false,
    dirty: false, saveState: 'idle', timer: null, busy: false, closed: false, filter: '', gen: 0,
  };
  let saveChain = Promise.resolve(true);
  let frame = 0;

  // ================================================================== layout
  const canvas = h('canvas', { class: 'ed-canvas', tabindex: 0 });
  const ctx = canvas.getContext('2d');
  const stage = h('div', { class: 'ed-stage' }, canvas,
    h('div', { class: 'ed-zoom' },
      h('button', { class: 'btn small', title: 'Zoom out', onclick: () => zoomBy(1 / 1.25) }, '−'),
      h('button', { class: 'btn small', title: 'Fit the picture (F)', onclick: fit }, 'Fit'),
      h('button', { class: 'btn small', title: 'Zoom in', onclick: () => zoomBy(1.25) }, '+')));

  function hb(label, onclick, opts = {}) {
    const b = h('button', { class: 'btn' + (opts.cls ? ' ' + opts.cls : ''), title: opts.title || '', onclick }, label);
    return b;
  }
  const bUndo = hb('Undo', () => undo(), { title: 'Ctrl+Z' });
  const bRedo = hb('Redo', () => redo(), { title: 'Ctrl+Y or Ctrl+Shift+Z' });
  const bMode = hb('Mode', () => toggleMode(), { title: 'Switch between box and polygon (M)' });
  const bMask = hb('ZeroFill: OFF', () => toggleMask(), { title: 'Black out regions of the picture (written to the file when you leave it)' });
  const header = h('div', { class: 'header ed-header' },
    h('button', { class: 'btn', title: 'Back to the workspaces (Esc)', onclick: () => leave() }, '⌂'),
    h('div', { class: 'brand' }, h('img', { src: '/static/jelibox.svg', alt: '' }),
      h('div', {}, h('b', {}, 'Jelibox'), h('small', { class: 'ed-title' }, instance))),
    hb('◀ Prev', () => step(-1), { title: 'A or ←' }), hb('Next ▶', () => step(1), { title: 'D or →' }),
    bUndo, bRedo, hb('Repeat', () => repeat(), { title: 'Copy the annotations of the previous picture here (E)' }),
    hb('Delete', () => deleteImage(), { cls: 'danger', title: 'Delete this picture and its annotations (Del)' }),
    bMode, bMask,
    h('div', { class: 'spacer' }),
    hb('Infer', () => inferDialog(ed), { cls: 'primary', title: 'Label this picture or the whole dataset with the Label Assistant (G)' }),
    hb('Label Assistant', () => settingsDialog(ed), { title: 'Settings of the Label Assistant mode chosen on the workspace page' }),
    hb('Train', () => openTrain(ed), { title: 'Train a model on this workspace (T)' }),
    hb('Export Model', () => openExportModel(ed), { title: 'Export the trained model (ONNX, TensorRT ...)' }),
    hb('Export Dataset', () => openExportDataset(ed), { title: 'Export the dataset as YOLO, Pascal VOC or COCO' }),
    h('button', { class: 'btn', onclick: (e) => { env.toggleTheme(); e.target.textContent = document.documentElement.dataset.theme === 'dark' ? 'Light' : 'Dark'; } },
      document.documentElement.dataset.theme === 'dark' ? 'Light' : 'Dark'));

  const classList = h('div', { class: 'ed-classes' });
  const visList = h('div', { class: 'ed-vis' });
  const textBtn = h('button', { class: 'btn small', onclick: () => { S.showText = !S.showText; textBtn.textContent = S.showText ? 'Hide label text' : 'Show label text'; redraw(); } }, 'Hide label text');
  const left = h('div', { class: 'ed-left' },
    h('div', { class: 'ed-sec' }, h('span', {}, 'CLASSES'), h('span', { class: 'spacer' }),
      h('button', { class: 'btn small primary', title: 'Add a class', onclick: () => addClassDialog() }, '+'),
      h('button', { class: 'btn small danger', title: 'Delete the selected class', onclick: () => deleteClassDialog() }, '−')),
    classList,
    h('div', { class: 'ed-sec' }, h('span', {}, 'VISIBILITY'), h('span', { class: 'spacer' }), textBtn),
    visList);

  const info = h('pre', { class: 'ed-info' });
  const search = h('input', { type: 'text', placeholder: 'Search pictures', class: 'ed-search', oninput: () => { S.filter = search.value.toLowerCase(); renderImages(); } });
  const imageList = h('div', { class: 'ed-images' });
  const right = h('div', { class: 'ed-right' },
    h('div', { class: 'ed-sec' }, h('span', {}, 'INFORMATION')), info,
    h('div', { class: 'ed-sec' }, h('span', {}, 'IMAGES')), search, imageList);

  const modeBadge = h('span', { class: 'ed-badge' });
  const forceLabel = h('span', { class: 'ed-force' });
  const progressLabel = h('span', { class: 'ed-progress' });
  const saveLabel = h('span', { class: 'ed-save' });
  const status = h('div', { class: 'ed-status' }, modeBadge, forceLabel, progressLabel, saveLabel, h('span', { class: 'spacer' }),
    h('span', { class: 'ed-hints' }, 'A/D: Navigate · B: BBox · M: Mode · E: Repeat · R: Remove · S: Change class · Del: Remove img · Esc: Exit'));

  root.append(h('div', { class: 'ed' }, header, h('div', { class: 'ed-body' }, left, stage, right), status));

  // ================================================================== small helpers
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const colorOf = (cls) => (S.classes.find((c) => c.name === cls) || S.classes[0] || { color: '#e5566d' }).color;
  const snap = () => JSON.stringify({ b: S.bboxes, p: S.polygons, m: S.masks });
  const inField = (e) => /^(INPUT|TEXTAREA|SELECT)$/.test((e.target || {}).tagName || '');
  const modalOpen = () => !!document.querySelector('.overlay');
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  function say(text) { toast(text); }

  function clearSelection() { S.selBox = null; S.selBoxes = new Set(); S.selPoly = null; S.selPt = null; }
  function hasSelection() { return S.selBox !== null || S.selBoxes.size > 0 || S.selPoly !== null; }

  // ================================================================== history and saving
  function pushUndo() {
    S.unreviewed.delete(S.name);                           // you are editing: it is yours now, SAM 2 may learn from it
    S.undo.push(snap());
    if (S.undo.length > HISTORY_LIMIT) S.undo.shift();
    S.redo = [];
  }
  function restore(text) {
    const o = JSON.parse(text);
    S.bboxes = o.b; S.polygons = o.p; S.masks = o.m;
    clearSelection(); S.preview = []; S.maskPts = [];
    touch(false);
  }
  function undo() { if (S.undo.length) { S.redo.push(snap()); restore(S.undo.pop()); } }
  function redo() { if (S.redo.length) { S.undo.push(snap()); restore(S.redo.pop()); } }

  // An edit was made: save soon, redraw, refresh the panels.
  function touch(markDirty = true) {
    if (markDirty) { S.dirty = true; S.saveState = 'unsaved'; scheduleSave(); }
    redraw(); updatePanels();
  }
  function scheduleSave() {
    clearTimeout(S.timer);
    S.timer = setTimeout(() => { if (!S.drag && !S.closed) saveNow(); }, AUTOSAVE_MS);
  }

  function payload(force = false) {
    return { bboxes: S.bboxes, polygons: S.polygons, masks: S.masks, version: S.version, force, reviewed: !S.unreviewed.has(S.name) };
  }

  // Saves the open picture. `always` writes even when nothing changed and the picture was already saved once
  // (leaving a picture marks it as done, like the desktop window). Resolves true when it is safe to go on.
  function saveNow(always = false) {
    saveChain = saveChain.then(() => doSave(always)).catch(() => false);
    return saveChain;
  }

  async function doSave(always) {
    if (!S.name || S.closed) return true;
    if (!S.dirty && !always) return true;
    if (!S.dirty && S.annotated.has(S.name) && !S.masks.length) return true;
    clearTimeout(S.timer);
    const name = S.name;
    S.saveState = 'saving'; updateStatus();
    try {
      let res;
      try {
        res = await api(`ws/${enc(S.instance)}/annotations/${enc(name)}`, 'PUT', payload());
      } catch (err) {
        if (!err.data || !err.data.conflict) throw err;
        const mine = await conflictChoice();
        if (mine === 'cancel') { S.saveState = 'unsaved'; updateStatus(); return false; }
        if (mine === 'theirs') { await reloadAnnotations(); S.dirty = false; S.saveState = 'saved'; updateStatus(); return true; }
        res = await api(`ws/${enc(S.instance)}/annotations/${enc(name)}`, 'PUT', payload(true));
      }
      if (name !== S.name) return true;                     // moved on while saving (cannot happen: navigation waits)
      S.version = res.version; S.dirty = false; S.saveState = 'saved';
      S.annotated.add(name);
      S.prev = clone({ bboxes: S.bboxes, polygons: S.polygons });
      if (res.masks) {                                      // ZeroFill is in the file now: show the real pixels
        S.masks = []; S.undo = []; S.redo = []; S.imgVersion += 1;
        S.img = await loadImg(imageUrl(name));
      }
      renderImages(); updatePanels(); redraw();
      return true;
    } catch (err) {
      S.saveState = 'error'; updateStatus();
      say('Could not save: ' + err.message);
      return false;
    }
  }

  function conflictChoice() {
    return new Promise((resolve) => {
      const body = h('div', {}, h('p', {}, 'Someone else saved this picture while you were editing it.'),
        h('p', { class: 'muted' }, 'Keep your version (theirs is overwritten), take theirs (your changes are dropped), or decide later.'));
      const overlay = dialog('This picture was changed', body, true);
      const done = (v) => { overlay.remove(); resolve(v); };
      body.append(h('div', { class: 'row' },
        h('button', { class: 'btn', onclick: () => done('cancel') }, 'Decide later'),
        h('button', { class: 'btn', onclick: () => done('theirs') }, 'Take theirs'),
        h('button', { class: 'btn danger', onclick: () => done('mine') }, 'Keep mine')));
    });
  }

  // Last chance when the tab closes: a keepalive request outlives the page.
  function flush() {
    if (S.dirty && S.name && !S.closed) {
      api(`ws/${enc(S.instance)}/annotations/${enc(S.name)}`, 'PUT', payload(), { keepalive: true }).catch(() => {});
    }
  }

  // ================================================================== pictures
  const imageUrl = (name) => `/api/ws/${enc(S.instance)}/image/${enc(name)}?v=${S.imgVersion}`;
  function loadImg(url) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('The picture could not be loaded.'));
      im.src = url;
    });
  }

  async function reloadAnnotations() {
    const a = await api(`ws/${enc(S.instance)}/annotations/${enc(S.name)}`);
    S.bboxes = a.bboxes; S.polygons = a.polygons; S.version = a.version; S.masks = [];
    S.undo = []; S.redo = []; clearSelection(); S.preview = []; S.maskPts = [];
    redraw(); updatePanels();
  }

  async function show(i) {
    const gen = ++S.gen;
    const name = S.images[i];
    S.busy = true;
    try {
      const [a, im] = await Promise.all([api(`ws/${enc(S.instance)}/annotations/${enc(name)}`), loadImg(imageUrl(name))]);
      if (gen !== S.gen) return;
      S.idx = i; S.name = name; S.img = im; S.w = a.width; S.h = a.height;
      S.bboxes = a.bboxes; S.polygons = a.polygons; S.version = a.version; S.masks = [];
      S.undo = []; S.redo = []; clearSelection(); S.preview = []; S.maskPts = []; S.dirty = false; S.saveState = 'idle';
      try { localStorage.setItem('jelibox-last-' + S.instance, name); } catch (e) { /* private mode */ }
      fit(); renderImages(); updatePanels();
    } catch (err) {
      say(err.message);
    } finally { S.busy = false; }
  }

  async function go(i) {
    if (S.nav || S.busy || !S.images.length) return;
    S.nav = true;                                           // one move at a time: a held key must not race itself
    try {
      if (S.name && !(await saveNow(true))) return;
      await show(((i % S.images.length) + S.images.length) % S.images.length);
    } finally { S.nav = false; }
  }
  const step = (d) => go(S.idx + d);

  async function deleteImage() {
    if (!S.name || S.busy) return;
    if (!(await confirmBox('Delete picture', `Delete the picture "${S.name}" and all its annotations?\n\nThis cannot be undone!`, 'Delete', true))) return;
    try {
      const res = await api(`ws/${enc(S.instance)}/images/${enc(S.name)}`, 'DELETE');
      S.images = res.images; S.annotated = new Set(res.annotated); S.dirty = false; S.name = null;
      if (!S.images.length) { say('All pictures were deleted.'); env.home(); return; }
      await show(Math.min(S.idx, S.images.length - 1));
    } catch (err) { say(err.message); }
  }

  function repeat() {
    if (!S.prev) { say('No previous annotations to repeat.'); return; }
    pushUndo();
    S.bboxes = clone(S.prev.bboxes); S.polygons = clone(S.prev.polygons);
    clearSelection(); touch();
  }

  // ================================================================== view
  function resize() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
    if (S.fitted && S.img) fit(); else redraw();             // a window that changed size: keep the picture centred
  }
  function cssSize() { const r = stage.getBoundingClientRect(); return { w: r.width, h: r.height }; }
  // the size of a picture that fits the window: zooming out never goes below it (as when you press Next / Previous)
  function fitScale() {
    const { w, h: hh } = cssSize();
    return Math.min((w - 40) / S.w, (hh - 40) / S.h, 3);
  }
  function fit() {
    const { w, h: hh } = cssSize();
    if (!S.w || !S.h || w < 50 || hh < 50) { redraw(); return; }
    const scale = fitScale();
    S.view = { scale, ox: (w - S.w * scale) / 2, oy: (hh - S.h * scale) / 2 };
    S.fitted = true;
    redraw(); updateInfo();
  }
  function zoomAt(factor, sx, sy) {
    const v = S.view, low = S.w && S.h ? fitScale() : 0.05;
    if (v.scale * factor <= low * 1.0001) { fit(); return; }          // zoomed out as far as it goes: centred, like a fresh picture
    const next = clamp(v.scale * factor, low, 30);
    const k = next / v.scale;
    S.view = { scale: next, ox: sx - (sx - v.ox) * k, oy: sy - (sy - v.oy) * k };
    S.fitted = false;
    redraw(); updateInfo();
  }
  function zoomBy(f) { const { w, h: hh } = cssSize(); zoomAt(f, w / 2, hh / 2); }

  function point(e) {
    const r = canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top, v = S.view;
    return { sx, sy, x: (sx - v.ox) / v.scale, y: (sy - v.oy) / v.scale };
  }
  const inside = (p) => ({ x: clamp(p.x, 0, S.w), y: clamp(p.y, 0, S.h) });
  const toScreen = (x, y) => [S.view.ox + x * S.view.scale, S.view.oy + y * S.view.scale];

  // ================================================================== drawing
  function redraw() { if (!frame) frame = requestAnimationFrame(() => { frame = 0; draw(); }); }

  function draw() {
    if (S.closed) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w, h: hh } = cssSize();
    ctx.clearRect(0, 0, w, hh);
    if (!S.img) return;
    const v = S.view;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(S.img, v.ox, v.oy, S.w * v.scale, S.h * v.scale);

    ctx.fillStyle = '#000';                                // ZeroFill regions waiting to be written
    for (const region of S.masks) { path(region); ctx.fill(); }

    S.polygons.forEach((poly, i) => {
      if (S.hidden.has(poly[1])) return;
      const color = colorOf(poly[1]), sel = i === S.selPoly;
      path(poly[0]);
      ctx.fillStyle = hexA(color, sel ? 0.28 : 0.14); ctx.fill();
      ctx.lineWidth = sel ? 3 : 2; ctx.strokeStyle = sel ? '#ff3b3b' : color; ctx.stroke();
      if (sel) for (let k = 0; k < poly[0].length; k++) handle(poly[0][k], k === S.selPt);
      if (S.showText && poly[0].length) {
        const [px, py] = toScreen(poly[0][0][0], poly[0][0][1]);
        ctx.font = '12px "Segoe UI", sans-serif'; ctx.fillStyle = color; ctx.fillText(poly[1], px + 5, py - 6);
      }
    });

    S.bboxes.forEach((b, i) => {
      if (S.hidden.has(b[4])) return;
      const single = i === S.selBox, multi = S.selBoxes.has(i);
      const color = single ? '#ff3b3b' : multi ? '#ffd400' : colorOf(b[4]);
      const [x1, y1] = toScreen(b[0], b[1]), [x2, y2] = toScreen(b[2], b[3]);
      ctx.lineWidth = single || multi ? 3 : 2; ctx.strokeStyle = color; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (single) for (const [hx, hy] of handleSpots(x1, y1, x2, y2)) square(hx, hy);
      if (S.showText) {
        ctx.font = '12px "Segoe UI", sans-serif';
        const tw = ctx.measureText(b[4]).width;
        ctx.fillStyle = color; ctx.fillRect(x1, y1 - 18, tw + 8, 18);
        ctx.fillStyle = '#fff'; ctx.fillText(b[4], x1 + 4, y1 - 5);
      }
    });

    if (S.preview.length) drawPreview(S.preview, S.mode === 'polygon' && !S.maskMode ? colorOf(S.curClass) : '#ff6464');
    if (S.maskPts.length) drawPreview(S.maskPts, '#ff6464');

    if (S.drag && S.drag.type === 'draw') {                // the box being drawn
      const a = S.drag.start, m = S.drag.now || a;
      const [x1, y1] = toScreen(Math.min(a.x, m.x), Math.min(a.y, m.y)), [x2, y2] = toScreen(Math.max(a.x, m.x), Math.max(a.y, m.y));
      ctx.lineWidth = 2; ctx.strokeStyle = '#00c8ff'; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    if (S.drag && S.drag.type === 'select') {              // shift + drag selection rectangle
      const a = S.drag.start, m = S.drag.now || a;
      const [x1, y1] = toScreen(Math.min(a.x, m.x), Math.min(a.y, m.y)), [x2, y2] = toScreen(Math.max(a.x, m.x), Math.max(a.y, m.y));
      ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffd400'; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    }

    if (S.force) { ctx.lineWidth = 3; ctx.strokeStyle = '#c83c3c'; ctx.strokeRect(v.ox + 3, v.oy + 3, S.w * v.scale - 6, S.h * v.scale - 6); }

    if (S.mouse && !(S.drag && S.drag.type === 'pan') && (S.maskMode || S.mode === 'bbox' || S.preview.length)) crosshair(S.mouse);
  }

  function path(points) {
    ctx.beginPath();
    points.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y); if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy); });
    ctx.closePath();
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  }
  function handleSpots(x1, y1, x2, y2) {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    return [[x1, y1], [mx, y1], [x2, y1], [x2, my], [x2, y2], [mx, y2], [x1, y2], [x1, my]];
  }
  function square(x, y) {
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#ff3b3b'; ctx.lineWidth = 1.5;
    ctx.fillRect(x - 4, y - 4, 8, 8); ctx.strokeRect(x - 4, y - 4, 8, 8);
  }
  function handle(p, active) {
    const [x, y] = toScreen(p[0], p[1]);
    ctx.beginPath(); ctx.arc(x, y, active ? 6 : 4.5, 0, 7);
    ctx.fillStyle = active ? '#ff3b3b' : '#fff'; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = '#ff3b3b'; ctx.stroke();
  }
  function drawPreview(points, color) {
    ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.beginPath();
    points.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y); if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy); });
    if (S.mouse) ctx.lineTo(S.mouse.sx, S.mouse.sy);
    ctx.stroke();
    points.forEach((p, i) => {
      const [sx, sy] = toScreen(p[0], p[1]);
      ctx.beginPath(); ctx.arc(sx, sy, i === 0 && points.length >= 3 ? 7 : 3.5, 0, 7);
      ctx.fillStyle = i === 0 ? '#fff' : color; ctx.fill(); ctx.stroke();
    });
  }
  function crosshair(m) {
    const { w, h: hh } = cssSize();
    for (const [width, style, dash] of [[3, 'rgba(0,0,0,.35)', []], [1, 'rgba(255,255,255,.95)', [5, 3]]]) {
      ctx.lineWidth = width; ctx.strokeStyle = style; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(m.sx, 0); ctx.lineTo(m.sx, hh); ctx.moveTo(0, m.sy); ctx.lineTo(w, m.sy); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // ================================================================== mouse
  const dist = (sx, sy, [x, y]) => { const [px, py] = toScreen(x, y); return Math.hypot(sx - px, sy - py); };

  function boxAt(p) {
    let best = null, area = Infinity;
    S.bboxes.forEach((b, i) => {
      if (S.hidden.has(b[4])) return;
      if (p.x >= b[0] && p.x <= b[2] && p.y >= b[1] && p.y <= b[3]) {
        const a = (b[2] - b[0]) * (b[3] - b[1]);
        if (a < area) { area = a; best = i; }
      }
    });
    return best;
  }
  function handleAt(b, sx, sy) {
    const [x1, y1] = toScreen(b[0], b[1]), [x2, y2] = toScreen(b[2], b[3]);
    const modes = ['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'];
    const size = Math.max(12, 12 * Math.max(S.view.scale, 0.5));
    const spots = handleSpots(x1, y1, x2, y2);
    for (let i = 0; i < 8; i++) if (Math.abs(sx - spots[i][0]) < size && Math.abs(sy - spots[i][1]) < size) return modes[i];
    return null;
  }
  function polygonVertexAt(p) {
    for (let i = 0; i < S.polygons.length; i++) {
      if (S.hidden.has(S.polygons[i][1])) continue;
      const pts = S.polygons[i][0];
      for (let k = 0; k < pts.length; k++) if (dist(p.sx, p.sy, pts[k]) <= 8) return [i, k];
    }
    return null;
  }
  function inPolygon(x, y, pts) {
    let c = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  function onDown(e) {
    if (!S.img || S.busy) return;
    canvas.focus();
    const p = point(e);
    if (e.button === 1 || (e.button === 0 && S.space)) {
      S.drag = { type: 'pan', sx: p.sx, sy: p.sy, ox: S.view.ox, oy: S.view.oy };
      canvas.setPointerCapture(e.pointerId); e.preventDefault(); return;
    }
    if (e.button === 2) { rightClick(p); return; }
    if (e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);

    if (S.maskMode) { addPoint(S.maskPts, p, finishMask); return; }
    if (S.mode === 'polygon') { polygonDown(p); return; }

    // box mode
    if (e.shiftKey) { S.selBox = null; S.selBoxes = new Set(); S.drag = { type: 'select', start: inside(p), now: inside(p) }; redraw(); return; }
    S.selBoxes = new Set(); S.selBox = null; S.selPoly = null;
    if (S.force) { S.drag = { type: 'draw', start: inside(p), now: inside(p) }; redraw(); return; }
    const i = boxAt(p);
    if (i !== null) {
      S.selBox = i;
      const mode = handleAt(S.bboxes[i], p.sx, p.sy);
      S.drag = mode ? { type: 'resize', mode, pushed: false } : { type: 'move', last: inside(p), pushed: false };
      updatePanels(); redraw(); return;
    }
    S.drag = { type: 'draw', start: inside(p), now: inside(p) };
    updatePanels(); redraw();
  }

  function onMove(e) {
    const p = point(e);
    S.mouse = p;
    const d = S.drag;
    if (!d) { if (S.img && (S.maskMode || S.mode === 'bbox' || S.preview.length)) redraw(); return; }
    const q = inside(p);
    if (d.type === 'pan') { S.view.ox = d.ox + p.sx - d.sx; S.view.oy = d.oy + p.sy - d.sy; S.fitted = false; redraw(); return; }
    if (d.type === 'draw' || d.type === 'select') { d.now = q; redraw(); return; }
    if (d.type === 'move' && S.selBox !== null) {
      const b = S.bboxes[S.selBox];
      let dx = q.x - d.last.x, dy = q.y - d.last.y;
      dx = clamp(dx, -b[0], S.w - b[2]); dy = clamp(dy, -b[1], S.h - b[3]);
      if (dx || dy) {
        if (!d.pushed) { pushUndo(); d.pushed = true; }
        b[0] += dx; b[2] += dx; b[1] += dy; b[3] += dy; d.last = { x: d.last.x + dx, y: d.last.y + dy };
        S.dirty = true; redraw();
      }
      return;
    }
    if (d.type === 'resize' && S.selBox !== null) {
      if (!d.pushed) { pushUndo(); d.pushed = true; }
      const b = S.bboxes[S.selBox], min = 5 / S.view.scale, m = d.mode;
      if (m.includes('l')) b[0] = Math.min(q.x, b[2] - min);
      if (m.includes('r')) b[2] = Math.max(q.x, b[0] + min);
      if (m.includes('t')) b[1] = Math.min(q.y, b[3] - min);
      if (m.includes('b')) b[3] = Math.max(q.y, b[1] + min);
      S.dirty = true; redraw(); return;
    }
    if (d.type === 'vertex') {
      if (!d.pushed) { pushUndo(); d.pushed = true; }
      S.polygons[S.selPoly][0][S.selPt] = [q.x, q.y];
      S.dirty = true; redraw();
    }
  }

  function onUp(e) {
    const d = S.drag;
    if (!d) return;
    S.drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ }
    if (d.type === 'draw') {
      const a = d.start, b = d.now || a;
      let x1 = Math.round(Math.min(a.x, b.x)), y1 = Math.round(Math.min(a.y, b.y));
      let x2 = Math.round(Math.max(a.x, b.x)), y2 = Math.round(Math.max(a.y, b.y));
      x1 = clamp(x1, 0, S.w - 2); y1 = clamp(y1, 0, S.h - 2);
      x2 = Math.max(x1 + 2, Math.min(x2, S.w)); y2 = Math.max(y1 + 2, Math.min(y2, S.h));
      if (x2 - x1 >= MIN_BOX && y2 - y1 >= MIN_BOX && S.curClass) {
        pushUndo(); S.bboxes.push([x1, y1, x2, y2, S.curClass]); clearSelection(); d.created = true;
      }
      S.force = false;                                      // "Force" is for one box, like the desktop
    } else if (d.type === 'select') {
      const a = d.start, b = d.now || a;
      const sx1 = Math.min(a.x, b.x), sx2 = Math.max(a.x, b.x), sy1 = Math.min(a.y, b.y), sy2 = Math.max(a.y, b.y);
      S.selBoxes = new Set();
      if ((sx2 - sx1) * S.view.scale > 4 && (sy2 - sy1) * S.view.scale > 4) {
        S.bboxes.forEach((bb, i) => { if (!S.hidden.has(bb[4]) && bb[0] < sx2 && bb[2] > sx1 && bb[1] < sy2 && bb[3] > sy1) S.selBoxes.add(i); });
      }
      redraw(); updatePanels(); return;
    }
    if (d.type === 'pan') { redraw(); return; }
    touch(Boolean(d.pushed || d.created));                  // a click that changed nothing is not an edit
  }

  function polygonDown(p) {
    if (!S.curClass) return;
    const hit = polygonVertexAt(p);
    if (hit && !S.preview.length) {
      S.selPoly = hit[0]; S.selPt = hit[1]; S.selBox = null; S.selBoxes = new Set();
      S.drag = { type: 'vertex', pushed: false }; updatePanels(); redraw(); return;
    }
    addPoint(S.preview, p, finishPolygon);
  }

  function addPoint(list, p, finish) {
    const q = inside(p);
    if (list.length >= 3 && dist(p.sx, p.sy, list[0]) < 10) { finish(); return; }
    list.push([q.x, q.y]); redraw();
  }

  function rightClick(p) {
    if (S.maskMode) { S.maskPts = []; redraw(); return; }
    if (S.mode !== 'polygon') return;
    const hit = polygonVertexAt(p);
    if (hit) {
      if (S.polygons[hit[0]][0].length > 3) { pushUndo(); S.polygons[hit[0]][0].splice(hit[1], 1); S.selPt = null; touch(); }
      return;
    }
    if (S.preview.length) { S.preview = []; redraw(); }
  }

  function cleanPoints(list) {
    const out = [];
    for (const pt of list) { const last = out[out.length - 1]; if (!last || last[0] !== pt[0] || last[1] !== pt[1]) out.push(pt); }
    return out;
  }
  function finishPolygon() {
    const pts = cleanPoints(S.preview);
    if (pts.length < 3) { say('A polygon needs at least 3 different points.'); return; }
    pushUndo(); S.polygons.push([pts, S.curClass]); S.preview = []; clearSelection(); S.selPoly = S.polygons.length - 1;
    touch();
  }
  function finishMask() {
    const pts = cleanPoints(S.maskPts);
    if (pts.length < 3) { say('A region needs at least 3 different points.'); return; }
    pushUndo(); S.masks.push(pts); S.maskPts = []; touch();
  }

  function onDouble(e) {
    if (!S.img) return;
    const p = point(e);
    if (S.maskMode) { if (S.maskPts.length) finishMask(); return; }
    if (S.mode === 'polygon' && S.preview.length) { finishPolygon(); return; }
    const i = boxAt(p);
    if (i !== null) { reassign('bbox', i); return; }
    for (let k = 0; k < S.polygons.length; k++) if (!S.hidden.has(S.polygons[k][1]) && inPolygon(p.x, p.y, S.polygons[k][0])) { reassign('polygon', k); return; }
  }

  function onWheel(e) {
    if (!S.img) return;
    e.preventDefault();
    if (S.drag && S.drag.type !== 'pan') return;
    const p = point(e);
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, p.sx, p.sy);
  }

  // ================================================================== editing commands
  function deleteSelected() {
    if (!hasSelection()) return;
    pushUndo();
    if (S.selBox !== null) S.bboxes.splice(S.selBox, 1);
    else if (S.selBoxes.size) S.bboxes = S.bboxes.filter((_, i) => !S.selBoxes.has(i));
    else if (S.selPoly !== null) S.polygons.splice(S.selPoly, 1);
    clearSelection(); touch();
  }
  function nextClass(cls) {
    const names = S.classes.map((c) => c.name), i = names.indexOf(cls);
    return names[((i < 0 ? 0 : i) + 1) % names.length];
  }
  function changeClassSelected() {
    if (S.selBox !== null) { pushUndo(); S.bboxes[S.selBox][4] = nextClass(S.bboxes[S.selBox][4]); touch(); }
    else if (S.selPoly !== null) { pushUndo(); S.polygons[S.selPoly][1] = nextClass(S.polygons[S.selPoly][1]); touch(); }
  }
  function reassign(kind, index) {
    const current = kind === 'bbox' ? S.bboxes[index][4] : S.polygons[index][1];
    const body = h('div', {});
    const overlay = dialog('Change class', body);
    for (const c of S.classes) {
      body.append(h('button', { class: 'mode' + (c.name === current ? ' on' : ''), onclick: () => {
        overlay.remove();
        if (c.name === current) return;
        pushUndo();
        if (kind === 'bbox') S.bboxes[index][4] = c.name; else S.polygons[index][1] = c.name;
        touch();
      } }, h('b', {}, c.name)));
    }
  }
  function toggleMode() {
    if (S.maskMode) toggleMask();
    S.mode = S.mode === 'bbox' ? 'polygon' : 'bbox';
    S.preview = []; clearSelection(); S.drag = null;
    toast(S.mode === 'polygon' ? 'Polygon mode: click to add points, double-click or Enter to finish, right-click a point to delete it.'
      : 'Box mode: drag to draw a box. B forces a new box over an existing one.');
    updatePanels(); redraw();
  }
  function toggleMask() {
    S.maskMode = !S.maskMode; S.maskPts = []; S.preview = []; clearSelection();
    bMask.textContent = 'ZeroFill: ' + (S.maskMode ? 'ON' : 'OFF');
    bMask.classList.toggle('primary', S.maskMode);
    if (S.maskMode) toast('ZeroFill: click points (3 or more), double-click or Enter to finish. The area is written black into the picture when you leave it. Ctrl+Z undoes.');
    redraw(); updateStatus();
  }
  function toggleForce() { S.force = !S.force; updateStatus(); redraw(); }
  function selectClass(i) {
    if (i < 0 || i >= S.classes.length) return;
    S.curClass = S.classes[i].name; renderClasses(); updateInfo();
  }

  // ================================================================== classes
  function addClassDialog() {
    const name = h('input', { type: 'text', maxlength: 16, placeholder: 'letters, digits, _ or -' });
    const error = h('div', { class: 'error', hidden: true });
    const body = h('div', {}, h('label', {}, 'Class name (16 characters at most)'), name, error);
    const overlay = dialog('Add class', body);
    const submit = async () => {
      try {
        if (!(await saveNow())) return;
        const res = await api(`ws/${enc(S.instance)}/classes`, 'POST', { name: name.value.trim() });
        S.classes = res.classes; S.curClass = name.value.trim();
        overlay.remove(); renderClasses(); renderVisibility(); updatePanels();
      } catch (err) { error.textContent = err.message; error.hidden = false; }
    };
    name.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
    body.append(h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Cancel'),
      h('button', { class: 'btn primary', onclick: submit }, 'Add')));
    name.focus();
  }

  async function deleteClassDialog() {
    if (!S.curClass) { say('Select a class to delete first.'); return; }
    const cls = S.curClass;
    const here = S.bboxes.filter((b) => b[4] === cls).length + S.polygons.filter((p) => p[1] === cls).length;
    const ok = await confirmBox('Delete class',
      `Deleting the class "${cls}" will:\n\n  • remove it from the dataset\n  • delete ${here} annotation(s) in this picture\n  • update all label files of the workspace\n  • it cannot be undone!\n\nAre you sure?`, 'Delete class', true);
    if (!ok) return;
    try {
      if (!(await saveNow())) return;
      const res = await api(`ws/${enc(S.instance)}/classes/${enc(cls)}`, 'DELETE');
      S.classes = res.classes; S.annotated = new Set(res.annotated);
      S.hidden.delete(cls);
      S.curClass = S.classes.length ? S.classes[0].name : null;
      await reloadAnnotations();
      renderClasses(); renderVisibility(); renderImages();
      say(res.message);
    } catch (err) { say(err.message); }
  }

  // ================================================================== panels
  function renderClasses() {
    classList.replaceChildren(...S.classes.map((c, i) => h('div', {
      class: 'ed-class' + (c.name === S.curClass ? ' on' : ''), onclick: () => selectClass(i), title: `Press ${i + 1}`,
    }, h('i', { style: `background:${c.color}` }), h('span', {}, `${i + 1}. ${c.name}`))));
  }
  function renderVisibility() {
    visList.replaceChildren(...S.classes.map((c) => {
      const box = h('input', { type: 'checkbox', onchange: () => { if (box.checked) S.hidden.delete(c.name); else S.hidden.add(c.name); updatePanels(); redraw(); } });
      box.checked = !S.hidden.has(c.name);
      return h('label', { class: 'ed-visrow' }, box, h('span', {}, c.name), h('b', { 'data-count': c.name }, '0'));
    }));
  }
  function renderImages() {
    const rows = [];
    S.images.forEach((name, i) => {
      if (S.filter && !name.toLowerCase().includes(S.filter)) return;
      rows.push(h('div', { class: 'ed-img' + (i === S.idx ? ' on' : '') + (S.annotated.has(name) ? ' done' : ''), 'data-i': i, title: name,
        onclick: () => go(i) }, name));
    });
    imageList.replaceChildren(...rows);
    const on = imageList.querySelector('.on');
    if (on) on.scrollIntoView({ block: 'nearest' });
  }
  function updateStatus() {
    modeBadge.textContent = S.maskMode ? '⬛ ZEROFILL' : S.mode === 'polygon' ? '◆ POLYGON MODE' : '▭ BBOX MODE';
    modeBadge.className = 'ed-badge' + (S.mode === 'polygon' || S.maskMode ? ' amber' : '');
    forceLabel.textContent = S.force ? 'Force: ON' : 'Force: OFF';
    forceLabel.className = 'ed-force' + (S.force ? ' on' : '');
    const n = S.annotated.size, total = S.images.length;
    progressLabel.textContent = `Annotated: ${n} / ${total}  (${total ? Math.round(100 * n / total) : 0}%)`;
    saveLabel.textContent = { idle: '', unsaved: 'Unsaved changes', saving: 'Saving ...', saved: 'Saved', error: 'Not saved!' }[S.saveState];
    saveLabel.className = 'ed-save ' + S.saveState;
    bMode.textContent = S.mode === 'polygon' ? 'Mode: Polygon' : 'Mode: Box';
    bUndo.disabled = !S.undo.length; bRedo.disabled = !S.redo.length;
  }
  function updateInfo() {
    if (!S.name) { info.textContent = ''; return; }
    const done = S.annotated.size, total = S.images.length, sep = '────────────────────────────';
    let visible = 0, hidden = 0;
    for (const b of S.bboxes) (S.hidden.has(b[4]) ? hidden++ : visible++);
    for (const p of S.polygons) (S.hidden.has(p[1]) ? hidden++ : visible++);
    const lines = [sep, ` IMAGE ${S.idx + 1} / ${total}`, ` ${S.name}`, ` ${S.w}×${S.h} px   Scale: ${S.view.scale.toFixed(2)}×`, sep,
      ' DATASET PROGRESS', ` ${done} / ${total} annotated (${total ? (100 * done / total).toFixed(1) : 0}%)`, ` ${total - done} remaining`, sep,
      ` CLASS  ${S.curClass || '-'}  (${S.classes.length} in all)`, ' ANNOTATIONS',
      ` Total:    ${S.bboxes.length + S.polygons.length}`, `    BBoxes:  ${S.bboxes.length}`, `    Polygon: ${S.polygons.length}`,
      ` Visible:  ${visible}` + (hidden ? `   Hidden: ${hidden}` : ''), ` Mode:     ${S.mode.toUpperCase()}`];
    info.textContent = lines.join('\n');
  }
  function updatePanels() {
    const counts = {};
    for (const b of S.bboxes) counts[b[4]] = (counts[b[4]] || 0) + 1;
    for (const p of S.polygons) counts[p[1]] = (counts[p[1]] || 0) + 1;
    visList.querySelectorAll('[data-count]').forEach((el) => { el.textContent = counts[el.dataset.count] || 0; });
    updateStatus(); updateInfo();
  }

  // ================================================================== keyboard
  function onKey(e) {
    if (S.closed) return;
    if (e.code === 'Space' && !inField(e) && !modalOpen()) { S.space = e.type === 'keydown'; if (e.type === 'keydown') e.preventDefault(); return; }
    if (e.type !== 'keydown' || inField(e) || modalOpen()) return;
    const key = e.key, low = key.toLowerCase(), ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && low === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (ctrl && low === 'y') { e.preventDefault(); redo(); return; }
    if (ctrl || e.altKey) return;
    if (key === 'ArrowLeft' || low === 'a') { e.preventDefault(); step(-1); }
    else if (key === 'ArrowRight' || low === 'd') { e.preventDefault(); step(1); }
    else if (low === 'r') deleteSelected();
    else if (low === 's') changeClassSelected();
    else if (low === 'e') repeat();
    else if (low === 'b') toggleForce();
    else if (low === 'm') toggleMode();
    else if (low === 'f') fit();
    else if (low === 'g') runAssist(ed);
    else if (low === 't') openTrain(ed);
    else if (key === 'Delete') deleteImage();
    else if (key === 'Enter') { if (S.maskMode) { if (S.maskPts.length) finishMask(); } else if (S.mode === 'polygon' && S.preview.length) finishPolygon(); }
    else if (key === 'Escape') escape();
    else if (/^[1-9]$/.test(key)) selectClass(Number(key) - 1);
  }
  function escape() {
    if (S.maskMode) { if (S.maskPts.length) { S.maskPts = []; redraw(); } else toggleMask(); return; }
    if (S.preview.length) { S.preview = []; redraw(); return; }
    leave();
  }

  async function leave() {
    if (S.name && !(await saveNow(true))) return;
    env.home();
  }

  // ================================================================== what the Label Assistant needs from the editor
  const ed = {
    instance,
    name: () => S.name,
    annotations: () => ({ bboxes: S.bboxes, polygons: S.polygons }),
    polygon: () => S.mode === 'polygon',
    saveNow: () => saveNow(),
    apply(result) {                                        // new annotations from Infer: undoable, not yet reviewed
      toast(result.message);
      if (!result.added) return;
      pushUndo();
      S.bboxes.push(...result.bboxes); S.polygons.push(...result.polygons);
      S.unreviewed.add(S.name);
      clearSelection(); touch();
    },
    async reloadAll() {                                    // Auto-annotate wrote files: show what is on disk now
      try {
        const fresh = await api(`ws/${enc(S.instance)}`);
        S.annotated = new Set(fresh.annotated);
        await reloadAnnotations(); renderImages(); updatePanels();
      } catch (err) { say(err.message); }
    },
  };

  // ================================================================== start and stop
  const onPageHide = () => flush();
  const onVisibility = () => { if (document.visibilityState === 'hidden') flush(); };
  const onResize = () => resize();
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('pointerleave', () => { if (!S.drag) { S.mouse = null; redraw(); } });
  canvas.addEventListener('dblclick', onDouble);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  window.addEventListener('keydown', onKey);
  window.addEventListener('keyup', onKey);
  window.addEventListener('resize', onResize);
  window.addEventListener('pagehide', onPageHide);
  document.addEventListener('visibilitychange', onVisibility);

  (async function start() {
    try {
      const info0 = await api(`ws/${enc(instance)}`);
      S.images = info0.images; S.annotated = new Set(info0.annotated); S.classes = info0.classes;
      S.curClass = S.classes.length ? S.classes[0].name : null;
      if (!S.images.length) { say('This dataset has no pictures.'); env.home(); return; }
      renderClasses(); renderVisibility(); resize();
      let first = 0;
      try { first = Math.max(0, S.images.indexOf(localStorage.getItem('jelibox-last-' + instance))); } catch (e) { /* private mode */ }
      await show(first);
    } catch (err) {
      say(err.message);
      env.home();
    }
  })();

  return function cleanup() {
    flush();
    S.closed = true;
    clearTimeout(S.timer);
    window.removeEventListener('keydown', onKey);
    window.removeEventListener('keyup', onKey);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('pagehide', onPageHide);
    document.removeEventListener('visibilitychange', onVisibility);
  };
}

// Export Dataset, Export Model and the pieces the Train window shares: the Train / Valid / Test bar and the augmentation
// picker. Everything is computed on the server; the page only asks and shows.

import { h, api, toast, dialog, notice, watchJob, pickServer } from './ui.js';

const enc = encodeURIComponent;

// ------------------------------------------------------------------ Train / Valid / Test bar
// Python's round() goes to the even number on .5; the exporters use it, so the counts shown must too.
function roundEven(x) {
  const f = Math.floor(x), diff = x - f;
  if (Math.abs(diff - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}

export function splitCounts(total, trainPct, validPct) {
  let train = roundEven(total * trainPct / 100), valid = roundEven(total * validPct / 100);
  if (train + valid > total) valid = Math.max(0, total - train);
  return [train, valid, total - train - valid];
}

// Two handles on a bar: train | valid | test. get() = {train, valid, test} in whole percent.
export function splitBar({ train = 70, valid = 20, total = 0, onChange = () => {} } = {}) {
  const state = { train, valid, total };
  const seg = (cls) => h('div', { class: 'seg ' + cls });
  const [a, b, c] = [seg('train'), seg('valid'), seg('test')];
  const bar = h('div', { class: 'splitbar' }, a, b, c);
  const label = h('div', { class: 'splitlabel' });
  const num = (key) => {
    const el = h('input', { type: 'number', min: 0, max: 100, step: 1, onchange: () => { set(key, Number(el.value)); } });
    return el;
  };
  const tIn = num('train'), vIn = num('valid');
  function set(key, value) {
    value = Math.max(0, Math.min(100, Math.round(value) || 0));
    if (key === 'train') { state.train = value; state.valid = Math.min(state.valid, 100 - value); }
    else state.valid = Math.min(value, 100 - state.train);
    draw(); onChange(get());
  }
  function get() { return { train: state.train, valid: state.valid, test: 100 - state.train - state.valid }; }
  function draw() {
    const g = get();
    a.style.width = g.train + '%'; b.style.width = g.valid + '%'; c.style.width = g.test + '%';
    tIn.value = g.train; vIn.value = g.valid;
    const n = splitCounts(state.total, g.train, g.valid);
    label.textContent = `Train ${g.train}% (${n[0]})   ·   Valid ${g.valid}% (${n[1]})   ·   Test ${g.test}% (${n[2]})`;
  }
  // drag a handle: the border between train|valid and valid|test
  function drag(which) {
    return (e) => {
      e.preventDefault();
      const rect = bar.getBoundingClientRect();
      const move = (ev) => {
        const pct = Math.round(100 * Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width)));
        if (which === 'train') { state.train = pct; state.valid = Math.min(state.valid, 100 - pct); }
        else state.valid = Math.max(0, pct - state.train);
        draw(); onChange(get());
      };
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    };
  }
  const h1 = h('div', { class: 'handle', onpointerdown: drag('train') }), h2 = h('div', { class: 'handle', onpointerdown: drag('valid') });
  a.append(h1); b.append(h2);
  const node = h('div', { class: 'splitwrap' }, bar, label,
    h('div', { class: 'inline', style: 'gap:6px;margin-top:6px' }, h('span', { class: 'muted' }, 'Train %'), tIn, h('span', { class: 'muted' }, 'Valid %'), vIn));
  draw();
  return { node, get, setTotal(n) { state.total = n; draw(); }, set: (t, v) => { state.train = t; state.valid = v; draw(); } };
}

// ------------------------------------------------------------------ the augmentation picker
function freshState(catalog) {
  const ops = {};
  for (const [name, spec] of Object.entries(catalog.operations)) {
    ops[name] = { enabled: false, magnitude: spec.default, probability: 50, options: { ...spec.options } };
  }
  return { ops, outputs: catalog.default_outputs, keep_original: true, valid: false, test: false, seed: '' };
}

const tileOn = (tile, st) => tile.ops.some((op) => st.ops[op].enabled);

function summaryOf(catalog, st) {
  return catalog.tiles.filter((t) => tileOn(t, st)).map((t) => {
    if (!t.slider) return t.title;
    const amount = `${Math.round(st.ops[t.slider[0]].magnitude)}${t.slider[2]}`;
    return t.slider[1].includes('Between') ? `${t.title} ±${amount}` : `${t.title} up to ${amount}`;
  });
}

// `training`: tiles Ultralytics already does while it trains are greyed out. onChange() after every change.
export async function augPicker(ed, { training = false, onChange = () => {} } = {}) {
  const base = `ws/${enc(ed.instance)}`;
  const catalog = await api(`${base}/augment/catalog?training=${training ? 1 : 0}`);
  const st = freshState(catalog);
  const grid = h('div', { class: 'augrid' });
  const sampleLabel = h('span', { class: 'muted' });
  const summary = h('div', { class: 'augsum' }), detail = h('div', { class: 'muted', style: 'font-size:12px' });
  const outputs = h('input', { type: 'number', min: catalog.outputs[0], max: catalog.outputs[1], value: st.outputs });
  const keep = h('input', { type: 'checkbox' }); keep.checked = true;
  const valid = h('input', { type: 'checkbox' }), test = h('input', { type: 'checkbox' });
  const seed = h('input', { type: 'text', placeholder: 'optional', style: 'width:110px' });
  let thumbs = {};

  async function loadThumbs() {
    try { thumbs = (await api(`${base}/augment/thumbnails`, 'POST', { ops: st.ops })).items; } catch (e) { thumbs = {}; }
    drawGrid();
  }
  async function loadSample() {
    try { const s = await api(`${base}/augment/sample`); sampleLabel.textContent = s.name ? `${s.name}${s.is_default ? ' (the first picture)' : ''}` : 'no picture'; }
    catch (e) { sampleLabel.textContent = ''; }
  }
  function outs() { return Math.max(catalog.outputs[0], Math.min(catalog.outputs[1], parseInt(outputs.value, 10) || catalog.default_outputs)); }
  function copies() { return keep.checked ? outs() - 1 : outs(); }
  function drawSummary() {
    const names = summaryOf(catalog, st);
    summary.textContent = names.length ? `${names.length} selected: ${names.join(', ')}` : 'No augmentation selected.';
    summary.className = 'augsum' + (names.length ? ' on' : '');
    if (!names.length) { detail.textContent = ''; return; }
    const c = copies();
    let text = c <= 0 ? 'One output that is the original adds nothing: raise the outputs or untick "Keep the original".'
      : keep.checked ? `Each training image becomes ${outs()} pictures: the original and ${c} augmented.`
        : `Each training image is replaced by ${outs()} augmented pictures.`;
    if (valid.checked || test.checked) text += ` The ${[valid.checked && 'validation', test.checked && 'test'].filter(Boolean).join(' and ')} pictures get the same treatment.`;
    detail.textContent = text;
  }
  function changed() { drawSummary(); onChange(); }
  function drawGrid() {
    grid.replaceChildren(...catalog.tiles.map((tile) => {
      const on = tileOn(tile, st);
      const disabled = !!tile.note;
      return h('button', { class: 'augtile' + (on ? ' on' : '') + (disabled ? ' off' : ''), title: tile.note || tile.description,
        onclick: () => (disabled ? notice(tile.title, tile.note) : openTile(tile)) },
      thumbs[tile.key] ? h('img', { src: thumbs[tile.key], alt: '' }) : h('div', { class: 'ph' }),
      h('span', {}, tile.title), on ? h('i', {}, '✓') : null);
    }));
  }

  function openTile(tile) {
    const work = {};
    for (const op of tile.ops) work[op] = JSON.parse(JSON.stringify(st.ops[op]));
    const firstTime = !tile.ops.some((op) => st.ops[op].enabled);
    const side = h('div', { class: 'tileside' }), stage = h('div', { class: 'tilestage' });
    const warning = h('div', { class: 'error', hidden: true });
    const overlay = dialog(tile.title, h('div', { class: 'tilebody' }, side, stage), true);
    overlay.firstChild.classList.add('wide');
    const controls = {};
    side.append(h('p', { class: 'muted' }, tile.description));
    const unit = tile.slider ? tile.slider[2] : '';
    let amountOut;
    if (tile.slider) {
      const spec = catalog.operations[tile.slider[0]];
      amountOut = h('b', { class: 'val' });
      const input = h('input', { type: 'range', min: spec.range[0], max: spec.range[1], step: 1, oninput: () => changedTile() });
      input.value = Math.min(spec.range[1], Math.max(spec.range[0], work[tile.slider[0]].magnitude));
      controls.magnitude = input;
      side.append(h('div', { class: 'slider2' }, h('span', {}, tile.slider[1]), amountOut), input);
    }
    for (const [op, option, text] of tile.choices) {
      const box = h('input', { type: 'checkbox', onchange: () => changedTile() });
      box.checked = option === null ? (work[op].enabled || firstTime) : !!work[op].options[option];
      controls[op + ':' + option] = box;
      side.append(h('label', { class: 'check big' }, box, ' ', text));
    }
    if (tile.count) {
      const [op, option, text, lo, hi] = tile.count;
      const input = h('input', { type: 'number', min: lo, max: hi, value: work[op].options[option] || 3, onchange: () => changedTile() });
      controls.count = input;
      side.append(h('div', { class: 'inline', style: 'margin-top:10px' }, h('b', {}, text), input));
    }
    const chanceOut = h('b', { class: 'val' });
    const chance = h('input', { type: 'range', min: 5, max: 100, step: 5, oninput: () => changedTile() });
    chance.value = work[tile.ops[0]].probability;
    side.append(h('div', { class: 'slider2', style: 'margin-top:14px' }, h('span', {}, 'Applied to'), chanceOut), chance,
      h('p', { class: 'muted', style: 'font-size:12px' }, 'of the augmented pictures (each picture draws its own combination).'), warning);

    function read() {
      for (const op of tile.ops) work[op].probability = Number(chance.value);
      if (tile.slider) work[tile.slider[0]].magnitude = Number(controls.magnitude.value);
      for (const [op, option] of tile.choices) {
        const v = controls[op + ':' + option].checked;
        if (option === null) work[op].enabled = v; else work[op].options[option] = v;
      }
      if (tile.count) { const [op, option, , lo, hi] = tile.count; work[op].options[option] = Math.max(lo, Math.min(hi, parseInt(controls.count.value, 10) || lo)); }
    }
    function valid_() {
      if (!tile.choices.length) return true;
      return tile.choices[0][1] === null ? tile.choices.some(([op]) => work[op].enabled) : tile.choices.some(([op, option]) => work[op].options[option]);
    }
    let timer = null;
    function changedTile() {
      read();
      if (amountOut) amountOut.textContent = `${Math.round(work[tile.slider[0]].magnitude)}${unit}`;
      chanceOut.textContent = `${Math.round(chance.value)}%`;
      warning.hidden = valid_(); warning.textContent = 'Tick at least one choice.';
      clearTimeout(timer); timer = setTimeout(render, 150);
    }
    async function render() {
      read();
      try {
        const ops = { ...JSON.parse(JSON.stringify(st.ops)), ...JSON.parse(JSON.stringify(work)) };
        const { items } = await api(`${base}/augment/previews`, 'POST', { tile: tile.key, ops });
        stage.replaceChildren(h('div', { class: 'previews' }, items.map((it, i) =>
          h('figure', { class: i === 0 ? 'orig' : '' }, h('img', { src: it.image, alt: it.label }), h('figcaption', {}, it.label)))));
      } catch (err) { stage.textContent = err.message; }
    }
    changedTile();
    const apply = () => {
      read();
      if (!valid_()) { warning.hidden = false; return; }
      for (const op of tile.ops) {
        const flipStyle = tile.choices.length && tile.choices[0][1] === null;
        st.ops[op] = { ...work[op], enabled: flipStyle ? work[op].enabled : true };
      }
      overlay.remove(); drawSummary(); loadThumbs(); onChange();
    };
    const remove = () => { for (const op of tile.ops) st.ops[op].enabled = false; overlay.remove(); drawSummary(); loadThumbs(); onChange(); };
    overlay.firstChild.append(h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Cancel'),
      tileOn(tile, st) ? h('button', { class: 'btn danger', onclick: remove }, 'Remove') : null,
      h('button', { class: 'btn primary', onclick: apply }, 'Apply')));
  }

  for (const el of [outputs, keep, valid, test]) el.addEventListener('input', changed);
  const node = h('div', { class: 'augpicker' },
    h('p', { class: 'muted' }, 'Augmentations create new training examples for your model to learn from. Click one to set it up; the previews use the picture below.'),
    h('div', { class: 'inline', style: 'margin:8px 0' },
      h('button', { class: 'btn small', onclick: async () => { if (!ed.name()) return; try { await api(`${base}/augment/sample`, 'PUT', { name: ed.name() }); } catch (e) { toast(e.message); } await loadSample(); loadThumbs(); } }, 'Use the picture on screen'),
      h('button', { class: 'btn small', onclick: async () => { await api(`${base}/augment/sample`, 'PUT', { name: '' }); await loadSample(); loadThumbs(); } }, 'First picture'),
      sampleLabel),
    grid,
    h('div', { class: 'inline', style: 'margin-top:10px' }, h('b', {}, 'Outputs per training image'), outputs, h('b', {}, '×'),
      h('label', { class: 'check' }, keep, ' Keep the original as one of them')),
    h('div', { class: 'inline', style: 'margin-top:6px' }, h('span', { class: 'muted' }, 'Also augment'),
      h('label', { class: 'check' }, valid, ' the validation pictures'), h('label', { class: 'check' }, test, ' the test pictures')),
    h('div', { class: 'inline', style: 'margin-top:6px' }, h('span', { class: 'muted' }, 'Seed'), seed,
      h('span', { class: 'muted', style: 'font-size:12px' }, 'the same number gives the same pictures again')),
    summary, detail);
  drawGrid(); drawSummary(); loadSample(); loadThumbs();

  return {
    node,
    // the request body piece, or null when nothing is picked
    get() {
      if (!summaryOf(catalog, st).length) return null;
      return { ops: st.ops, outputs: outs(), keep_original: keep.checked, splits: ['train', ...(valid.checked ? ['val'] : []), ...(test.checked ? ['test'] : [])], seed: seed.value.trim() };
    },
    growth(trainCount) {                        // how many training pictures there will be
      if (!summaryOf(catalog, st).length) return trainCount;
      const c = copies();
      return c <= 0 ? trainCount : trainCount * (c + (keep.checked ? 1 : 0));
    },
    enabled: () => summaryOf(catalog, st).length > 0 && copies() > 0,
    per: () => (copies() > 0 ? copies() + (keep.checked ? 1 : 0) : 1),
    alsoValid: () => valid.checked, alsoTest: () => test.checked,
  };
}

// ------------------------------------------------------------------ result window with a download
export function resultDialog(title, lines, token, fileName) {
  const body = h('div', {}, ...lines.map((l) => h('p', { style: 'white-space:pre-line;word-break:break-all' }, l)));
  const overlay = dialog(title, body, true);
  const link = token ? h('a', { class: 'btn primary', href: `/api/downloads/${token}`, download: fileName || '', style: 'text-align:center;text-decoration:none' }, 'Download again') : null;
  body.append(h('div', { class: 'row' }, link, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close')));
  if (link) setTimeout(() => link.click(), 150);            // the export is not kept on the server: it goes to this device now
}

// ------------------------------------------------------------------ Export Dataset
export async function openExportDataset(ed) {
  const base = `ws/${enc(ed.instance)}`;
  if (ed.name()) await ed.saveNow();                       // what is on screen goes into the export
  let info;
  try { info = await api(`${base}/export/dataset`); } catch (err) { toast(err.message); return; }
  const fmt = h('select', {}, info.formats.map((f) => h('option', { value: f.id }, f.label)));
  const count = h('span', { class: 'muted' });
  const split = splitBar({ train: 70, valid: 20, onChange: () => refresh() });
  const orient = h('input', { type: 'checkbox' }); orient.checked = true;
  const resize = h('input', { type: 'checkbox', onchange: () => { mode.disabled = rw.disabled = rh.disabled = !resize.checked; refresh(); } });
  const mode = h('select', { disabled: true, onchange: () => refresh() }, info.resize.modes.map((m) => h('option', { value: m.id }, m.label)));
  const rw = h('input', { type: 'number', value: 640, min: info.resize.range[0], max: info.resize.range[1], disabled: true }), rh = h('input', { type: 'number', value: 640, min: info.resize.range[0], max: info.resize.range[1], disabled: true });
  const resizeHint = h('p', { class: 'muted', style: 'font-size:12px' });
  const total = h('b', {}), status = h('div', { class: 'muted' });
  let picker = null;
  const body = h('div', { class: 'form' });
  const overlay = dialog('Export Dataset', body, true);
  overlay.firstChild.classList.add('xwide');

  const totalImages = () => info.formats.find((f) => f.id === fmt.value).count;
  function refresh() {
    const n = totalImages(); split.setTotal(n);
    count.textContent = `${n} images`;
    const g = split.get();
    let [tr, va, te] = splitCounts(n, g.train, g.valid);
    if (picker && picker.enabled()) {
      const per = picker.per();
      tr *= per; if (picker.alsoValid()) va *= per; if (picker.alsoTest()) te *= per;
    }
    total.textContent = `Export:   Train ${tr}   ·   Valid ${va}   ·   Test ${te}`;
    resizeHint.textContent = resize.checked ? (info.resize.modes.find((m) => m.id === mode.value) || {}).hint || '' : '';
  }
  fmt.addEventListener('change', refresh);
  const section = (number, title, ...kids) => h('div', { class: 'xcard' }, h('div', { class: 'xhead' }, number ? h('span', { class: 'xnum' }, number) : null, h('b', {}, title)), ...kids);
  body.append(
    section(null, 'Format',
      h('div', { class: 'inline' }, h('span', { class: 'muted' }, 'Format'), fmt, count),
      h('div', { class: 'muted', style: 'margin-top:8px;font-size:12px' }, 'The export is downloaded to this device as a .zip. Nothing is kept on the server.')),
    section(1, 'Train / Valid / Test split', h('p', { class: 'muted' }, 'Drag the two handles, or type the percentages. Pictures are picked at random.'), split.node),
    section(2, 'Preprocessing', h('p', { class: 'muted' }, 'Applied to every exported picture (the augmented ones too). The labels move with the pixels, and your original pictures are never changed.'),
      h('label', { class: 'check big' }, orient, ' Auto-Orient'),
      h('p', { class: 'muted', style: 'font-size:12px;margin:0 0 8px 26px' }, 'Discard the EXIF rotation tag and save the pictures upright, so every program shows them the way you annotated them.'),
      h('div', { class: 'inline' }, h('label', { class: 'check big' }, resize, ' Resize'), mode, rw, h('span', {}, '×'), rh), resizeHint),
    section(3, 'Augmentation', h('div', { id: 'aug-slot', class: 'muted' }, 'Loading ...')));
  const footerStatus = h('div', {}, total, status);
  const exportBtn = h('button', { class: 'btn primary', onclick: start }, 'Export');
  body.append(footerStatus, h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close'), exportBtn));
  refresh();
  try {
    picker = await augPicker(ed, { training: false, onChange: refresh });
    body.querySelector('#aug-slot').replaceWith(picker.node);
    refresh();
  } catch (err) { body.querySelector('#aug-slot').textContent = err.message; }

  async function start() {
    const g = split.get();
    const request = {
      format: fmt.value, train_pct: g.train, valid_pct: g.valid,
      preprocess: { auto_orient: orient.checked, resize: resize.checked ? { mode: mode.value, width: rw.value, height: rh.value } : null },
      augment: picker ? picker.get() : null,
    };
    exportBtn.disabled = true;
    try {
      const started = await api(`${base}/export/dataset`, 'POST', request);
      overlay.remove();
      const job = await watchJob('Export Dataset', started);
      if (job.state === 'error') { await notice('Export failed', job.error); return; }
      const r = job.result;
      resultDialog('Export Completed', [`Your browser is downloading ${r.name} (${r.files} files).`, ...r.notes], r.download, r.name);
    } catch (err) { exportBtn.disabled = false; status.textContent = err.message; status.className = 'error'; }
  }
}

// ------------------------------------------------------------------ Export Model
export async function openExportModel(ed) {
  const base = `ws/${enc(ed.instance)}`;
  let info;
  try { info = await api(`${base}/export/model`); } catch (err) { toast(err.message); return; }
  const fmt = h('select', {}, info.formats.map((f) => h('option', { value: f.id }, f.label)));
  fmt.value = 'onnx';
  const size = h('input', { type: 'number', value: 640, min: 32, max: 8192 });
  const boxes = Object.fromEntries(info.options.map((o) => [o, h('input', { type: 'checkbox' })]));
  const error = h('div', { class: 'error', hidden: true });
  const body = h('div', { class: 'form' },
    info.has_model ? h('p', { class: 'muted' }, `Model: ${info.model}`) : h('div', { class: 'note' }, 'This workspace has no trained model yet. Train one first (the Train button).'),
    h('label', {}, 'Format'), fmt, h('label', {}, 'Image size (px)'), size,
    h('div', { style: 'margin-top:8px' }, info.options.map((o) => h('label', { class: 'check' }, boxes[o], ' ' + o[0].toUpperCase() + o.slice(1)))),
    h('p', {}, h('a', { href: 'https://docs.ultralytics.com/modes/export/#arguments', target: '_blank', rel: 'noopener' }, 'Ultralytics export arguments')), error);
  const overlay = dialog('Export Model', body, true);
  const go = h('button', { class: 'btn primary', disabled: !info.has_model, onclick: async () => {
    error.hidden = true; go.disabled = true;
    try {
      const started = await api(`${base}/export/model`, 'POST', { format: fmt.value, imgsz: size.value, ...Object.fromEntries(Object.entries(boxes).map(([k, v]) => [k, v.checked])) });
      overlay.remove();
      const job = await watchJob('Export Model', started);
      if (job.state === 'error') { await notice('Export Failed', job.error); return; }
      resultDialog('Export Completed', [job.result.download ? `Your browser is downloading ${job.result.name}.` : 'The export finished, but it left no file to download.'], job.result.download, job.result.name);
    } catch (err) { go.disabled = false; error.textContent = err.message; error.hidden = false; }
  } }, 'Export');
  body.append(h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close'), go));
}

// The Label Assistant in the editor: Infer (one picture or the whole dataset), the settings window of each mode.
// `ed` is what the editor lends: instance, name(), annotations(), polygon(), apply(result), reloadAll(), saveNow().

import { h, api, toast, dialog, confirmBox, notice, watchJob, pickServer } from './ui.js';

const enc = encodeURIComponent;
const SKIP = '— skip —';

// POST that may need a yes first (a model that is not on the server yet): ask, then send it again.
async function startJob(path, body) {
  for (;;) {
    try { return await api(path, 'POST', body); }
    catch (err) {
      if (err.status === 409 && err.data && err.data.needs_confirm) {
        const p = err.data.needs_confirm;
        if (!(await confirmBox(p.title, p.message, 'Download', false))) return null;
        body = { ...body, confirm_download: true };
        continue;
      }
      throw err;
    }
  }
}

// ------------------------------------------------------------------ Infer
export async function runInfer(ed) {
  if (!ed.name()) return;
  const a = ed.annotations();
  let started;
  try {
    started = await startJob(`ws/${enc(ed.instance)}/infer`, { name: ed.name(), bboxes: a.bboxes, polygons: a.polygons, polygon: ed.polygon() });
  } catch (err) { toast(err.message); return; }
  if (!started) return;
  const job = await watchJob('Label Assistant', started);
  if (job.state === 'error') { await notice('Label Assistant', job.error); return; }
  ed.apply(job.result);
}

export async function runAutoAnnotate(ed) {
  let settings;
  try { settings = await api(`ws/${enc(ed.instance)}/assistant`); } catch (err) { toast(err.message); return; }
  const only = h('input', { type: 'checkbox' });
  only.checked = settings.batch.only_unlabeled;
  const body = h('div', {},
    h('p', {}, `The Label Assistant (${settings.title}) will label every picture of this dataset and save each one as it goes.`),
    h('label', { class: 'check' }, only, ' Only pictures that have no annotations yet'),
    h('p', { class: 'muted' }, 'Annotations that are already on a picture are never replaced: a prediction that overlaps one is dropped.'));
  const overlay = dialog(`Auto-annotate all  -  ${settings.title}`, body, true);
  const go = await new Promise((resolve) => {
    body.append(h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => resolve(false) }, 'Close'),
      h('button', { class: 'btn primary', onclick: () => resolve(true) }, 'Start')));
  });
  overlay.remove();
  if (!go) return;
  if (!(await ed.saveNow())) return;                       // the run reads and writes the files: what is on screen goes first
  api(`ws/${enc(ed.instance)}/assistant`, 'PUT', { section: 'batch', only_unlabeled: only.checked }).catch(() => {});
  let started;
  try { started = await startJob(`ws/${enc(ed.instance)}/autoannotate`, { only_unlabeled: only.checked, polygon: ed.polygon() }); }
  catch (err) { toast(err.message); return; }
  if (!started) return;
  const job = await watchJob('Auto-annotate all', started, true);
  await ed.reloadAll();
  if (job.state === 'error') { await notice('Auto-annotate all', job.error); return; }
  const r = job.result;
  const failed = r.failed_count ? '\n\nFailed pictures:\n' + r.failed.slice(0, 8).map(([n, m]) => `  ${n}: ${m}`).join('\n') + (r.failed_count > 8 ? '\n  ...' : '') : '';
  await notice('Auto-annotate all', `${r.summary}\n\nTook ${r.seconds} s.${failed}`);
}

export function inferDialog(ed) {
  let scope = 'single';
  try { scope = localStorage.getItem('jelibox-infer-scope') === 'all' ? 'all' : 'single'; } catch (e) { /* private mode */ }
  const body = h('div', {});
  const overlay = dialog('Infer', body);
  const radios = {};
  const choices = [['single', 'Single image', 'Label the picture on screen. You can undo it with Ctrl+Z.'],
    ['all', 'Full dataset', 'Label every picture of this dataset. A window shows the progress.']];
  body.append(h('p', { class: 'muted' }, 'What should Infer label?'));
  for (const [value, title, detail] of choices) {
    const r = h('input', { type: 'radio', name: 'scope', onchange: () => { scope = value; try { localStorage.setItem('jelibox-infer-scope', value); } catch (e) { /* private mode */ } } });
    r.checked = scope === value; radios[value] = r;
    body.append(h('label', { class: 'check big' }, r, ' ', h('b', {}, title)), h('p', { class: 'muted', style: 'margin:0 0 8px 26px' }, detail));
  }
  body.append(h('p', { class: 'muted' }, 'Press G to use the label assist feature with this choice.'),
    h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close'),
      h('button', { class: 'btn primary', onclick: () => { overlay.remove(); runAssist(ed); } }, 'Run now')));
}

// G: what the Infer window is set to
export function runAssist(ed) {
  let scope = 'single';
  try { scope = localStorage.getItem('jelibox-infer-scope') === 'all' ? 'all' : 'single'; } catch (e) { /* private mode */ }
  return scope === 'all' ? runAutoAnnotate(ed) : runInfer(ed);
}

// ------------------------------------------------------------------ settings window
function slider(label, value, min, max, step, digits = 2) {
  const out = h('b', { class: 'val' }, Number(value).toFixed(digits));
  const input = h('input', { type: 'range', min, max, step, oninput: () => { out.textContent = Number(input.value).toFixed(digits); } });
  input.value = value;
  return { node: h('div', { class: 'slider' }, h('span', {}, label), input, out), get: () => Number(input.value) };
}

function field(label, node, hint) {
  return h('div', { class: 'field' }, h('label', {}, label), node, hint ? h('p', { class: 'muted', style: 'font-size:12px;margin:3px 0 0' }, hint) : null);
}

function select(values, current, labels = {}) {
  const el = h('select', {}, values.map((v) => h('option', { value: v }, labels[v] || v)));
  el.value = current;
  return el;
}

function classSelect(classes, current, withSkip) {
  const el = h('select', {}, withSkip ? h('option', { value: '' }, SKIP) : null, classes.map((c) => h('option', { value: c }, c)));
  el.value = current || '';
  return el;
}

// rows of "search for" -> "workspace class"
function targetRows(initial, classes, placeholder) {
  const box = h('div', { class: 'rows' });
  const rows = [];
  function add(prompt = '', cls = '') {
    const text = h('input', { type: 'text', placeholder, maxlength: 40, value: prompt });
    const pick = classSelect(classes, cls || classes[0], false);
    const row = { text, pick };
    row.node = h('div', { class: 'maprow' }, text, h('span', { class: 'arrow' }, '→'), pick,
      h('button', { class: 'icon danger', style: 'opacity:1', title: 'Remove', onclick: () => { rows.splice(rows.indexOf(row), 1); row.node.remove(); } }, '✕'));
    rows.push(row); box.append(row.node);
  }
  (initial.length ? initial : []).forEach((t) => add(t.prompt, t.map_to));
  const node = h('div', {}, box, h('button', { class: 'btn small', onclick: () => add() }, '＋ Add'));
  return { node, get: () => rows.map((r) => ({ prompt: r.text.value, map_to: r.pick.value })) };
}

// "model class -> workspace class" table, filled in when a model is chosen
function mapTable(classes) {
  const box = h('div', { class: 'maptable' });
  let selects = {};
  return {
    node: box,
    set(names, saved) {
      selects = {};
      const lower = Object.fromEntries(classes.map((c) => [c.toLowerCase(), c]));
      box.replaceChildren(...names.map((name) => {
        const pick = classSelect(classes, (saved && saved[name]) || lower[name.toLowerCase()] || '', true);
        selects[name] = pick;
        return h('div', { class: 'maprow' }, h('span', {}, name), h('span', { class: 'arrow' }, '→'), pick);
      }));
    },
    get: () => Object.fromEntries(Object.entries(selects).filter(([, el]) => el.value).map(([k, el]) => [k, el.value])),
    count: () => Object.keys(selects).length,
  };
}

async function inspect(ed, kind, path) {
  return api(`ws/${enc(ed.instance)}/assistant/inspect`, 'POST', { kind, path });
}

function yoloWorldForm(data, ed) {
  const classes = data.classes;
  const provider = { value: data.provider === 'custom_model' ? 'custom_model' : 'yolo_world' };
  const radio = (value, title, detail) => {
    const r = h('input', { type: 'radio', name: 'provider', onchange: () => { provider.value = value; } });
    r.checked = provider.value === value;
    return h('div', {}, h('label', { class: 'check big' }, r, ' ', h('b', {}, title)), h('p', { class: 'muted', style: 'margin:0 0 6px 26px;font-size:12px' }, detail));
  };
  const model = select(data.options.yolo_world_models, data.yolo_world.model);
  const targets = targetRows(data.yolo_world.target_classes, classes, 'e.g. white horse');
  const conf = slider('Confidence threshold', data.confidence, 0.05, 0.95, 0.05);
  const map = mapTable(classes);
  const info = h('p', { class: 'muted' });
  let modelPath = data.custom_model.path || '';
  async function showModel() {
    if (!modelPath) {
      info.textContent = data.own_model ? 'Uses the model you trained in this workspace (classes are matched by position in your class list).'
        : 'Your trained model was not found in this workspace. Train one, or browse to any Ultralytics model.';
      map.node.replaceChildren(); return;
    }
    info.textContent = modelPath;
    try { const r = await inspect(ed, 'model', modelPath); map.set(r.names, data.custom_model.class_map); }
    catch (err) { info.textContent = modelPath + '\n' + err.message; }
  }
  const browse = h('button', { class: 'btn small', onclick: async () => {
    const path = await pickServer({ ext: '.pt', start: '' });
    if (!path) return;
    try { await inspect(ed, 'model', path); } catch (err) { toast(err.message); return; }
    modelPath = path; data.custom_model.class_map = {}; provider.value = 'custom_model';
    form.querySelector('input[name=provider][data-v=custom_model]').checked = true;
    showModel();
  } }, 'Browse custom model ...');
  const clear = h('button', { class: 'btn small', onclick: () => { modelPath = ''; showModel(); } }, "Use the workspace's own model");
  const form = h('div', {},
    radio('yolo_world', 'YOLO-World', 'Zero-shot: describe what to look for, no training needed.'),
    field('Model', model), h('label', {}, 'TARGET CLASSES — search for → save as workspace class'), targets.node,
    h('hr'),
    radio('custom_model', 'My own model', 'Any Ultralytics model: the one you trained in this workspace, or one you choose.'),
    info, h('div', { class: 'row', style: 'margin-top:4px' }, browse, clear), map.node, h('hr'), conf.node);
  form.querySelectorAll('input[name=provider]').forEach((r, i) => r.setAttribute('data-v', ['yolo_world', 'custom_model'][i]));
  showModel();
  return { node: form, collect: () => ({ section: 'yolo_world', provider: provider.value, model: model.value, confidence: conf.get(),
    targets: targets.get(), model_path: modelPath, model_class_map: map.get() }) };
}

function locateForm(data) {
  const la = data.locate_anything, classes = data.classes;
  const targets = targetRows(la.target_classes, classes, 'English or Chinese');
  const mode = select(data.options.locate_modes, la.generation_mode);
  const device = select(data.options.locate_devices, la.device);
  const side = h('input', { type: 'number', min: 256, max: 2048, step: 128, value: la.short_side });
  const passes = h('input', { type: 'number', min: 1, max: 8, value: la.passes });
  const temp = slider('Temperature (passes > 1)', la.temperature, 0, 1, 0.05);
  const node = h('div', {},
    h('p', { class: 'muted' }, 'NVIDIA LocateAnything-3B finds every instance of the categories you list, in one pass over the whole folder.'),
    h('label', {}, 'CATEGORIES — look for → save as workspace class'), targets.node, h('hr'),
    field('Decoding', mode, 'hybrid = best recall, fast = quickest, slow = steadiest on long answers.'), field('Device', device),
    field('Image short side (px)', side), field('Passes (merged by agreement)', passes), temp.node, h('hr'),
    h('pre', { class: 'status' }, (data.status || []).join('\n')));
  return { node, collect: () => ({ section: 'locate', targets: targets.get(), generation_mode: mode.value, device: device.value,
    short_side: side.value, passes: passes.value, temperature: temp.get() }) };
}

function headForm(data, ed) {
  const hd = data.custom_head, classes = data.classes;
  const path = h('input', { type: 'text', value: hd.head_path, placeholder: 'models/…/head_best.pt' });
  const headInfo = h('p', { class: 'muted' });
  const map = mapTable(classes);
  const weights = h('input', { type: 'text', value: hd.detector_weights, placeholder: 'empty = the detector the head was trained on' });
  let detectorNames = data.options.coco_names;
  const ids = h('input', { type: 'text' });
  ids.value = hd.detector_classes.map((i) => detectorNames[i] || String(i)).join(', ');
  const hint = h('p', { class: 'muted', style: 'font-size:12px' });
  const conf = slider('Detector confidence', data.confidence, 0.05, 0.95, 0.05);
  const iou = slider('Detector NMS IoU', hd.iou, 0.1, 0.9, 0.05);
  const minHead = slider('Minimum head confidence', hd.min_head_conf, 0, 0.95, 0.05);
  function showHint() {
    const named = detectorNames.map((n, i) => (n ? `${i} ${n}` : null)).filter(Boolean);
    hint.textContent = `Names or numbers, e.g. ${detectorNames[0] || 0}  -  this detector: ${named.slice(0, 10).join(', ')}${named.length > 10 ? `, ... (${named.length} classes)` : ''}`;
  }
  async function loadHead(p, quiet) {
    try { const r = await inspect(ed, 'head', p); map.set(r.names, hd.class_map); headInfo.textContent = r.info; headInfo.className = 'ok'; }
    catch (err) { headInfo.textContent = err.message.split('\n')[0]; headInfo.className = 'muted'; if (!quiet) toast(err.message); }
  }
  const pickHead = h('button', { class: 'btn small', onclick: async () => { const p = await pickServer({ ext: '.pt' }); if (p) { path.value = p; loadHead(p); } } }, 'Browse head ...');
  const pickDet = h('button', { class: 'btn small', onclick: async () => {
    const p = await pickServer({ ext: '.pt' });
    if (!p) return;
    try { const r = await inspect(ed, 'detector', p); detectorNames = r.all_names; weights.value = p; ids.value = detectorNames[0] || '0'; showHint(); }
    catch (err) { toast(err.message); }
  } }, 'Browse custom model ...');
  showHint();
  if (hd.head_path) loadHead(hd.head_path, true);
  const node = h('div', {},
    h('p', { class: 'muted' }, 'A YOLO detector finds the boxes (person, bottle, ...). The model you trained gives each box one of your own classes.'),
    field('HEAD CHECKPOINT (head_best.pt)', h('div', { class: 'inline' }, path, pickHead), null), headInfo, h('hr'),
    field('Detect these classes', ids, null), hint,
    field('Detector weights (optional)', h('div', { class: 'inline' }, weights, pickDet), 'Empty = the detector the head was trained on. A model you choose here must be that detector (same architecture, same layers): the head reads its features.'),
    conf.node, iou.node, minHead.node, h('hr'), h('label', {}, 'HEAD CLASS → WORKSPACE CLASS'), map.node);
  return { node, collect: () => ({ section: 'head', head_path: path.value.trim(), detector_weights: weights.value.trim(),
    detector_classes: ids.value, iou: iou.get(), min_head_conf: minHead.get(), confidence: conf.get(), class_map: map.get(),
    has_head_names: map.count() > 0 }) };
}

function sam2Form(data, ed) {
  const s = data.sam2_dynamic;
  const labels = Object.fromEntries(data.options.sam2_models.map((m) => [m.id, `${m.id} - ${m.note} (${m.size} MB)`]));
  const model = select(data.options.sam2_models.map((m) => m.id), s.model, labels);
  const size = select(data.options.sam2_sizes.map(String), String(s.imgsz));
  const refs = slider('Reference images', s.max_references, 1, 20, 1, 0);
  const score = slider('Minimum score', s.min_score, 0, 0.5, 0.01);
  const info = h('p', {});
  function show(n) {
    info.textContent = n ? `${n} annotated image(s) remembered. Each is kept once: going back to an image replaces what was remembered for it.`
      : 'None yet. Annotate an image and move on (A / D), or load the images you already labeled.';
  }
  show(data.references);
  const call = async (action) => {
    try { const r = await api(`ws/${enc(ed.instance)}/assistant/references`, 'POST', { action }); show(r.references);
      if (action === 'load' && !r.added) toast('No labeled images were found in this workspace.'); }
    catch (err) { toast(err.message); }
  };
  const node = h('div', {},
    h('p', { class: 'muted' }, 'SAM 2 learns from the images you annotate. Every image you leave becomes a reference, and Infer then finds the same objects in the next image. There is nothing to train.'),
    field('MODEL', model, 'Downloaded once (about the size shown) into models/_sam2. On a CPU use tiny or small.'),
    field('Image size', size, 'smaller = faster, slightly less exact'), h('hr'),
    refs.node, h('p', { class: 'muted', style: 'font-size:12px' }, 'SAM 2 uses the newest annotated images. More references can be steadier but are slower.'),
    score.node, h('p', { class: 'muted', style: 'font-size:12px' }, "SAM 2's scores run low (0.2 is a confident find). Raise this if it labels things that are not there."), h('hr'),
    h('label', {}, 'REFERENCES (kept on the server while it runs)'), info,
    h('div', { class: 'row', style: 'margin-top:4px' },
      h('button', { class: 'btn small', onclick: () => call('load') }, 'Learn from my labeled images'),
      h('button', { class: 'btn small', onclick: () => call('forget') }, 'Forget all')));
  return { node, collect: () => ({ section: 'sam2', model: model.value, imgsz: size.value, max_references: refs.get(), min_score: score.get() }) };
}

export async function settingsDialog(ed) {
  let data;
  try { data = await api(`ws/${enc(ed.instance)}/assistant`); } catch (err) { toast(err.message); return; }
  const builders = { yolo_world: yoloWorldForm, locate_anything: locateForm, custom_head: headForm, sam2_dynamic: sam2Form };
  const form = builders[data.mode](data, ed);
  const error = h('div', { class: 'error', hidden: true });
  const body = h('div', { class: 'form' }, form.node, error);
  const overlay = dialog(`Label Assistant  -  ${data.title}`, body, true);
  overlay.firstChild.classList.add('wide');
  const save = h('button', { class: 'btn primary', onclick: async () => {
    error.hidden = true;
    try {
      const res = await api(`ws/${enc(ed.instance)}/assistant`, 'PUT', form.collect());
      overlay.remove();
      toast(res.warnings.length ? res.warnings[0] : 'Label Assistant settings saved.');
    } catch (err) { error.textContent = err.message; error.hidden = false; error.scrollIntoView({ block: 'nearest' }); }
  } }, 'Save');
  body.append(h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Cancel'), save));
}

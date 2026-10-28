// The Train window: setup (the same fields in every mode, plus the head choice in Custom Model mode) and the progress
// view. The training runs on the server and goes on when this page is closed; opening the window again shows it.

import { h, api, toast, dialog, confirmBox, notice, watchJob, pickServer } from './ui.js';
import { splitBar, splitCounts, augPicker } from './export.js';

const enc = encodeURIComponent;

const YOLO_COLUMNS = [['epoch', 'Epoch'], ['box_loss', 'Box'], ['cls_loss', 'Cls'], ['dfl_loss', 'DFL'], ['precision', 'Precision'],
  ['recall', 'Recall'], ['map50', 'mAP50'], ['map5095', 'mAP50-95'], ['seconds', 'Time']];
const HEAD_COLUMNS = [['epoch', 'Epoch'], ['loss', 'Loss'], ['acc', 'Accuracy'], ['f1', 'Macro-F1'], ['seconds', 'Time']];

function fmtTime(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  const m = Math.floor(seconds / 60), s = String(seconds % 60).padStart(2, '0');
  return seconds >= 3600 ? `${Math.floor(seconds / 3600)}:${String(m % 60).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

function cell(key, value) {
  if (key === 'epoch') return value;
  if (key === 'seconds') return fmtTime(value);
  if (key === 'acc') return `${(value * 100).toFixed(1)}%`;
  return Number(value).toFixed(3);
}

export async function openTrain(ed) {
  if (ed.name()) await ed.saveNow();                       // the labels on screen must be on disk before they are counted
  const base = `ws/${enc(ed.instance)}`;
  let info, mode;
  try { [info, mode] = await Promise.all([api(`${base}/train?polygon=${ed.polygon() ? 1 : 0}`), api('mode')]); }
  catch (err) { toast(err.message); return; }
  const body = h('div', { class: 'form' });
  const overlay = dialog('Train model', body, true);
  overlay.firstChild.classList.add('xwide');
  let timer = null;
  const close = () => { clearTimeout(timer); overlay.remove(); };
  if (info.run) showProgress(info.run); else showSetup();

  // ================================================================== setup
  function showSetup() {
    const headChoice = mode.mode === 'custom_head';
    let kind = headChoice ? 'head' : 'yolo';
    const c = info.counts, classes = info.classes;
    const labelled = () => c.labelled || 0, background = () => c.background || 0;
    const error = h('div', { class: 'error', hidden: true });

    // --- type
    const typeBox = h('div', { class: 'xcard' });
    const radios = [];
    if (headChoice) {
      typeBox.append(h('div', { class: 'xhead' }, h('b', {}, 'TRAINING TYPE')));
      for (const [value, title, note] of [
        ['head', 'Head only (fast)', 'Fits a small head on a frozen detector. Minutes on a CPU. Used by the Custom Model mode.'],
        ['yolo', 'Full YOLO training (every layer)', 'Trains all the weights, with Ultralytics\' own augmentation and loss. Much slower: a GPU is strongly recommended. The model is used by YOLO-World mode as My trained model.']]) {
        const r = h('input', { type: 'radio', name: 'kind', onchange: () => { kind = value; applyKind(); } });
        r.checked = value === kind; radios.push(r);
        typeBox.append(h('label', { class: 'check big' }, r, ' ', h('b', {}, title)), h('p', { class: 'muted', style: 'margin:0 0 6px 26px;font-size:12px' }, note));
      }
    }

    // --- data
    const dataInfo = h('div', {}), counted = h('span', { class: 'muted', style: 'font-size:12px' }), notes = h('div', { style: 'color:var(--amber);font-size:12px' });
    const refresh = h('button', { class: 'btn small', onclick: recount }, '⟳ Refresh');
    const valShare = h('input', { type: 'number', min: 5, max: 50, value: 20, onchange: () => drawData() });
    const split = splitBar({ train: 70, valid: 20, onChange: () => drawData() });
    const bg = h('input', { type: 'checkbox', onchange: () => drawData() }); bg.checked = !!background();
    const bgLabel = h('label', { class: 'check' }, bg, h('span', {}));
    const shuffle = h('input', { type: 'checkbox' });
    const headVal = h('div', { class: 'inline' }, h('span', { class: 'muted' }, 'Validation share (last images)'), valShare, '%');
    const yoloSplit = h('div', {}, split.node, bgLabel, h('label', { class: 'check' }, shuffle, ' Mix the pictures before splitting (random)'));
    const dataCard = h('div', { class: 'xcard' }, h('div', { class: 'xhead' }, h('b', {}, 'DATA')), dataInfo,
      h('div', { class: 'inline', style: 'margin:6px 0' }, refresh, counted), headVal, yoloSplit, notes);

    function drawData() {
      const n = labelled();
      counted.textContent = c.counted ? `Counted ${new Date(c.counted * 1000).toLocaleString()}. Press Refresh after you add, delete or label pictures; the training itself always uses the files as they are when it starts.`
        : 'Not counted yet. Press Refresh to count the pictures (the training counts them itself when it starts).';
      bgLabel.lastChild.textContent = ` Use the ${background()} pictures with nothing in them as background (they teach the model what is NOT there)`;
      bgLabel.style.display = kind === 'yolo' && background() ? '' : 'none';
      if (!c.counted) { dataInfo.textContent = 'The pictures of this workspace were not counted yet.'; notes.textContent = ''; split.setTotal(0); return; }
      let text = `${n} labelled images`;
      if (n >= 2) {
        let tr, va, te;
        if (kind === 'yolo') {
          const total = n + (bg.checked ? background() : 0);
          split.setTotal(total);
          const g = split.get();
          [tr, va, te] = splitCounts(total, g.train, g.valid);
          if (bg.checked && background()) text += ` + ${background()} with nothing in them (background)`;
        } else {
          const share = Math.max(5, Math.min(50, parseInt(valShare.value, 10) || 20));
          va = Math.max(1, Math.round(n * share / 100)); tr = n - va; te = 0;
        }
        text += `: ${tr} to train, ${va} to validate` + (te ? `, ${te} to test.` : '.');
      } else text += '.';
      const per = classes.map((k, i) => [k, (c.boxes || [])[i] || 0]);
      const used = per.filter(([, k]) => k).map(([name, k]) => `${k} ${name}`);
      dataInfo.textContent = text + (used.length ? '\nBoxes per class: ' + used.join(', ') : '');
      dataInfo.style.whiteSpace = 'pre-line';
      const ns = [];
      if (n < info.min_labelled) ns.push(`At least ${info.min_labelled} labelled images are needed.`);
      const unused = per.filter(([, k]) => !k).map(([name]) => name);
      if (unused.length && used.length) ns.push('No examples yet for: ' + unused.join(', ') + ' (it will not learn them).');
      notes.textContent = ns.join('  ');
    }
    async function recount() {
      refresh.disabled = true; refresh.textContent = 'Counting ...';
      try {
        const started = await api(`${base}/train/recount`, 'POST', {});
        await watchJob('Counting the pictures', started);
        const fresh = await api(`${base}/train?polygon=${ed.polygon() ? 1 : 0}`);
        Object.assign(info, fresh); Object.assign(c, fresh.counts);
        bg.checked = !!background();
      } catch (err) { toast(err.message); }
      refresh.disabled = false; refresh.textContent = '⟳ Refresh';
      drawData();
    }

    // --- model
    const modelTitle = h('div', { class: 'xhead' }, h('b', {}));
    // A real list (a text box with a datalist only offers what matches what is already typed, so it looked empty).
    // A file picked with Browse is added to the list as "(custom)".
    const modelList = (models, value, onchange) => {
      const sel = h('select', { style: 'flex:1', onchange }, models.map((m) => h('option', { value: m }, m)));
      choose(sel, value);
      return sel;
    };
    function choose(sel, value) {
      if (![...sel.options].some((o) => o.value === value)) {
        const name = value.split(/[\\/]/).pop();
        sel.append(h('option', { value }, `${name}  (custom file)`));
      }
      sel.value = value;
    }
    const baseIn = modelList(info.stock.yolo, info.defaults.base);
    const detIn = modelList(info.stock.detector, info.defaults.detector, () => updateTaps());
    const taps = h('input', { type: 'text', value: '15, 18, 21', style: 'width:130px' }), tapsHint = h('span', { class: 'muted', style: 'font-size:12px' });
    const cont = h('input', { type: 'checkbox' }); cont.checked = info.has_model; cont.disabled = !info.has_model;
    const browse = h('button', { class: 'btn small', onclick: async () => { const p = await pickServer({ ext: '.pt' }); if (p) { choose(kind === 'yolo' ? baseIn : detIn, p); updateTaps(); } } }, 'Browse custom model ...');
    const modelNote = h('p', { class: 'muted', style: 'font-size:12px' });
    const yoloModel = h('div', { class: 'inline' }, baseIn), headModel = h('div', { class: 'inline' }, detIn);
    const tapsRow = h('div', { class: 'inline' }, h('span', { class: 'muted' }, 'Taps (neck layers)'), taps, tapsHint);
    const contRow = h('label', { class: 'check' }, cont, ' Continue from the model trained here before (keeps what it learned)');
    function updateTaps() {
      const known = info.taps[detIn.value.trim()];
      if (known) { taps.value = known.join(', '); tapsHint.textContent = ''; } else tapsHint.textContent = 'model family not recognised - check these';
    }
    const modelCard = h('div', { class: 'xcard' }, modelTitle, yoloModel, headModel, h('div', { style: 'margin:6px 0' }, browse), modelNote, tapsRow, contRow);

    // --- numbers
    const num = (value, min, max, step = 1, width = '90px') => h('input', { type: 'number', value, min, max, step, style: `width:${width}` });
    const sq = num(info.defaults.imgsz, 32, 2048, 32), hh = num(info.defaults.head_size[0], 32, 2048, 32), ww = num(info.defaults.head_size[1], 32, 2048, 32);
    const epochs = num(info.defaults.epochs, 1, 1000), batch = num(info.defaults.batch, 1, 128);
    const lr = h('input', { type: 'text', value: String(info.defaults.lr), style: 'width:90px' });
    const freeze = num(0, 0, 100), wd = h('input', { type: 'text', value: '0.0005', style: 'width:90px' });
    const cw = h('input', { type: 'checkbox' }); cw.checked = true;
    const flip = h('input', { type: 'checkbox' }); flip.checked = true;
    const row = (label, node, extra) => h('div', { class: 'inline', style: 'margin:5px 0' }, h('span', { class: 'muted', style: 'width:210px;display:inline-block' }, label), node, extra || null);
    const yoloRows = h('div', {}, row('Image size (square)', sq, h('span', { class: 'muted', style: 'font-size:12px' }, 'multiple of 32')),
      row('Freeze first layers', freeze, h('span', { class: 'muted', style: 'font-size:12px' }, '0 = train every layer, 10 = keep the backbone')), row('Weight decay', wd));
    const headRows = h('div', {}, row('Image size (height × width)', h('span', {}, hh, ' × ', ww), h('span', { class: 'muted', style: 'font-size:12px' }, 'multiples of 32')),
      h('label', { class: 'check' }, cw, ' Class weights (rare classes count more)'),
      h('label', { class: 'check' }, flip, ' Flip images horizontally (turn off when logos or text matter)'));
    const gpu = h('p', { style: 'color:var(--amber);font-size:12px' });
    const numbers = h('div', { class: 'xcard' }, h('div', { class: 'xhead' }, h('b', {}, 'TRAINING')), yoloRows, headRows,
      row('Epochs', epochs), row('Batch size', batch), row('Learning rate', lr), gpu);
    function showGpu() {
      gpu.textContent = info.gpu === false ? 'No NVIDIA GPU is available to PyTorch on the server, so training runs on the CPU: expect minutes per epoch for the head, and much longer for a full model. Try fewer epochs first.' : '';
    }
    if (info.gpu === null) {
      const t = setInterval(async () => { try { const g = (await api(`${base}/train?polygon=${ed.polygon() ? 1 : 0}`)).gpu; if (g !== null) { info.gpu = g; showGpu(); clearInterval(t); } } catch (e) { clearInterval(t); } }, 1500);
    }
    showGpu();

    // --- augmentation (full training only)
    const augSlot = h('div', { class: 'muted' }, 'Loading ...');
    const augCard = h('div', { class: 'xcard' }, h('div', { class: 'xhead' }, h('b', {}, 'AUGMENTATION')),
      h('p', { class: 'muted', style: 'font-size:12px' }, 'Add augmented copies of the training pictures (the validation and test ones only if you tick them in the picker). The ones Ultralytics already does while it trains are greyed out in the picker.'),
      augSlot);
    let picker = null;
    augPicker(ed, { training: true }).then((p) => { picker = p; augSlot.replaceWith(p.node); }).catch((e) => { augSlot.textContent = e.message; });

    function applyKind() {
      const yolo = kind === 'yolo';
      modelTitle.firstChild.textContent = yolo ? 'BASE MODEL' : 'DETECTOR';
      yoloModel.style.display = yolo ? '' : 'none'; headModel.style.display = yolo ? 'none' : '';
      tapsRow.style.display = yolo ? 'none' : ''; contRow.style.display = yolo ? '' : 'none';
      yoloRows.style.display = yolo ? '' : 'none'; headRows.style.display = yolo ? 'none' : '';
      headVal.style.display = yolo ? 'none' : ''; yoloSplit.style.display = yolo ? '' : 'none';
      augCard.style.display = yolo ? '' : 'none';
      modelNote.textContent = yolo
        ? `A stock name is downloaded the first time. Training starts from these pretrained weights and changes all of them; pick a -seg model for polygons (this workspace is in ${info.task === 'seg' ? 'polygon' : 'box'} mode).`
        : 'A stock name is downloaded the first time. A model you choose must be the detector you will annotate with: the head reads its features.';
      drawData();
    }
    updateTaps(); applyKind();

    async function start(confirm = false) {
      error.hidden = true;
      const g = split.get();
      const req = kind === 'yolo'
        ? { kind, base: baseIn.value.trim(), imgsz: sq.value, epochs: epochs.value, batch: batch.value, lr: lr.value, freeze: freeze.value,
          weight_decay: wd.value, valid_pct: g.valid, test_pct: g.test, shuffle: shuffle.checked, background: bg.checked && !!background(),
          'continue': cont.checked && info.has_model, augment: picker ? picker.get() : null, polygon: ed.polygon(), confirm }
        : { kind, detector: detIn.value.trim(), taps: taps.value, height: hh.value, width: ww.value, epochs: epochs.value, batch: batch.value,
          lr: lr.value, val_share: valShare.value, class_weights: cw.checked, hflip: flip.checked, polygon: ed.polygon(), confirm };
      try {
        const res = await api(`${base}/train`, 'POST', req);
        showProgress(res);
      } catch (err) {
        if (err.status === 409 && err.data && err.data.needs_confirm) {
          const p = err.data.needs_confirm;
          if (await confirmBox(p.title, p.message, 'Yes, continue')) start(true);
          return;
        }
        error.textContent = err.message; error.hidden = false; error.scrollIntoView({ block: 'nearest' });
      }
    }

    body.replaceChildren(h('p', { class: 'muted' }, kind === 'yolo' ? 'Trains a whole YOLO model on this workspace\'s labels with Ultralytics. Every layer learns your classes and boxes, and the result is a normal YOLO model.'
      : 'Trains a head on this workspace\'s labels: the box is what the detector finds and its class is what the head learns. The detector itself is never changed.'),
    ...(headChoice ? [typeBox] : []), dataCard, modelCard, numbers, augCard, error,
    h('div', { class: 'row' }, h('button', { class: 'btn', onclick: close }, 'Close'), h('button', { class: 'btn primary', onclick: () => start(false) }, 'Start training')));
  }

  // ================================================================== progress
  function showProgress(first) {
    let run = first;
    const title = h('h3', { class: 'runtitle', style: 'margin:0 0 4px' }), status = h('p', { class: 'muted', style: 'white-space:pre-line' });
    const bar = h('div', { class: 'bar' }, h('i', {})), stats = h('pre', { class: 'status' });
    const head = h('thead', {}), tbody = h('tbody', {});
    const table = h('table', { class: 'trainrows' }, head, tbody);
    const log = h('pre', { class: 'status log' });
    const finish = h('button', { class: 'btn primary', onclick: async () => { try { await api('train/finish-early', 'POST', {}); } catch (e) { toast(e.message); } } }, 'Finish early');
    const stop = h('button', { class: 'btn danger', onclick: async () => {
      if (!(await confirmBox('Train model', `Stop the training now?\n\nThe best ${run.kind === 'yolo' ? 'model' : 'head'} so far is kept.`, 'Stop now', true))) return;
      try { await api('train/stop', 'POST', {}); } catch (e) { toast(e.message); }
    } }, 'Stop now');
    const closeBtn = h('button', { class: 'btn', onclick: async () => {
      if (run.state === 'ended') { try { await api('train/dismiss', 'POST', {}); } catch (e) { /* gone */ } close(); return; }
      close(); toast('The training goes on on the server. Press Train to see it again.');
    } }, 'Close');
    let columns = null, shown = 0;
    body.replaceChildren(
      h('p', { class: 'muted', style: 'font-size:12px' }, `Workspace ${run.workspace}  ·  started by ${run.user}`),
      title, status, bar, stats, h('div', { class: 'tablebox' }, table), log,
      h('div', { class: 'row' }, closeBtn, stop, finish));

    function render() {
      const cols = run.kind === 'yolo' ? YOLO_COLUMNS : HEAD_COLUMNS;
      if (columns !== cols) { columns = cols; head.replaceChildren(h('tr', {}, cols.map(([, t]) => h('th', {}, t)))); tbody.replaceChildren(); shown = 0; }
      title.textContent = run.title;
      title.style.color = run.final ? (run.final.kind === 'ok' ? 'var(--green)' : run.final.kind === 'warn' ? 'var(--amber)' : 'var(--danger-fg)') : '';
      status.textContent = run.status;
      const b = run.bar || { done: 0, total: 0 };
      bar.classList.toggle('busy', b.phase === 'download' || (!b.total && run.state !== 'ended'));
      bar.firstChild.style.width = b.total ? Math.min(100, 100 * b.done / b.total) + '%' : '';
      if (run.rows_total) {
        const last = run.rows_all ? run.rows_all[run.rows_all.length - 1] : null;
        const secs = (run.rows_all || []).map((r) => r.seconds);
        const avg = secs.length ? secs.reduce((a, c) => a + c, 0) / secs.length : 0;
        const left = avg && run.state !== 'ended' ? ` (about ${fmtTime(avg * Math.max(0, run.total_epochs - run.rows_total))} left)` : '';
        if (last) {
          stats.textContent = run.kind === 'yolo'
            ? `Epoch ${last.epoch} / ${last.epochs}      box ${last.box_loss.toFixed(3)}  cls ${last.cls_loss.toFixed(3)}  dfl ${last.dfl_loss.toFixed(3)}\nPrecision ${last.precision.toFixed(3)}   Recall ${last.recall.toFixed(3)}   mAP50 ${last.map50.toFixed(3)}   mAP50-95 ${last.map5095.toFixed(3)}   best epoch so far ${run.best_epoch}\n${fmtTime(run.seconds)} elapsed${left}`
            : `Epoch ${last.epoch} / ${last.epochs}      training loss ${last.loss.toFixed(3)}\nValidation: accuracy ${(last.acc * 100).toFixed(1)}%      macro-F1 ${last.f1.toFixed(3)}      best epoch so far ${run.best_epoch}\n${fmtTime(run.seconds)} elapsed${left}`;
        }
      } else {
        stats.textContent = run.state === 'ended' ? '' : `${fmtTime(run.seconds)} elapsed`;
      }
      log.textContent = run.log.join('\n'); log.scrollTop = log.scrollHeight;
      const live = run.state === 'running';
      finish.disabled = !live || run.finishing || run.stopping; finish.textContent = run.finishing ? 'Finishing ...' : 'Finish early';
      stop.disabled = !live || run.stopping; stop.textContent = run.stopping ? 'Stopping ...' : 'Stop now';
      closeBtn.textContent = run.state === 'ended' ? 'Done' : 'Close';
    }
    const allRows = [];
    run.rows_all = allRows;
    function addRows(rows) {
      for (const r of rows) {
        allRows.push(r);
        const best = r.best;
        tbody.append(h('tr', { class: best ? 'best' : '' }, columns.map(([k]) => h('td', {}, cell(k, r[k])))));
      }
      tbody.parentElement.parentElement.scrollTop = 1e9;
    }
    render(); addRows(run.rows); render();
    async function poll() {
      if (!document.body.contains(overlay)) return;
      try {
        const data = await api(`train?rows=${allRows.length}`);
        if (data.run) {
          run = data.run; run.rows_all = allRows;
          addRows(run.rows); render();
          if (run.state === 'ended') return;
        } else { close(); return; }
      } catch (err) { /* a hiccup: try again */ }
      timer = setTimeout(poll, 1000);
    }
    if (run.state !== 'ended') timer = setTimeout(poll, 1000);
  }
}

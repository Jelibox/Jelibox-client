// Analyze Dataset: the numbers come from the server (utils/web/analysis.py), the charts are drawn here as SVG / divs.
// The resize simulation is only calculated on the server; nothing is resized on disk.

import { h, api, toast, dialog, notice, confirmBox, watchJob } from './ui.js';

const enc = encodeURIComponent;
const NS = 'http://www.w3.org/2000/svg';
const PALETTE = ['#4A5BF0', '#1F9D74', '#E08A1E', '#C9353A', '#8E5BD9', '#2B9BC9', '#B8A12A', '#D4568F'];
const SIZES = [320, 416, 512, 640, 800, 960, 1280];
const color = (i) => PALETTE[i % PALETTE.length];
const px = (pair) => (pair ? `${Math.round(pair[0])} x ${Math.round(pair[1])} px` : '-');

function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) el.append(kid);
  return el;
}
const tip = (text) => s('title', {}, text);

// 1, 2, 5 x 10^k steps from 0 up to the first one at or above `max`
function niceTicks(max, count = 5) {
  if (!(max > 0)) return [0];
  const raw = max / count, pow = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / pow;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
  const out = [];
  for (let v = 0; ; v += step) {                                  // the last tick is the first one at or above max
    out.push(Math.round(v * 1e6) / 1e6);
    if (v >= max - step * 0.001) break;
  }
  return out;
}

const W = 520, H = 280, M = { l: 52, r: 14, t: 14, b: 44 };

// An SVG with axes. x / y are the ranges drawn; returns {svg, X, Y} to place things.
function plot({ xmax, ymax, xlabel, ylabel, xticks, yticks, xtext, categories }) {
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img' });
  const X = (v) => M.l + (v / xmax) * (W - M.l - M.r);
  const Y = (v) => H - M.b - (v / ymax) * (H - M.t - M.b);
  for (const t of yticks || niceTicks(ymax)) {
    svg.append(s('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: 'grid' }),
      s('text', { x: M.l - 6, y: Y(t) + 4, class: 'tick', 'text-anchor': 'end' }, String(t)));
  }
  if (!categories) {
    for (const t of xticks || niceTicks(xmax).filter((v) => v <= xmax)) {
      svg.append(s('line', { x1: X(t), x2: X(t), y1: H - M.b, y2: H - M.b + 4, class: 'axis' }),
        s('text', { x: X(t), y: H - M.b + 16, class: 'tick', 'text-anchor': 'middle' }, xtext ? xtext(t) : String(t)));
    }
  }
  svg.append(s('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: 'axis' }),
    s('line', { x1: M.l, x2: M.l, y1: M.t, y2: H - M.b, class: 'axis' }),
    s('text', { x: (M.l + W - M.r) / 2, y: H - 6, class: 'label', 'text-anchor': 'middle' }, xlabel || ''),
    s('text', { x: 12, y: (M.t + H - M.b) / 2, class: 'label', 'text-anchor': 'middle', transform: `rotate(-90 12 ${(M.t + H - M.b) / 2})` }, ylabel || ''));
  return { svg, X, Y };
}

function box(title, ...content) {
  return h('div', { class: 'achart' }, h('h4', {}, title), ...content);
}

const empty = (text) => h('p', { class: 'muted', style: 'margin:30px 0;text-align:center' }, text);

// Horizontal bars as divs: [{label, value, color, tip}]
function bars(rows, total) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  return h('div', { class: 'hbars' }, ...rows.map((r) => h('div', { class: 'hbar', title: r.tip || '' },
    h('span', { class: 'hl' }, r.label),
    h('span', { class: 'hb' }, h('i', { style: `width:${(100 * r.value / max).toFixed(1)}%;background:${r.color || 'var(--accent)'}` })),
    h('span', { class: 'hv' }, String(r.value) + (total ? `  ${(100 * r.value / total).toFixed(1)}%` : '')))));
}

function histogram(counts, edges, { limit, xlabel, ylabel, noun }) {
  const ymax = Math.max(1, ...counts), xmax = edges[edges.length - 1] || 1;
  const { svg, X, Y } = plot({ xmax, ymax: niceTicks(ymax).slice(-1)[0] || ymax, xlabel, ylabel });
  counts.forEach((c, i) => {
    const x0 = X(edges[i]), x1 = X(edges[i + 1]);
    svg.append(s('rect', { x: x0 + 0.5, y: Y(c), width: Math.max(0, x1 - x0 - 1), height: Math.max(0, H - M.b - Y(c)), class: 'bar-fill' },
      tip(`${Math.round(edges[i])}-${Math.round(edges[i + 1])} px: ${c} ${noun}`)));
  });
  if (limit <= xmax) {
    svg.append(s('line', { x1: X(limit), x2: X(limit), y1: M.t, y2: H - M.b, class: 'limit' }),
      s('text', { x: X(limit) + 4, y: M.t + 10, class: 'limit-text' }, `${limit} px`));
  }
  return svg;
}

function scatter(v) {
  const maxX = Math.max(1, ...v.scatter.map((p) => p[0])), maxY = Math.max(1, ...v.scatter.map((p) => p[1]));
  const xt = niceTicks(maxX), yt = niceTicks(maxY);
  const { svg, X, Y } = plot({ xmax: xt[xt.length - 1] || maxX, ymax: yt[yt.length - 1] || maxY, xlabel: 'width px', ylabel: 'height px' });
  const readout = h('div', { class: 'readout muted' }, 'Move over the points.');
  for (const p of v.scatter) svg.append(s('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.4, fill: color(p[2]), 'fill-opacity': 0.6 }));
  const xmax = xt[xt.length - 1] || maxX, ymax = yt[yt.length - 1] || maxY;
  if (v.limit <= xmax) svg.append(s('line', { x1: X(v.limit), x2: X(v.limit), y1: M.t, y2: H - M.b, class: 'limit' }));
  if (v.limit <= ymax) svg.append(s('line', { x1: M.l, x2: W - M.r, y1: Y(v.limit), y2: Y(v.limit), class: 'limit' }));
  svg.addEventListener('mousemove', (e) => {
    const r = svg.getBoundingClientRect(), k = W / r.width;
    const mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k;
    let best = null, d2 = 144;                                   // within 12 units
    for (const p of v.scatter) { const d = (X(p[0]) - mx) ** 2 + (Y(p[1]) - my) ** 2; if (d < d2) { d2 = d; best = p; } }
    readout.textContent = best ? `${v.size_classes[best[2]]}: ${Math.round(best[0])} x ${Math.round(best[1])} px` : 'Move over the points.';
  });
  const note = v.scatter_total > v.scatter.length ? ` (${v.scatter.length} of ${v.scatter_total} objects shown)` : '';
  return box('Width x height' + note, svg, readout);
}

function boxplot(v) {
  const top = Math.max(1, ...v.boxes.map((b) => b.high), v.limit);
  const yt = niceTicks(top), ymax = yt[yt.length - 1] || top;
  const { svg, X, Y } = plot({ xmax: v.boxes.length, ymax, xlabel: '', ylabel: 'px', categories: true });
  const step = (W - M.l - M.r) / v.boxes.length;
  v.boxes.forEach((b, i) => {
    const cx = M.l + step * (i + 0.5), half = Math.min(26, step * 0.3);
    svg.append(s('line', { x1: cx, x2: cx, y1: Y(b.low), y2: Y(b.high), class: 'axis' }),
      s('line', { x1: cx - half / 2, x2: cx + half / 2, y1: Y(b.low), y2: Y(b.low), class: 'axis' }),
      s('line', { x1: cx - half / 2, x2: cx + half / 2, y1: Y(b.high), y2: Y(b.high), class: 'axis' }),
      s('rect', { x: cx - half, y: Y(b.q3), width: half * 2, height: Math.max(1, Y(b.q1) - Y(b.q3)), fill: color(i), 'fill-opacity': 0.8, stroke: 'var(--border)' },
        tip(`${b.name}: lowest ${b.low.toFixed(0)}, median ${b.median.toFixed(0)}, highest ${b.high.toFixed(0)} px`)),
      s('line', { x1: cx - half, x2: cx + half, y1: Y(b.median), y2: Y(b.median), class: 'median' }),
      s('text', { x: cx, y: H - M.b + 16, class: 'tick', 'text-anchor': 'middle' }, b.name.length > 10 ? b.name.slice(0, 9) + '…' : b.name));
  });
  svg.append(s('line', { x1: M.l, x2: W - M.r, y1: Y(v.limit), y2: Y(v.limit), class: 'limit' }));
  return box('Shortest side per class', svg);
}

function lineChart(v) {
  const rows = v.table, sides = rows.map((r) => (r.smallest ? Math.min(...r.smallest) : 0));
  const top = Math.max(1, ...sides, v.limit), yt = niceTicks(top), ymax = yt[yt.length - 1] || top;
  const xmax = rows[rows.length - 1].size;
  const { svg, X, Y } = plot({ xmax, ymax, xlabel: 'model input size (px)', ylabel: 'px', xticks: rows.map((r) => r.size) });
  svg.append(s('polyline', { points: rows.map((r, i) => `${X(r.size)},${Y(sides[i])}`).join(' '), class: 'line' }));
  rows.forEach((r, i) => svg.append(
    s('circle', { cx: X(r.size), cy: Y(sides[i]), r: 3.5, class: 'dot' }, tip(`${r.size} px: ${sides[i].toFixed(1)} px`)),
    s('text', { x: X(r.size), y: Y(sides[i]) - 8, class: 'tick', 'text-anchor': 'middle' }, sides[i].toFixed(1))));
  svg.append(s('line', { x1: M.l, x2: W - M.r, y1: Y(v.limit), y2: Y(v.limit), class: 'limit' }),
    s('text', { x: W - M.r - 4, y: Y(v.limit) - 4, class: 'limit-text', 'text-anchor': 'end' }, `${v.limit} px`));
  return box("Smallest object's shortest side vs input size", svg);
}

function datasetsChart(v) {
  const max = Math.max(1, ...v.instances.map((i) => i.images));
  const cols = v.instances.map((i) => {
    const un = i.images - i.labelled;
    return h('div', { class: 'col', title: `${i.name}: ${i.labelled} labelled, ${un} without objects` },
      h('div', { class: 'stack' },
        h('i', { class: 'none', style: `height:${(100 * un / max).toFixed(1)}%` }),
        h('i', { class: 'lab', style: `height:${(100 * i.labelled / max).toFixed(1)}%` })),
      h('small', {}, i.name.split('-').pop()));
  });
  return box('Images per dataset', h('div', { class: 'cols' }, ...cols),
    h('p', { class: 'muted legend' }, h('span', { class: 'sw lab' }), ' labelled   ', h('span', { class: 'sw none' }), ' no objects'));
}

// ------------------------------------------------------------------ tabs
function overview(v) {
  const total = v.class_counts.reduce((a, c) => a + c.count, 0);
  return h('div', { class: 'agrid' },
    box('Objects per class', v.class_counts.length
      ? bars(v.class_counts.map((c, i) => ({ label: c.name, value: c.count, color: color(i), tip: `${c.name}: ${c.count}` })), total)
      : empty('No annotated objects yet')),
    v.instances.length ? datasetsChart(v) : null);
}

function sizeTab(v) {
  const where = v.size ? `at ${v.size} px` : 'original';
  if (!v.objects || !v.hist_w.length) return empty('No objects');
  return h('div', { class: 'agrid' },
    box(`Width of objects (${where})`, histogram(v.hist_w, v.hist_edges, { limit: v.limit, xlabel: 'px (the last column also holds everything larger)', ylabel: 'objects', noun: 'objects' })),
    box(`Height of objects (${where})`, histogram(v.hist_h, v.hist_edges, { limit: v.limit, xlabel: 'px (the last column also holds everything larger)', ylabel: 'objects', noun: 'objects' })),
    scatter(v), boxplot(v),
    h('p', { class: 'muted legend wide' }, ...v.size_classes.flatMap((c, i) => [h('span', { class: 'sw', style: `background:${color(i)}` }), ` ${c}   `])));
}

function imagesTab(v) {
  const res = v.resolutions.length
    ? bars(v.resolutions.map((r) => ({ label: r.label, value: r.count, tip: `${r.label}: ${r.count} image(s)` })))
    : empty('No image sizes in the XML files');
  let perImage = empty('No labelled images yet');
  if (v.per_image.length) {
    const ymax = niceTicks(Math.max(1, ...v.per_image.map((p) => p.images))).slice(-1)[0] || 1;
    const n = v.per_image.length;
    const { svg, X, Y } = plot({ xmax: n + 1, ymax, xlabel: 'objects in the image (40 = 40 or more)', ylabel: 'images', xticks: [], categories: true });
    v.per_image.forEach((p, i) => {
      const x0 = X(i + 0.5), x1 = X(i + 1.5);
      svg.append(s('rect', { x: x0 + 1, y: Y(p.images), width: Math.max(1, x1 - x0 - 2), height: Math.max(0, H - M.b - Y(p.images)), class: 'bar-fill' }, tip(`${p.objects} object(s): ${p.images} image(s)`)));
      if (n <= 20 || (i + 1) % 5 === 0 || i === 0) svg.append(s('text', { x: (x0 + x1) / 2, y: H - M.b + 16, class: 'tick', 'text-anchor': 'middle' }, String(p.objects)));
    });
    perImage = svg;
  }
  return h('div', { class: 'agrid' },
    box('Image resolutions' + (v.resolutions_more ? ` (top 10, ${v.resolutions_more} more in other sizes)` : ''), res),
    box('Objects per image', perImage));
}

function tableTab(v) {
  const mode = v.mode === 'letterbox' ? 'letterbox' : 'stretch';
  const rows = v.table.map((r) => h('tr', {}, h('td', {}, `${r.size} px`), h('td', {}, px(r.smallest)),
    h('td', {}, r.median_side === null ? '-' : `${r.median_side.toFixed(1)} px`), h('td', {}, String(r.small)), h('td', {}, `${r.small_percent.toFixed(1)}%`)));
  return h('div', {},
    h('p', { style: 'font-weight:600' }, `Where does the smallest object end up for each model input size? (${mode}, calculated)`),
    h('table', { class: 'atable' }, h('thead', {}, h('tr', {}, ...['Input size', 'Smallest object', 'Median shortest side', `Objects < ${v.limit} px`, 'Share'].map((t) => h('th', {}, t)))), h('tbody', {}, ...rows)),
    lineChart(v));
}

const TABS = [['overview', 'Overview', overview], ['size', 'Object size', sizeTab], ['images', 'Images', imagesTab], ['table', 'Resize table', tableTab]];

// ------------------------------------------------------------------ the window
export async function openAnalyze(workspace) {
  let started;
  try { started = await api(`workspaces/${enc(workspace)}/analysis`, 'POST', {}); }
  catch (err) { toast(err.message); return; }
  const job = await watchJob(`Analyzing "${workspace}"`, started);
  if (job.state !== 'done') { if (job.state === 'error') notice('Analyze Dataset', job.error || 'The analysis failed.'); return; }
  show(workspace, job.result.token, job.result.view);
}

function show(workspace, token, first) {
  let v = first, tab = 'overview';
  const sizeIn = h('input', { type: 'number', min: 16, max: 8192, value: v.size || 640, list: 'analyze-sizes', style: 'width:90px' });
  const sizeList = h('datalist', { id: 'analyze-sizes' }, ...SIZES.map((n) => h('option', { value: n })));
  const mode = h('select', { onchange: () => ask({ size: v.size }) },
    h('option', { value: 'letterbox' }, 'Letterbox (keep ratio)'), h('option', { value: 'stretch' }, 'Stretch to square'));
  const limitIn = h('input', { type: 'number', min: 1, max: 1024, value: v.limit, style: 'width:70px', onchange: () => ask({ size: v.size }) });
  const simBtn = h('button', { class: 'btn primary', onclick: () => ask({ size: sizeIn.value }) }, 'Simulate');
  const resetBtn = h('button', { class: 'btn', onclick: () => ask({ size: null }) }, 'Original sizes');
  const cards = h('div', { class: 'cards' }), alert = h('div', { class: 'strip', hidden: true }), tabs = h('div', { class: 'tabs' }), page = h('div', { class: 'apage' });
  const sub = h('span', { class: 'muted' });
  const body = h('div', {},
    h('div', { class: 'row abar' }, sub, h('div', { class: 'spacer' })),
    h('div', { class: 'row abar' },
      h('b', {}, 'Resize simulation'), 'input size', sizeIn, 'px', mode, simBtn, resetBtn, h('div', { class: 'spacer' }),
      h('b', {}, 'Small object limit'), limitIn, 'px'),
    cards, alert, tabs, page, sizeList);
  const overlay = dialog(`Analyze Dataset - ${workspace}`, body, true);
  overlay.firstChild.classList.add('analyze');
  overlay.firstChild.append(h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => overlay.remove() }, 'Close')));

  const card = (title, value, note = '') => h('div', { class: 'acard' }, h('small', {}, title.toUpperCase()), h('b', {}, value), h('span', {}, note || ' '));

  function render() {
    sub.textContent = `${v.images} images in ${v.datasets} dataset(s), read from the VOC XML files`;
    mode.value = v.mode;
    limitIn.value = v.limit;
    simBtn.textContent = v.size ? `Simulating ${v.size} px` : 'Simulate';
    const list = [card('Images', String(v.images), `${v.labelled} labelled, ${v.unlabelled} without objects`),
      card('Objects', String(v.objects), `${v.classes} classes`)];
    if (v.size) {
      list.push(card('Smallest object', px(v.smallest), `was ${px(v.smallest_original)}  (at ${v.size} px)`));
      if (v.not_simulated) list.push(card('Not simulated', String(v.not_simulated), 'image size unknown'));
    } else list.push(card('Smallest object', px(v.smallest_original), 'original image sizes'));
    if (v.median) list.push(card('Median object', `${v.median[0]} x ${v.median[1]} px`, 'width x height'));
    cards.replaceChildren(...list);

    alert.hidden = !v.small;
    if (v.small) {
      const where = v.size ? `at ${v.size} px` : 'in the original images';
      const worst = v.small.by_class.slice(0, 4).map((c) => `${c.name} (${c.count})`).join(', ');
      alert.replaceChildren(h('span', {}, `⚠  ${v.small.count} object(s) (${v.small.percent.toFixed(1)}%) have a side under ${v.limit} px ${where}.  Smallest: ${px(v.small.smallest)}.  Most affected: ${worst}.`),
        h('button', { class: 'btn', onclick: removeSmall }, 'Remove these objects'));
    }
    tabs.replaceChildren(...TABS.map(([key, title]) => h('button', { class: 'tab' + (tab === key ? ' on' : ''), onclick: () => { tab = key; render(); } }, title)));
    page.replaceChildren(TABS.find((t) => t[0] === tab)[2](v));
  }

  async function ask({ size }) {
    try {
      v = await api(`analysis/${token}/view`, 'POST', { size: size === null || size === '' ? null : Number(size), mode: mode.value, limit: Number(limitIn.value) });
    } catch (err) { toast(err.message); limitIn.value = v.limit; return; }
    render();
  }

  async function removeSmall() {
    const info = v.small;
    if (!info) return;
    const where = v.size ? `measured at ${v.size} px, ${v.mode}` : 'measured in the original images';
    const ok = await confirmBox('Remove small objects',
      `This will MODIFY the dataset of workspace "${workspace}": ${info.count} object(s) with a side under ${v.limit} px (${where}) will be permanently deleted from the annotation files (the Pascal VOC XML files and their YOLO labels). The images themselves are not touched. This cannot be undone.\n\nMake sure you have backed up the current dataset before continuing (Export Dataset can save a copy).\n\nDelete these objects now?`,
      'Delete objects', true);
    if (!ok) return;
    let started;
    try { started = await api(`analysis/${token}/remove-small`, 'POST', { size: v.size, mode: v.mode, limit: v.limit }); }
    catch (err) { notice('Remove small objects', err.message); return; }
    const job = await watchJob('Removing small objects ...', started);
    if (job.state !== 'done') { notice('Remove small objects', `Removing the objects failed:\n${job.error || job.state}`); return; }
    v = job.result.view;
    render();
    let text = `Removed ${job.result.objects} object(s) from ${job.result.files} annotation file(s).`;
    if (job.result.emptied) text += `\n${job.result.emptied} image(s) now have no objects.`;
    notice('Remove small objects', text);
  }

  render();
}

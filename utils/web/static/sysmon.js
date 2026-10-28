// Resource monitor of the server computer, on the workspace page: CPU / RAM / GPU use, CPU / GPU temperature, storage.
// One panel for the whole page session: it keeps its history while you move between workspaces, polls every 2 seconds
// while it is on screen and does nothing while the editor or a hidden tab is showing.

import { h, api } from './ui.js';

const POLL_MS = 2000;
const KEEP = 60;                                           // samples in a chart: two minutes
const history = { cpu: [], ram: [], gpu: [], cpuTemp: [], gpuTemp: [], disk: [] };
let panel = null, failures = 0;

export function fmtBytes(n) {
  if (n === null || n === undefined) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n >= 100 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}

// green / amber / red from the value and where each kind of number turns worrying
export function level(kind, value) {
  if (value === null || value === undefined) return 'na';
  const [warn, hot] = kind === 'temp' ? [70, 85] : [60, 85];
  return value >= hot ? 'hot' : value >= warn ? 'warn' : 'ok';
}

const NS = 'http://www.w3.org/2000/svg';
function spark(values) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 100 26'); svg.setAttribute('class', 'spark'); svg.setAttribute('preserveAspectRatio', 'none');
  if (values.length > 1) {
    const step = 100 / (KEEP - 1), left = 100 - step * (values.length - 1);
    const line = document.createElementNS(NS, 'polyline');
    line.setAttribute('points', values.map((v, i) => `${(left + i * step).toFixed(1)},${(25 - Math.max(0, Math.min(100, v)) / 100 * 24).toFixed(1)}`).join(' '));
    svg.append(line);
  }
  return svg;
}

function tile(title) {
  const value = h('b', {}, '...'), sub = h('span', { class: 'sub' }, ' '), fill = h('i', {}), chart = h('div', { class: 'chart' });
  const el = h('div', { class: 'mon na' }, h('small', {}, title), value, h('div', { class: 'meter' }, fill), chart, sub);
  return {
    el,
    set({ text, note, percent, kind = 'load', series, tip }) {
      value.textContent = text;
      sub.textContent = note || ' ';
      fill.style.width = percent === null || percent === undefined ? '0%' : Math.min(100, percent) + '%';
      el.className = 'mon ' + level(kind, percent);
      el.title = tip || '';
      chart.replaceChildren(spark(series || []));
    },
  };
}

function push(key, value) {
  const list = history[key];
  list.push(value === null || value === undefined ? 0 : value);
  if (list.length > KEEP) list.shift();
}

function build() {
  const tiles = {
    cpu: tile('CPU USAGE'), ram: tile('RAM USAGE'), cpuTemp: tile('CPU TEMPERATURE'),
    gpu: tile('GPU USAGE'), gpuTemp: tile('GPU TEMPERATURE'), disk: tile('STORAGE'),
  };
  const state = h('span', { class: 'muted' }, 'connecting ...');
  const node = h('section', { class: 'monitor' },
    h('div', { class: 'mon-head' }, h('b', {}, 'Server resources'), state),
    h('div', { class: 'mon-grid' }, tiles.cpu.el, tiles.ram.el, tiles.cpuTemp.el, tiles.gpu.el, tiles.gpuTemp.el, tiles.disk.el));
  node.update = (snap) => {
    const cpu = snap.cpu, ram = snap.ram, gpus = snap.gpus || [], gpu = gpus[0] || null, disks = snap.disks || [];
    push('cpu', cpu.percent); push('ram', ram.percent); push('cpuTemp', cpu.temp);
    push('gpu', gpu && gpu.percent); push('gpuTemp', gpu && gpu.temp);
    const main = disks.find((d) => d.here) || disks[0] || null;
    push('disk', main && main.percent);
    tiles.cpu.set({ text: cpu.percent === null ? 'n/a' : `${cpu.percent.toFixed(0)}%`, percent: cpu.percent, series: history.cpu,
      note: cpu.cores ? `${cpu.cores} threads` : '' });
    tiles.ram.set({ text: ram.percent === null ? 'n/a' : `${ram.percent.toFixed(0)}%`, percent: ram.percent, series: history.ram,
      note: ram.total ? `${fmtBytes(ram.used)} of ${fmtBytes(ram.total)}` : '' });
    tiles.cpuTemp.set({ text: cpu.temp === null ? 'n/a' : `${cpu.temp.toFixed(0)} °C`, percent: cpu.temp, kind: 'temp', series: history.cpuTemp,
      note: cpu.temp === null ? 'This computer does not report it' : '' });
    const more = gpus.length > 1 ? `  (+${gpus.length - 1} more)` : '';
    const gpuTip = gpus.map((g, i) => `GPU ${i}: ${g.name}  ${g.percent ?? '-'}%  ${g.temp ?? '-'} °C`
      + (g.mem_total ? `  ${fmtBytes(g.mem_used)} / ${fmtBytes(g.mem_total)}` : '')).join('\n');
    tiles.gpu.set({ text: gpu && gpu.percent !== null ? `${gpu.percent.toFixed(0)}%` : 'n/a', percent: gpu ? gpu.percent : null, series: history.gpu, tip: gpuTip,
      note: gpu ? `${gpu.name}${gpu.mem_total ? ` · ${fmtBytes(gpu.mem_used)} / ${fmtBytes(gpu.mem_total)}` : ''}${more}` : 'No NVIDIA or AMD GPU found' });
    tiles.gpuTemp.set({ text: gpu && gpu.temp !== null ? `${gpu.temp.toFixed(0)} °C` : 'n/a', percent: gpu ? gpu.temp : null, kind: 'temp', series: history.gpuTemp, tip: gpuTip,
      note: gpu ? gpu.name : 'No NVIDIA or AMD GPU found' });
    const driveTip = disks.map((d) => `${d.mount}  ${d.percent.toFixed(0)}%  (${fmtBytes(d.used)} of ${fmtBytes(d.total)})${d.here ? '  - Jelibox is here' : ''}`).join('\n');
    tiles.disk.set({ text: main ? `${main.percent.toFixed(0)}%` : 'n/a', percent: main ? main.percent : null, series: history.disk, tip: driveTip,
      note: main ? `${fmtBytes(main.used)} of ${fmtBytes(main.total)} · ${main.mount}${disks.length > 1 ? `  (+${disks.length - 1} more)` : ''}` : '' });
    state.textContent = `updates every ${POLL_MS / 1000} s`;
  };
  node.offline = () => { state.textContent = 'no answer from the server'; };
  return node;
}

async function tick() {
  if (!panel || !panel.isConnected || document.hidden) return;
  try {
    panel.update(await api('system'));
    failures = 0;
  } catch (err) {
    if (++failures >= 2) panel.offline();
  }
}

// The one panel of the page; put it wherever it should show (moving it around keeps its history).
export function monitorPanel() {
  if (!panel) {
    panel = build();
    setInterval(tick, POLL_MS);
    setTimeout(tick, 200);
  }
  setTimeout(tick, 200);                                       // just put on screen: draw without waiting for the next poll
  return panel;
}

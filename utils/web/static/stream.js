// Stream: run the workspace's trained model on a video and watch the annotated pictures. The videos are kept on the
// server in temp/stream (a file sent from this device, or a YouTube video the server downloads); Clear cache removes them.

import { h, api, toast, dialog, notice, confirmBox, watchJob } from './ui.js';

const enc = encodeURIComponent;
const mb = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + ' MB');

// XMLHttpRequest, because fetch cannot say how much of a big video has gone up.
function upload(file, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', '/api/stream/upload?name=' + enc(file.name));
    xhr.setRequestHeader('X-Jelibox', '1');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) progress(e.loaded, e.total); };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.error || xhr.statusText || 'The upload failed.'));
    };
    xhr.onerror = () => reject(new Error('The upload was interrupted.'));
    xhr.send(file);
  });
}

export async function openStream(ed) {
  const base = `ws/${enc(ed.instance)}`;
  let info;
  try { info = await api(`${base}/stream`); } catch (err) { toast(err.message); return; }
  let session = null, timer = null, confTimer = null;

  // ---------------------------------------------------------------- the source
  const list = h('select', { class: 'stream-list', size: 6, onchange: () => buttons() });
  const fileInput = h('input', { type: 'file', accept: 'video/*,.mkv,.avi,.mov,.m4v', onchange: () => pick() });
  const upText = h('div', { class: 'muted', hidden: true }), upBar = h('div', { class: 'bar', hidden: true }, h('i', {}));
  const ytInput = h('input', { type: 'url', placeholder: 'https://www.youtube.com/watch?v=...' });
  const ytBtn = h('button', { class: 'btn', onclick: () => youtube() }, 'Download');
  const cacheText = h('span', { class: 'muted' });
  const clearBtn = h('button', { class: 'btn small', title: 'Delete the uploaded and downloaded videos and the junk in the temp folder', onclick: () => clearCache() }, 'Clear cache');

  const conf = h('input', { type: 'range', min: 0.1, max: 0.95, step: 0.05, value: 0.4, oninput: () => changeConf() });
  const confOut = h('b', {}, '0.40');
  const startBtn = h('button', { class: 'btn primary', onclick: () => start() }, '▶  Start');
  const stopBtn = h('button', { class: 'btn', onclick: () => stop() }, '■  Stop');

  const side = h('div', { class: 'stream-side' },
    info.model ? h('div', { class: 'muted' }, 'Model: ', h('b', {}, info.model))
      : h('div', { class: 'note' }, 'This workspace has no trained model yet. Train one first (the Train button), or choose a model in the Label Assistant.'),
    h('h4', {}, 'Video file'),
    fileInput, upBar, upText,
    h('h4', {}, 'YouTube'),
    info.youtube ? h('div', { class: 'inline' }, ytInput, ytBtn)
      : h('div', { class: 'note' }, 'yt-dlp is not installed on the server, so YouTube links do not work. Upload a video file instead.'),
    h('h4', {}, 'Videos on the server'),
    list,
    h('div', { class: 'inline', style: 'flex-wrap:wrap' }, cacheText, h('span', { class: 'spacer' }), clearBtn),
    h('h4', {}, 'Confidence'),
    h('div', { class: 'slider2' }, h('span', {}, 'Show detections above'), confOut), conf,
    h('div', { class: 'row', style: 'margin-top:8px' }, startBtn, stopBtn));

  // ---------------------------------------------------------------- the picture and its numbers
  const placeholder = () => h('div', { class: 'muted', style: 'padding:24px;text-align:center' },
    'Upload a video or download one from YouTube, pick it in the list and press Start.');
  const view = h('div', { class: 'stream-view' }, placeholder());
  const num = () => h('b', {}, '0');
  const [nFps, nFrames, nDet, nLoops] = [num(), num(), num(), num()];
  const stat = (label, el) => h('div', {}, el, h('small', { class: 'muted' }, label));
  const counts = h('div', { class: 'muted', style: 'margin-top:8px;min-height:1.4em' });
  const message = h('div', { class: 'error', hidden: true, style: 'margin-top:8px' });
  const main = h('div', { class: 'stream-main' }, view,
    h('div', { class: 'stream-stats' }, stat('FPS', nFps), stat('Frames', nFrames), stat('Detections', nDet), stat('Loops', nLoops)),
    counts, message);

  const body = h('div', {}, h('div', { class: 'stream' }, side, main));
  const overlay = dialog('Stream', body, true);
  overlay.firstChild.classList.add('xwide');
  body.append(h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => close() }, 'Close')));

  // ---------------------------------------------------------------- what is on screen
  const running = () => !!session;
  function buttons() {
    startBtn.disabled = running() || !list.value || !info.model;
    stopBtn.disabled = !running();
    list.disabled = fileInput.disabled = ytInput.disabled = ytBtn.disabled = clearBtn.disabled = running();
  }
  function drawVideos(select) {
    const keep = select || list.value;
    list.replaceChildren(...info.videos.map((v) => h('option', { value: v.name }, `${v.name}   ·   ${mb(v.bytes)}`)));
    if (info.videos.some((v) => v.name === keep)) list.value = keep;
    else if (info.videos.length) list.selectedIndex = 0;
    const c = info.cache, parts = [];
    if (c.videos) parts.push(`${c.videos} video${c.videos === 1 ? '' : 's'} (${mb(c.video_bytes)})`);
    if (c.junk) parts.push(`${c.junk} leftover file${c.junk === 1 ? '' : 's'} (${mb(c.junk_bytes)})`);
    cacheText.textContent = parts.length ? 'Cache: ' + parts.join(' + ') : 'The cache is empty.';
    buttons();
  }
  async function refresh(select) {
    try { info = await api(`${base}/stream`); } catch (err) { toast(err.message); return; }
    drawVideos(select);
  }
  function say(text, isError = true) { message.hidden = !text; message.textContent = text || ''; message.className = isError ? 'error' : 'muted'; }

  // ---------------------------------------------------------------- upload and download
  async function pick() {
    const file = fileInput.files[0];
    if (!file) return;
    say('');
    upBar.hidden = upText.hidden = false;
    upBar.firstChild.style.width = '0';
    try {
      const saved = await upload(file, (done, total) => {
        upBar.firstChild.style.width = (100 * done / total) + '%';
        upText.textContent = `Uploading ${file.name}: ${mb(done)} of ${mb(total)}`;
      });
      toast(`${saved.name} uploaded`);
      await refresh(saved.name);
    } catch (err) { say(err.message); }
    fileInput.value = '';
    upBar.hidden = upText.hidden = true;
  }
  async function youtube() {
    const url = ytInput.value.trim();
    if (!url) { toast('Paste a YouTube link first.'); return; }
    say('');
    ytBtn.disabled = true;
    try {
      const started = await api('stream/youtube', 'POST', { url });
      const job = await watchJob('Download from YouTube', started, true);
      if (job.state === 'error') await notice('Download failed', job.error);
      else if (job.state === 'cancelled') toast('Download stopped');
      else { ytInput.value = ''; await refresh(job.result.name); toast(`${job.result.name} downloaded`); }
    } catch (err) { say(err.message); }
    buttons();
  }
  async function clearCache() {
    const c = info.cache;
    if (!c.videos && !c.junk) { toast('The cache is already empty.'); return; }
    const lines = [];
    if (c.videos) lines.push(`${c.videos} video${c.videos === 1 ? '' : 's'} (${mb(c.video_bytes)}) in the server's temp/stream folder`);
    if (c.junk) lines.push(`${c.junk} leftover file${c.junk === 1 ? '' : 's'} (${mb(c.junk_bytes)}): unfinished downloads, and exports or uploads nobody picked up`);
    if (!(await confirmBox('Clear cache', 'This deletes from the server:\n\n• ' + lines.join('\n• ') + '\n\nYour datasets, annotations and models are not touched.', 'Delete', true))) return;
    try {
      const r = await api('stream/clear-cache', 'POST', {});
      toast(`Removed ${r.removed} item${r.removed === 1 ? '' : 's'}, freed ${mb(r.bytes)}` + (r.failed ? ` (${r.failed} could not be removed)` : ''));
    } catch (err) { toast(err.message); }
    await refresh();
  }

  // ---------------------------------------------------------------- the stream
  async function start() {
    say('');
    startBtn.disabled = true;
    try {
      session = await api(`${base}/stream/start`, 'POST', { video: list.value, conf: Number(conf.value) });
    } catch (err) { say(err.message); buttons(); return; }
    attach();
  }
  function attach() {
    buttons();
    view.replaceChildren(h('div', { class: 'muted', style: 'padding:24px' }, 'Loading the model ...'));
    const img = h('img', { alt: 'The video with the model\'s detections', src: `/api/stream/${session.id}/feed` });
    img.addEventListener('load', () => view.replaceChildren(img), { once: true });
    timer = setInterval(poll, 1000);
    poll();
  }
  async function poll() {
    if (!session) return;
    let s;
    try { s = await api('stream/' + session.id); } catch (err) { finish({ state: 'error', error: err.message }); return; }
    nFps.textContent = s.fps.toFixed(1); nFrames.textContent = s.frames; nDet.textContent = s.detections; nLoops.textContent = s.loops;
    counts.textContent = Object.entries(s.counts).map(([k, v]) => `${v}× ${k}`).join('   ·   ') || (s.state === 'running' ? 'No detections in this picture' : '');
    if (s.state === 'ended' || s.state === 'error') finish(s);
  }
  function finish(s) {
    clearInterval(timer); timer = null; session = null;
    view.replaceChildren(placeholder());
    if (s.state === 'error') say(s.error);
    else if (s.note) say(s.note, false);
    buttons();
  }
  async function stop() {
    if (!session) return;
    stopBtn.disabled = true;
    try { await api(`stream/${session.id}/stop`, 'POST', {}); } catch (err) { toast(err.message); }
    setTimeout(poll, 300);
  }
  function changeConf() {
    confOut.textContent = Number(conf.value).toFixed(2);
    if (!session) return;
    clearTimeout(confTimer);
    confTimer = setTimeout(() => api('stream/' + session.id, 'POST', { conf: Number(conf.value) }).catch(() => {}), 200);
  }
  function close() {
    clearInterval(timer); clearTimeout(confTimer);
    if (session) api(`stream/${session.id}/stop`, 'POST', {}, { keepalive: true }).catch(() => {});
    overlay.remove();
  }

  drawVideos();
  if (info.session) { session = info.session; conf.value = session.conf; confOut.textContent = session.conf.toFixed(2); attach(); }
}

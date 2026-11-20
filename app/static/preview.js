/* SPDX-License-Identifier: MPL-2.0 */
(() => {
  'use strict';
  const panel = document.createElement('aside');
  panel.id = 'source-preview';
  panel.hidden = true;
  panel.setAttribute('aria-label', 'Source document preview');
  panel.innerHTML = `
    <div id="preview-resize" role="separator" aria-label="Resize source preview" aria-orientation="vertical" tabindex="0"></div>
    <div class="preview-heading"><div class="bar">
      <h2 id="preview-title"></h2><button id="preview-close" class="ghost" aria-label="Close source preview">Close</button>
    </div></div>
    <div class="preview-toolbar">
      <button id="preview-original" class="ghost" aria-pressed="false">Original PDF</button>
      <button id="preview-text" class="ghost" aria-pressed="false">Indexed text</button>
      <button id="preview-prev" class="ghost" aria-label="Previous page">Previous</button>
      <span id="preview-location"></span>
      <button id="preview-next" class="ghost" aria-label="Next page">Next</button>
      <button id="preview-jump" class="ghost">Back to citation</button>
    </div>
    <p id="preview-warning" role="status"></p>
    <p id="preview-status" role="status"></p>
    <div id="preview-body" tabindex="0" aria-label="Document content"></div>`;
  document.body.appendChild(panel);
  const el = id => document.getElementById(id);
  let controller, sequence = 0, source, data, mode, page = 1, pages = 1, opener;
  let textOffset = 0;
  const textPageSize = 500;

  function cancel() { controller?.abort(); controller = new AbortController(); return ++sequence; }
  async function post(url, body) {
    const response = await fetch(url, {method: 'POST', signal: controller.signal,
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Unable to open this source.');
    return result;
  }
  function controls(loading = false) {
    const pdf = mode === 'pdf';
    el('preview-original').disabled = !data || data.kind !== 'pdf';
    el('preview-text').disabled = !data;
    el('preview-original').setAttribute('aria-pressed', String(pdf));
    el('preview-text').setAttribute('aria-pressed', String(mode === 'text'));
    el('preview-prev').disabled = !data || loading || (pdf ? page <= 1 : textOffset === 0);
    el('preview-next').disabled = !data || loading || (pdf ? page >= pages : textOffset + textPageSize >= data.lines.length);
    el('preview-jump').disabled = !data || loading;
  }
  function scrollToMatch(node) {
    if (!node) return;
    const body = el('preview-body');
    body.scrollTop += node.getBoundingClientRect().top - body.getBoundingClientRect().top - body.clientHeight / 3;
  }
  function showText(jump = true) {
    cancel(); mode = 'text';
    if (jump) {
      const index = Math.max(0, data.lines.findIndex(line => line.n >= source.start_line));
      textOffset = Math.max(0, index - 50);
    }
    const shown = data.lines.slice(textOffset, textOffset + textPageSize);
    const container = document.createElement('div'); container.className = 'preview-lines';
    for (const line of shown) {
      const row = document.createElement('div'); row.className = 'preview-line';
      if (line.n >= source.start_line && line.n <= source.end_line) row.classList.add('cited');
      const number = document.createElement('span'); number.className = 'line-number'; number.textContent = line.n;
      const text = document.createElement('span'); text.className = 'line-text'; text.textContent = line.text || ' ';
      row.append(number, text); container.append(row);
    }
    el('preview-body').replaceChildren(container);
    el('preview-location').textContent = `Lines ${shown[0]?.n || 0}–${shown.at(-1)?.n || 0}`;
    el('preview-status').textContent = 'Indexed text • highlighted lines were cited. Formatting follows extraction; omitted blank lines may leave gaps in numbering.';
    controls();
    el('preview-body').scrollTop = 0;
    if (jump) requestAnimationFrame(() => scrollToMatch(container.querySelector('.cited')));
  }
  async function showPage(number) {
    const ticket = cancel(); mode = 'pdf'; page = number;
    controls(true);
    el('preview-location').textContent = `Page ${page}`;
    el('preview-status').textContent = 'Rendering page locally…';
    el('preview-body').replaceChildren();
    try {
      const result = await post('/preview/page', {...source, version: data.version, page});
      if (ticket !== sequence) return;
      pages = result.pages;
      const container = document.createElement('div'); container.className = 'preview-page';
      const image = document.createElement('img'); image.alt = `${source.path}, page ${page}`;
      image.onload = () => { if (ticket === sequence) scrollToMatch(container.querySelector('.preview-box')); };
      container.append(image);
      for (const box of result.boxes) {
        const mark = document.createElement('div'); mark.className = 'preview-box';
        mark.style.left = `${box.x * 100}%`; mark.style.top = `${box.y * 100}%`;
        mark.style.width = `${box.w * 100}%`; mark.style.height = `${box.h * 100}%`;
        container.append(mark);
      }
      el('preview-body').replaceChildren(container); image.src = result.image;
      el('preview-body').scrollTop = 0;
      el('preview-location').textContent = `Page ${page} of ${pages}`;
      el('preview-status').textContent = result.matched ? 'Exact cited passage highlighted on the original page.' :
        (data.cited_pages.includes(page) ? 'Cited page • exact text could not be located. No highlight is shown. Use Indexed text for the passage.' :
          'Original page • no located citation on this page.');
    } catch (error) {
      if (ticket !== sequence || error.name === 'AbortError') return;
      el('preview-status').textContent = error.message;
    } finally { if (ticket === sequence) controls(); }
  }
  async function open(sourceValue, trigger) {
    const ticket = cancel(); source = sourceValue; opener = trigger; data = null; mode = null;
    panel.hidden = false; document.body.classList.add('preview-open');
    el('preview-title').textContent = source.path;
    el('preview-warning').textContent = '';
    el('preview-location').textContent = '';
    el('preview-status').textContent = 'Opening cited document…';
    el('preview-body').replaceChildren(); controls(true); el('preview-close').focus({preventScroll: true});
    try {
      const result = await post('/preview', source);
      if (ticket !== sequence) return;
      data = result; el('preview-warning').textContent = data.warning;
      if (data.kind === 'pdf') await showPage(data.initial_page);
      else showText();
    } catch (error) {
      if (ticket === sequence && error.name !== 'AbortError') el('preview-status').textContent = error.message;
    }
  }
  function close() {
    cancel(); panel.hidden = true; document.body.classList.remove('preview-open');
    el('preview-body').replaceChildren(); data = null; opener?.focus({preventScroll: true});
  }
  document.addEventListener('click', event => {
    const citation = event.target.closest('.cite[data-n], .source-open');
    if (citation) {
      const refs = citation.closest('.turn')?.previewSources || [];
      const match = refs.find(item => String(item.n) === citation.dataset.n);
      if (match) open(match, citation);
    }
    const search = event.target.closest('.preview-search');
    if (search) {
      const match = el('results').previewSources?.[Number(search.dataset.index)];
      if (match) open(match, search);
    }
  });
  el('preview-close').onclick = close;
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !panel.hidden) close(); });
  el('preview-text').onclick = () => showText();
  el('preview-original').onclick = () => showPage(data.initial_page);
  el('preview-jump').onclick = () => mode === 'pdf' ? showPage(data.initial_page) : showText();
  for (const [id, direction] of [['preview-prev', -1], ['preview-next', 1]]) {
    el(id).onclick = () => {
      if (mode === 'pdf') showPage(page + direction);
      else { textOffset = Math.max(0, textOffset + direction * textPageSize); showText(false); }
    };
  }
  const resize = el('preview-resize');
  function setWidth(width) {
    const actual = Math.max(320, Math.min(innerWidth * .7, width));
    document.documentElement.style.setProperty('--preview-width', `${actual}px`);
    resize.setAttribute('aria-valuenow', String(Math.round(actual)));
  }
  resize.addEventListener('pointerdown', event => { resize.setPointerCapture(event.pointerId); event.preventDefault(); });
  resize.addEventListener('pointermove', event => { if (resize.hasPointerCapture(event.pointerId)) setWidth(innerWidth - event.clientX); });
  resize.addEventListener('pointerup', event => resize.releasePointerCapture(event.pointerId));
  resize.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
      event.preventDefault(); setWidth(panel.offsetWidth + (event.key === 'ArrowLeft' ? 30 : -30));
    }
  });
})();

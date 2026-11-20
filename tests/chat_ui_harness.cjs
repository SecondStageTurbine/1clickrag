// Execute the shipped UI handlers with a minimal DOM and real HTTP /chat API.
// This is a DOM harness, not a browser rendering or visual-layout test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const base = process.argv[2];

class Element {
  constructor(tag = '') {
    this.tag = tag; this.children = []; this.dataset = {}; this.listeners = {};
    this.value = ''; this.checked = false; this.textContent = ''; this._html = '';
    this.queries = new Map();
  }
  set innerHTML(value) { this._html = value; this.children = []; this.queries.clear(); }
  get innerHTML() { return this._html; }
  get options() { return this.children; }
  get selectedOptions() { return this.children.filter(c => c.value === this.value); }
  appendChild(child) { this.children.push(child); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute() {}
  focus() {}
  scrollIntoView() {}
  querySelector(selector) {
    if (!this.queries.has(selector)) this.queries.set(selector, new Element(selector));
    return this.queries.get(selector);
  }
}
const nodes = new Map();
const get = id => {
  if (!nodes.has(id)) nodes.set(id, new Element(id));
  return nodes.get(id);
};
const storage = new Map();
const requests = [];
let truncated = false;
const context = vm.createContext({
  document: {getElementById: get, createElement: tag => new Element(tag),
    querySelector: get, querySelectorAll: () => []},
  localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
  TextDecoder, setInterval: () => 0, clearInterval: () => {}, console,
  fetch: async (url, options) => {
    if (url === '/chat/config') return Response.json({enabled: true});
    if (url === '/chat/models') return Response.json({default: 'test-local', backends: [
      {id: 'test-local', label: 'Test', model: 'scripted-test-model', endpoint: base, local: true}]});
    if (url === '/chat') {
      requests.push(JSON.parse(options.body));
      if (truncated) return new Response('event: delta\ndata: {"text":"Partial"}\n\n');
    }
    return fetch(base + url, options);
  },
});
const html = fs.readFileSync(path.join(__dirname, '../app/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/refreshStatus\(\);\s*loadChatConfig\(\);\s*setInterval\(refreshStatus, 5000\);/, '');
vm.runInContext(script, context);

(async () => {
  await vm.runInContext('loadChatConfig()', context);
  assert.equal(get('chat-mode').value, 'general');
  assert.equal(get('chat-mode').options.length, 6);
  get('topk').value = '8'; get('hops').value = '0';
  get('hybrid').checked = true; get('retrieve').checked = true;
  get('prefix').value = 'self-test.xlsx';
  const choose = value => {
    get('chat-mode').value = value;
    get('chat-mode').listeners.change();
  };
  choose('diagnostic_review');
  get('msg').value = 'Review Low pressure and ARINC Label 213';
  await vm.runInContext('send()', context);
  assert.equal(requests[0].mode, 'diagnostic_review');
  assert.equal(requests[0].review, null);
  const snapshot = vm.runInContext('reviewSnapshot', context);
  assert.ok(snapshot.analysis.includes('[1]'));
  assert.ok(snapshot.evidence.includes('ARINC Label: 213'));
  choose('diagnostic_peer_review');
  get('msg').value = 'Check the grounding of that answer';
  await vm.runInContext('send()', context);
  assert.equal(requests[1].mode, 'diagnostic_peer_review');
  assert.equal(requests[1].review.analysis, snapshot.analysis);
  assert.equal(requests[1].review.evidence, snapshot.evidence);
  assert.equal(vm.runInContext('reviewSnapshot', context), snapshot);
  const turns = get('transcript').children;
  const panel = turns.at(-1).querySelector('.body').children[0];
  assert.ok(panel.querySelector('pre').textContent.includes('[P1]'));
  assert.ok(!panel.querySelector('pre').textContent.includes('[1]'));

  choose('security_peer_review');
  get('msg').value = 'Check source-marking fidelity';
  await vm.runInContext('send()', context);
  assert.equal(requests[2].review.analysis, snapshot.analysis); // Not review-of-review.
  assert.equal(storage.get('rag-chat-mode'), 'security_peer_review');

  choose('general');
  truncated = true;
  get('msg').value = 'Test interrupted stream';
  await vm.runInContext('send()', context);
  assert.equal(vm.runInContext('reviewSnapshot', context), snapshot);
  assert.ok(get('transcript').children.at(-1).querySelector('.body').innerHTML.includes('before completion'));
  assert.equal(requests.at(-1).review, null);

  get('reset').listeners.click();
  assert.equal(vm.runInContext('reviewSnapshot', context), null);
  assert.equal(vm.runInContext('history.length', context), 0);
  choose('diagnostic_peer_review');
  get('msg').value = 'No analysis exists';
  const count = requests.length;
  await vm.runInContext('send()', context);
  assert.equal(requests.length, count);
  assert.ok(get('mode-hint').textContent.includes('First complete an analysis'));
  console.log('UI handoff checks passed (real /chat HTTP; scripted model; minimal DOM).');
})().catch(error => { console.error(error); process.exitCode = 1; });

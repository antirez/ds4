// Tests for the chat UI Markdown/LaTeX renderer.
//
//   node tools/test_chat_ui.js                 test chat-ui.html
//   node tools/test_chat_ui.js served.html     test a page fetched from ds4-server
//
// Running it against a page fetched from the running server is the useful case:
// it proves the C string-literal escaping in ds4_server.c did not corrupt the
// regexes, which a compile cannot catch.

const fs = require('fs');
const path = require('path');

const file = process.argv[2] || path.join(__dirname, '..', 'chat-ui.html');
const html = fs.readFileSync(file, 'utf8');

const m = html.match(/<script>\n([\s\S]*?)<\/script>/);
if (!m) { console.error('FAIL: no inline <script> found in ' + file); process.exit(1); }
let src = m[1];

// The served copy has the test hook stripped; re-inject one inside the IIFE.
if (!src.includes('__ds4')) {
  const i = src.lastIndexOf('})();');
  if (i < 0) { console.error('FAIL: no IIFE tail to inject into'); process.exit(1); }
  src = src.slice(0, i) + '  window.__ds4 = { renderMarkdown: renderMarkdown };\n' + src.slice(i);
}

// Minimal DOM: the renderer only needs createElement for escaping helpers.
function el(){
  return {
    value: '', className: '', textContent: '', innerHTML: '', disabled: false,
    style: { cssText: '', height: '' }, scrollTop: 0, scrollHeight: 0,
    classList: { contains: () => false, add(){}, remove(){} },
    addEventListener(){}, appendChild(){}, replaceWith(){}, remove(){},
    querySelectorAll: () => [], querySelector: () => null,
  };
}
const document = {
  getElementById: () => el(), createElement: () => el(),
  querySelector: () => null, querySelectorAll: () => [],
};
const windowObj = { katex: undefined };   // exercise the offline fallback path

new Function('document', 'localStorage', 'window', 'fetch', 'AbortController', 'TextDecoder', src)(
  document,
  { getItem: () => null, setItem(){} },
  windowObj,
  async () => { throw new Error('offline in test'); },
  AbortController,
  TextDecoder
);

const render = windowObj.__ds4.renderMarkdown;

let pass = 0, fail = 0;
function check(name, input, mustContain, mustNotContain){
  const out = render(input);
  const okIn = (mustContain || []).every(s => out.includes(s));
  const okOut = (mustNotContain || []).every(s => !out.includes(s));
  if (okIn && okOut) { pass++; console.log('  ok   ' + name); return; }
  fail++;
  console.log('  FAIL ' + name);
  console.log('       got: ' + JSON.stringify(out.slice(0, 300)));
  if (!okIn) console.log('       missing: ' + JSON.stringify((mustContain || []).filter(s => !out.includes(s))));
  if (!okOut) console.log('       unwanted: ' + JSON.stringify((mustNotContain || []).filter(s => out.includes(s))));
}

console.log('markdown:');
check('heading', '## Title here', ['<h2>Title here</h2>']);
check('bold', 'this is **bold** text', ['<strong>bold</strong>']);
check('italic', 'this is *em* text', ['<em>em</em>']);
check('inline code', 'call `foo()` now', ['<code>foo()</code>']);
check('fenced code', '```python\nx = 1 < 2\n```', ['<pre><code>', 'x = 1 &lt; 2']);
check('ul list', '- one\n- two', ['<ul>', '<li>one</li>', '<li>two</li>']);
check('ol list', '1. first\n2. second', ['<ol>', '<li>first</li>']);
check('link', '[site](https://x.com)', ['<a href="https://x.com"', '>site</a>']);
check('blockquote', '> quoted', ['<blockquote>', 'quoted']);
check('hr', 'a\n\n---\n\nb', ['<hr>']);
check('table', '| a | b |\n|---|---|\n| 1 | 2 |', ['<table>', '<th>a</th>', '<td>1</td>']);
check('paragraph', 'hello world', ['<p>hello world</p>']);

console.log('escaping:');
check('script tag escaped', 'hi <script>alert(1)</scr' + 'ipt>', ['&lt;script&gt;'], ['<script>alert']);
check('img onerror escaped', '<img src=x onerror=alert(1)>', ['&lt;img'], ['<img src=x']);
check('code keeps lt', '`a < b`', ['a &lt; b']);

console.log('math:');
const maxwell = '$$\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}$$';
check('display math', maxwell, ['math-display', '\\nabla', '\\frac{\\partial'], ['$$']);
check('inline math', 'let $x^2$ be', ['math-raw', 'x^2'], ['$x^2$']);
check('bracket display', '\\[a+b\\]', ['math-display', 'a+b']);
check('paren inline', 'val \\(y_1\\) here', ['math-raw', 'y_1']);
check('math not italicised', '$a * b * c$', ['a * b * c'], ['<em>']);
check('currency not math', 'costs $5 and $7 total', ['costs $5'], ['math-raw']);

console.log('mixed:');
check('math inside list', '- item $x$\n- other', ['<li>', 'math-raw']);
check('code not mathed', '```\n$$not math$$\n```', ['$$not math$$'], ['math-display']);

console.log('\n' + path.basename(file) + ': pass=' + pass + ' fail=' + fail);
process.exit(fail ? 1 : 0);

const test = require('node:test');
const assert = require('node:assert/strict');
const remark = require('remark');
const plugin = require('./index');
const { renderPaper } = plugin;

const paper = `title: The Illusion of Thinking
institution: Apple
url: https://arxiv.org/abs/2506.06941
logo: /paper-logos/apple.ico
authors: Parshin Shojaee et al.
year: 2025
venue: NeurIPS
summary: 문제 복잡도에 따른 추론 모델의 강점과 한계.`;

test('a paper fence becomes one full-card link, leaving other code and prose intact', () => {
  const tree = remark().parse(
    `앞 문장\n\n\`\`\`paper\n${paper}\n\`\`\`\n\n\`\`\`yaml\n${paper}\n\`\`\`\n\n뒤 문장`,
  );
  plugin({ markdownAST: tree });
  assert.equal(tree.children[0].children[0].value, '앞 문장');
  assert.equal(tree.children[1].type, 'html');
  assert.equal(tree.children[2].type, 'code');
  assert.equal(tree.children[3].children[0].value, '뒤 문장');
  const html = tree.children[1].value;
  assert.equal((html.match(/<a\b/g) || []).length, 1);
  assert.match(html, /href="https:\/\/arxiv.org\/abs\/2506.06941"/);
  assert.match(html, /target="_blank" rel="noopener noreferrer"/);
  assert.match(html, /NeurIPS · 2025/);
  assert.match(html, /새 탭에서 열림/);
  assert.match(html, /src="\/paper-logos\/apple.ico" alt=""/);
});

test('minimal cards do not require a logo, description, or publication metadata', () => {
  const html = renderPaper('title: 논문\ninstitution: 연구소\nurl: https://example.org/paper');
  assert.match(html, /paper-card__placeholder/);
  assert.doesNotMatch(html, /<img|undefined|null|NaN/);
});

test('YAML folded summaries and quoted values are supported', () => {
  const html = renderPaper(
    `${paper}\nsummary: >-\n  첫 번째 문장.\n  두 번째 문장.`.replace(
      'summary: 문제 복잡도에 따른 추론 모델의 강점과 한계.\n',
      '',
    ),
  );
  assert.match(html, /첫 번째 문장. 두 번째 문장./);
});

test('text and image attributes cannot inject HTML', () => {
  const html = renderPaper(
    "title: '<script>alert(\"x\")</script>'\ninstitution: 'A & B'\nurl: 'https://example.org/?a=1&b=\"x\"'\nlogo: '/paper-logos/x\"onerror=\"alert(1).svg'",
  );
  assert.doesNotMatch(html, /<script|src="[^"]*"onerror=/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /A &amp; B/);
  assert.match(html, /&quot;onerror=&quot;/);
});

test('unsafe destinations and logos are rejected before they reach the page', () => {
  for (const url of [
    'javascript:alert(1)',
    'data:text/html,hello',
    '//example.org',
    '/paper',
    'https://user:secret@example.org',
  ]) {
    assert.throws(() => renderPaper(paper.replace('https://arxiv.org/abs/2506.06941', url)), /url/);
  }
  for (const logo of [
    'javascript:alert(1)',
    'data:image/svg+xml,hello',
    '//example.org/logo.svg',
    '/\\example.org/logo.svg',
    './logo.svg',
  ]) {
    assert.throws(() => renderPaper(paper.replace('/paper-logos/apple.ico', logo)), /logo/);
  }
});

test('missing fields and malformed YAML report the source file and line', () => {
  for (const source of [
    'title: [',
    'null',
    '[]',
    'title: 논문',
    paper.replace('year: 2025', 'year: [2025]'),
  ]) {
    const markdownAST = remark().parse(`\n\n\`\`\`paper\n${source}\n\`\`\``);
    assert.throws(
      () => plugin({ markdownAST, markdownNode: { fileAbsolutePath: '/blog/post.md' } }),
      /\[paper-card\] \/blog\/post.md:3/,
    );
  }
});

test('an unsafe YAML tag is never executed', () => {
  assert.throws(
    () => renderPaper(`${paper}\nsummary: !!js/function 'function () { return 1; }'`),
    /unknown tag/,
  );
});

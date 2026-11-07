const assert = require('node:assert/strict');
const { test } = require('node:test');
const remark = require('remark');
const colorNames = require('color-name');
const parser = require('./index').setParserPlugins()[0];

const parse = (source) => remark().use(parser).parse(source);
const find = (node, type) => [
  ...(node.type === type ? [node] : []),
  ...(node.children || []).flatMap((child) => find(child, type)),
];
const text = (node) => node.value || (node.children || []).map(text).join('');
const highlights = (source) => find(parse(source), 'highlight');
const color = (node) => node.data.hProperties['data-highlight-color'];

test('default, explicit and multiple highlights preserve surrounding prose', () => {
  const tree = parse('앞 ==문장== 중간 ==yellow:노랑== ==green:초록== 뒤');
  const nodes = find(tree, 'highlight');
  assert.deepEqual(nodes.map(color), ['yellow', 'yellow', 'green']);
  assert.deepEqual(nodes.map(text), ['문장', '노랑', '초록']);
  assert.equal(text(tree), '앞 문장 중간 노랑 초록 뒤');
});

test('all CSS color names are accepted, without case sensitivity', () => {
  for (const name of Object.keys(colorNames)) {
    const [node] = highlights(`==${name.toUpperCase()}:색상==`);
    assert.equal(color(node), name);
    assert.equal(text(node), '색상');
    assert.match(
      node.data.hProperties.style,
      /^--highlight-light: #[0-9a-f]{6}; --highlight-dark: #[0-9a-f]{6};$/,
    );
  }
});

test('three and six digit HEX colors work and only validated colors reach CSS', () => {
  for (const name of ['#9CF', '#FFCC80', '#000', '#ffffff']) {
    const [node] = highlights(`==${name}:색상==`);
    assert.equal(color(node), name.toLowerCase());
    assert.equal(text(node), '색상');
  }
  for (const prefix of ['invalid', '#12345', 'red;position:fixed', 'constructor']) {
    const [node] = highlights(`==${prefix}:문장==`);
    assert.equal(color(node), 'yellow');
    assert.equal(text(node), `${prefix}:문장`);
    assert.equal(
      node.data.hProperties.style,
      '--highlight-light: #ffec9e; --highlight-dark: #67531d;',
    );
  }
});

test('normal Markdown works inside and around highlights', () => {
  const tree = parse(
    '### ==green:**신뢰**==\n\n**==blue:강조==** 그리고 ==pink:*분홍*과 [링크](/posts)==',
  );
  const nodes = find(tree, 'highlight');
  assert.deepEqual(nodes.map(text), ['신뢰', '강조', '분홍과 링크']);
  assert.equal(nodes[0].children[0].type, 'strong');
  assert.equal(find(tree, 'strong')[1].children[0].type, 'highlight');
  assert.equal(find(nodes[2], 'link')[0].url, '/posts');
});

test('literal examples, code, HTML attributes and link destinations stay unchanged', () => {
  const cases = [
    '\\==문장==',
    '`==green:코드==`',
    '```js\nconst example = "==blue:code==";\n```',
    '<span title="==green:attribute==">HTML</span>',
    '[링크](/example?value===green:literal==)',
    'x === y',
    '===문장===',
    '== 문장 ==',
    '==첫 줄\n둘째 줄==',
    '==닫지 않은 문장',
    '==green:==',
  ];
  for (const source of cases) assert.equal(highlights(source).length, 0, source);
  const [node] = highlights('==코드 `a==b` 포함==');
  assert.equal(text(node), '코드 a==b 포함');
  assert.equal(find(node, 'inlineCode')[0].value, 'a==b');
});

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const linear = channels.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return linear.reduce((total, c, index) => total + c * [0.2126, 0.7152, 0.0722][index], 0);
}

test('named colors and extreme HEX colors keep readable contrast in both themes', () => {
  for (const name of [...Object.keys(colorNames), '#000000', '#ffffff', '#ff00ff', '#00ffff']) {
    const [node] = highlights(`==${name}:문장==`);
    const [light, dark] = node.data.hProperties.style.match(/#[0-9a-f]{6}/g);
    for (const [foreground, background] of [
      ['#37352f', light],
      ['#e6e6e6', dark],
    ]) {
      const a = luminance(foreground),
        b = luminance(background);
      assert.ok((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) >= 4.5, name);
    }
  }
});

const underlines = (source) => find(parse(source), 'underline');

test('plain and designed underlines preserve text and support every line style', () => {
  const [plain] = underlines('앞 ++밑줄++ 뒤');
  assert.equal(text(plain), '밑줄');
  assert.equal(plain.data.hName, 'span');
  assert.equal(plain.data.hProperties['data-underline-color'], 'currentColor');
  assert.match(
    plain.data.hProperties.style,
    /--underline-style: solid; --underline-thickness: 1.5px;/,
  );
  for (const style of ['solid', 'wavy', 'dashed', 'dotted', 'double']) {
    const [node] = underlines(`++${style}:문장++`);
    assert.equal(text(node), '문장');
    assert.equal(node.data.hProperties['data-underline-style'], style);
  }
});

test('underline color, line style, and weight can be combined in any order', () => {
  for (const options of ['blue,wavy,thick', 'THICK, BLUE, WAVY', 'wavy,thick,blue']) {
    const [node] = underlines(`++${options}:강조++`);
    assert.equal(text(node), '강조');
    assert.equal(node.data.hProperties['data-underline-color'], 'blue');
    assert.match(
      node.data.hProperties.style,
      /--underline-style: wavy; --underline-thickness: 3px;/,
    );
  }
  const [node] = underlines('++#9CF,double,thin:하늘색++');
  assert.equal(node.data.hProperties['data-underline-color'], '#9cf');
  assert.match(node.data.hProperties.style, /--underline-thickness: 1px;/);
});

test('underline markup works with bold text, highlights, headings and wrapped text', () => {
  const tree = parse('### ++wavy:**굵게**++\n\n==green:++밑줄++== 그리고 ++blue:==형광펜==++');
  assert.deepEqual(find(tree, 'underline').map(text), ['굵게', '밑줄', '형광펜']);
  assert.equal(find(tree, 'underline')[0].children[0].type, 'strong');
  assert.equal(find(tree, 'highlight')[0].children[0].type, 'underline');
  assert.equal(find(tree, 'underline')[2].children[0].type, 'highlight');
});

test('underline delimiters stay literal in code, escapes, attributes and URLs', () => {
  for (const source of [
    '\\++문장++',
    '`++wavy:코드++`',
    '```text\n++밑줄++\n```',
    '<span title="++속성++">HTML</span>',
    '[링크](/example?value=++literal++)',
    '++ 문장 ++',
    '+++문장+++',
    '++첫 줄\n둘째 줄++',
    '++닫지 않은 문장',
    '++wavy:++',
    'x++ + y++',
  ])
    assert.equal(underlines(source).length, 0, source);
  const [node] = underlines('++코드 `a++b` 포함++');
  assert.equal(find(node, 'inlineCode')[0].value, 'a++b');
});

test('invalid or repeated design options remain text and cannot inject CSS', () => {
  for (const prefix of [
    'red;position:fixed',
    'wavy,dashed',
    'red,blue',
    'thin,thick',
    'constructor',
    'unknown',
  ]) {
    const [node] = underlines(`++${prefix}:문장++`);
    assert.equal(text(node), `${prefix}:문장`);
    assert.equal(node.data.hProperties['data-underline-style'], 'solid');
    assert.equal(
      node.data.hProperties.style,
      '--underline-style: solid; --underline-thickness: 1.5px;',
    );
  }
});

test('underline colors remain visible on both light and dark backgrounds', () => {
  for (const name of [...Object.keys(colorNames), '#fff', '#000', '#9cf']) {
    const [node] = underlines(`++${name}:문장++`);
    const [light, dark] = node.data.hProperties.style.match(/#[0-9a-f]{6}/g);
    for (const [stroke, background] of [
      [light, '#ffffff'],
      [dark, '#232326'],
    ]) {
      const a = luminance(stroke),
        b = luminance(background);
      assert.ok((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) >= 3, name);
    }
  }
});

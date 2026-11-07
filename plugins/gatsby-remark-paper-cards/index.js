const yaml = require('js-yaml');

const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]),
  );

function webUrl(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password;
  } catch {
    return false;
  }
}

function renderPaper(source) {
  const paper = yaml.safeLoad(source, { schema: yaml.JSON_SCHEMA });
  if (!paper || typeof paper !== 'object' || Array.isArray(paper)) {
    throw new Error('논문 정보는 title: 값 형식으로 작성해 주세요.');
  }
  for (const field of ['title', 'institution', 'url']) {
    if (typeof paper[field] !== 'string' || !paper[field].trim()) {
      throw new Error(`${field} 항목이 필요합니다.`);
    }
  }
  if (!webUrl(paper.url)) throw new Error('url에는 http 또는 https 논문 주소를 넣어 주세요.');
  for (const field of ['logo', 'summary', 'authors', 'year', 'venue']) {
    if (paper[field] != null && !['string', 'number'].includes(typeof paper[field])) {
      throw new Error(`${field} 항목은 한 개의 문자열이나 숫자로 작성해 주세요.`);
    }
  }
  if (paper.logo && !webUrl(paper.logo) && !/^\/(?!\/)[^\s\\?#]+$/.test(paper.logo)) {
    throw new Error(
      'logo에는 /paper-logos/logo.svg 같은 절대 경로나 https 이미지 주소를 넣어 주세요.',
    );
  }

  const logo = paper.logo
    ? `<img class="paper-card__logo" src="${escape(
        paper.logo,
      )}" alt="" width="64" height="64" loading="lazy" decoding="async">`
    : '<svg class="paper-card__placeholder" viewBox="0 0 48 48" fill="none" aria-hidden="true"><path d="M14 6h13l9 9v27H14zM27 6v10h9M20 24h10M20 30h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const metadata = [paper.venue, paper.year].filter((value) => value != null && value !== '');
  const text = (className, value) =>
    value != null && value !== ''
      ? `<span class="paper-card__${className}">${escape(value)}</span>`
      : '';

  return `<a class="paper-card" href="${escape(
    paper.url,
  )}" target="_blank" rel="noopener noreferrer">
    <span class="paper-card__institution">${logo}<span class="paper-card__institution-name">${escape(
    paper.institution,
  )}</span></span>
    <span class="paper-card__body">
      <span class="paper-card__title">${escape(paper.title)}</span>
      ${text('summary', paper.summary)}
      <span class="paper-card__details">${text('authors', paper.authors)}${text(
    'metadata',
    metadata.join(' · '),
  )}</span>
      <span class="paper-card__action">논문 읽기 <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M5 15 15 5M5 5h10v10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="paper-card__sr-only"> (새 탭에서 열림)</span></span>
    </span>
  </a>`;
}

module.exports = ({ markdownAST, markdownNode }) => {
  function visit(node) {
    if (node.type === 'code' && node.lang === 'paper') {
      try {
        node.value = renderPaper(node.value);
        node.type = 'html';
        delete node.lang;
        delete node.meta;
      } catch (error) {
        const file = markdownNode?.fileAbsolutePath || 'Markdown';
        throw new Error(
          `[paper-card] ${file}:${node.position?.start.line || '?'} — ${error.message}`,
        );
      }
    }
    if (node.children) node.children.forEach(visit);
  }
  visit(markdownAST);
  return markdownAST;
};

module.exports.renderPaper = renderPaper;

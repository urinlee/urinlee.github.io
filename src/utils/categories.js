const CATEGORIES = [
  { name: '개발', slug: 'development' },
  { name: 'CTF/Wargame', slug: 'ctf-wargame' },
  { name: 'Bug Bounty', slug: 'bug-bounty' },
  { name: '블로그/기술문서', slug: 'blog-docs' },
  { name: '논문/컨퍼런스', slug: 'papers-conferences' },
  { name: '공모전/자격증', slug: 'contests-certifications' },
];

const CATEGORY_NAMES = ['All', ...CATEGORIES.map(({ name }) => name)];

const getPostCategories = (categories) => (categories ? [categories.trim()] : []);

const getCategoryPath = (category) => {
  if (category === 'All') return '/posts';
  const entry = CATEGORIES.find(({ name }) => name === category);
  if (!entry) throw new Error(`Unknown category: ${category}`);
  return `/posts/${entry.slug}`;
};

module.exports = { CATEGORIES, CATEGORY_NAMES, getPostCategories, getCategoryPath };

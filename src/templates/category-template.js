import React, { useMemo, useCallback } from 'react';
import { navigate } from 'gatsby';

import Layout from '../layout';
import Seo from '../components/seo';
import Post from '../models/post';
import CategoryPageHeader from '../components/category-page-header';
import PostTabs from '../components/post-tabs';
import { getCategoryPath } from '../utils/categories';

function CategoryTemplate({ pageContext }) {
  const { edges, currentCategory } = pageContext;
  const { categories } = pageContext;
  const currentTabIndex = useMemo(
    () => categories.findIndex((category) => category === currentCategory),
    [categories, currentCategory],
  );
  const posts = edges.map(({ node }) => new Post(node));

  const onTabIndexChange = useCallback(
    (e, value) => {
      navigate(getCategoryPath(categories[value]));
    },
    [categories],
  );

  return (
    <Layout>
      <Seo title="Posts" />
      <CategoryPageHeader title={categories[currentTabIndex]} subtitle={`${posts.length} posts`} />
      <PostTabs
        tabIndex={currentTabIndex}
        onChange={onTabIndexChange}
        tabs={categories}
        posts={posts}
      />
    </Layout>
  );
}

export default CategoryTemplate;

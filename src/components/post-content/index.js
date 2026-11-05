import React from 'react';
import './style.scss';

function PostContent({ html, readingStyle }) {
  return (
    <div
      className={
        readingStyle === 'reflection' ? 'post-content post-content--reflection' : 'post-content'
      }
    >
      <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

export default PostContent;

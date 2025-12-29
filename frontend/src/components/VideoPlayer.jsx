import React from 'react';

function VideoPlayer({ src }) {
  return (
    <video className="video" src={src} controls />
  );
}

export default VideoPlayer;

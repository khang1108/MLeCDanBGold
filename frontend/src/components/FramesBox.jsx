import React from 'react';
import FrameCard from './FrameCard';

// 20 Mock Frames with diverse, high-quality images and descriptive captions
const MOCK_FRAMES = [
  { id: 1, imageUrl: 'https://picsum.photos/seed/frame1/1024/768', caption: 'A sleek red sports car racing down a scenic mountain highway during sunset.' },
  { id: 2, imageUrl: 'https://picsum.photos/seed/frame2/1024/768', caption: 'Close-up of hands typing lines of clean code on a mechanical keyboard.' },
  { id: 3, imageUrl: 'https://picsum.photos/seed/frame3/1024/768', caption: 'A golden retriever puppy jumping to catch a bright yellow tennis ball.' },
  { id: 4, imageUrl: 'https://picsum.photos/seed/frame4/1024/768', caption: 'Aerial view of misty pine forests stretching across rolling green hills.' },
  { id: 5, imageUrl: 'https://picsum.photos/seed/frame5/1024/768', caption: 'Two scientists in white lab coats analyzing biological samples under a microscope.' },
  { id: 6, imageUrl: 'https://picsum.photos/seed/frame6/1024/768', caption: 'A barista pouring steamed milk to make latte art in a white ceramic cup.' },
  { id: 7, imageUrl: 'https://picsum.photos/seed/frame7/1024/768', caption: 'Group of young professionals brainstorming in a modern glass conference room.' },
  { id: 8, imageUrl: 'https://picsum.photos/seed/frame8/1024/768', caption: 'A close-up shot of a smartphone screen displaying a colorful dashboard app.' },
  { id: 9, imageUrl: 'https://picsum.photos/seed/frame9/1024/768', caption: 'A dense flock of white seagulls soaring high over crashing ocean waves.' },
  { id: 10, imageUrl: 'https://picsum.photos/seed/frame10/1024/768', caption: 'A modern skyscraper reflecting a bright blue sky on its glass windows.' },
  { id: 11, imageUrl: 'https://picsum.photos/seed/frame11/1024/768', caption: 'Chef garnishing a gourmet plate of pasta with fresh green basil leaves.' },
  { id: 12, imageUrl: 'https://picsum.photos/seed/frame12/1024/768', caption: 'A cozy coffee shop corner with a book and a steaming hot mug on a wooden table.' },
  { id: 13, imageUrl: 'https://picsum.photos/seed/frame13/1024/768', caption: 'Fast-moving city traffic at night creating beautiful neon light trails.' },
  { id: 14, imageUrl: 'https://picsum.photos/seed/frame14/1024/768', caption: 'A hiker standing on a high peak looking out over a massive valley.' },
  { id: 15, imageUrl: 'https://picsum.photos/seed/frame15/1024/768', caption: 'A person wearing VR goggles interacting with virtual graphs in the air.' },
  { id: 16, imageUrl: 'https://picsum.photos/seed/frame16/1024/768', caption: 'Close-up of water droplets bead-forming on a fresh green leaf.' },
  { id: 17, imageUrl: 'https://picsum.photos/seed/frame17/1024/768', caption: 'A couple walking hand in hand down a quiet rain-slicked city street.' },
  { id: 18, imageUrl: 'https://picsum.photos/seed/frame18/1024/768', caption: 'A vintage record player spinning a black vinyl record in warm light.' },
  { id: 19, imageUrl: 'https://picsum.photos/seed/frame19/1024/768', caption: 'Stunning view of the Milky Way galaxy glowing bright over a dark desert.' },
  { id: 20, imageUrl: 'https://picsum.photos/seed/frame20/1024/768', caption: 'A cute black cat sleeping curled up on a soft gray knitted blanket.' }
];

const FramesBox = ({ searchQuery, onFrameClick }) => {
  // Filter frames based on the search query
  const filteredFrames = MOCK_FRAMES.filter((frame) =>
    frame.caption.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <section className="frames-container">
      {filteredFrames.length > 0 ? (
        <div className="frames-grid">
          {filteredFrames.map((frame) => (
            <FrameCard
              key={frame.id}
              index={frame.id}
              imageUrl={frame.imageUrl}
              caption={frame.caption}
              onClick={() => onFrameClick(frame)}
            />
          ))}
        </div>
      ) : (
        /* Empty State */
        <div className="frames-empty-state">
          <svg
            className="frames-empty-icon"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.2}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m15.75 15.75-2.489-2.489m0 0a3.375 3.375 0 1 0-4.773-4.773 3.375 3.375 0 0 0 4.774 4.774ZM21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
            />
          </svg>
          <p className="body-md frames-empty-text">
            No frames found matching your query
          </p>
          <p className="caption frames-empty-subtext">
            Try adjusting your search terms or lowering the similarity threshold.
          </p>
        </div>
      )}
    </section>
  );
};

export default FramesBox;

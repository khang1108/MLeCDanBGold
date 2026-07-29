import React, { useEffect, useState } from "react";

// Auto-scan all animated GIFs or images placed inside src/assets/gifs/
let gifsList = [];
try {
  const gifsContext = require.context(
    "../../../assets/gifs",
    false,
    /\.(gif|png|jpe?g|webp)$/i,
  );
  gifsList = gifsContext.keys().map(gifsContext);
} catch (err) {
  gifsList = [];
}

const GifLoaderOverlay = ({ isVisible }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!isVisible) {
      setCurrentIndex(0);
      return;
    }

    // Auto-loop through discovered GIFs every 3.5s
    let gifInterval = null;
    if (gifsList.length > 1) {
      gifInterval = setInterval(() => {
        setCurrentIndex((prev) => (prev + 1) % gifsList.length);
      }, 3500);
    }

    return () => {
      if (gifInterval) clearInterval(gifInterval);
    };
  }, [isVisible]);

  if (!isVisible) return null;

  const currentGifSrc = gifsList.length > 0 ? gifsList[currentIndex] : null;

  return (
    <div className="gif-loader-overlay">
      {currentGifSrc ? (
        <img
          src={currentGifSrc}
          alt="Search Loading GIF"
          className="gif-loader-img"
        />
      ) : (
        <div className="gif-loader-placeholder">
          <span className="gif-spinner-ring"></span>
        </div>
      )}
    </div>
  );
};

export default GifLoaderOverlay;

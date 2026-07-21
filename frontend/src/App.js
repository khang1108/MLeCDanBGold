import React, { useState } from 'react';
import QueryInput from './components/QueryInput';
import ToolBox from './components/ToolBox';
import FramesBox from './components/FramesBox';
import ImageModal from './components/ImageModal';

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

function App() {
  const [query, setQuery] = useState('');
  const [selectedFrame, setSelectedFrame] = useState(null);

  // Parameter states
  const [topK, setTopK] = useState(20);
  const [temperature, setTemperature] = useState(0.2);
  const [filter, setFilter] = useState('');
  const [searchMode, setSearchMode] = useState('accurate');

  // Search Results, loading, and error states
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);

  const handleSearch = (q) => {
    if (!q || q.trim() === '') {
      setResults([]);
      setLatencyMs(null);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    setLatencyMs(null);

    // Simulate search latency delay (600ms)
    setTimeout(() => {
      const normalizedQuery = q.toLowerCase().trim();

      // Trigger error simulation if user searches "error"
      if (normalizedQuery === 'error') {
        setError('Failed to contact search server. Network connection refused (localhost:8000).');
        setResults([]);
        setIsLoading(false);
        return;
      }

      // Filter local frames
      const matches = MOCK_FRAMES.filter(frame => 
        frame.caption.toLowerCase().includes(normalizedQuery)
      );

      // Map to SearchResult models
      const searchResults = matches.map((frame, index) => {
        const rank = index + 1;
        const scoreBase = 0.95 - (index * 0.03); // decreasing scores
        
        // Generate simulated scores
        const visualScore = Math.max(0.1, +(scoreBase - 0.05).toFixed(2));
        const captionScore = Math.max(0.1, +(scoreBase - 0.1).toFixed(2));
        const ocrScore = index % 3 === 0 ? +(scoreBase - 0.02).toFixed(2) : null;
        const asrScore = index % 5 === 0 ? +(scoreBase - 0.08).toFixed(2) : null;
        const fusionScore = +(scoreBase - 0.01).toFixed(2);
        const rerankerScore = searchMode === 'accurate' ? +scoreBase.toFixed(2) : null;
        const finalScore = searchMode === 'accurate' 
          ? rerankerScore 
          : fusionScore;

        return {
          rank,
          frame_id: `L21_V0001_${String(frame.id).padStart(8, '0')}`,
          video_id: 'L21_V0001',
          frame_idx: frame.id,
          timestamp_ms: frame.id * 45000 + 1800,
          thumbnail_url: frame.imageUrl,
          frame_url: frame.imageUrl,
          caption: frame.caption,
          ocr_text: index % 3 === 0 ? `Detected text frame ${frame.id}` : null,
          asr_text: index % 5 === 0 ? `Audio transcript segment ${frame.id}` : null,
          scores: {
            visual: visualScore,
            caption: captionScore,
            ocr: ocrScore,
            asr: asrScore,
            fusion: fusionScore,
            reranker: rerankerScore,
            final: finalScore
          }
        };
      });

      // Slice to topK
      const finalResults = searchResults.slice(0, topK);

      // Generate simulated latency breakdown in ms
      const qProc = Math.floor(Math.random() * 15) + 10;
      const qEnc = Math.floor(Math.random() * 20) + 30;
      const cRet = Math.floor(Math.random() * 30) + 40;
      const fuse = Math.floor(Math.random() * 5) + 3;
      const rerank = searchMode === 'accurate' ? Math.floor(Math.random() * 150) + 400 : 0;
      const tempRef = Math.floor(Math.random() * 15) + 20;
      const mat = Math.floor(Math.random() * 10) + 10;
      const total = qProc + qEnc + cRet + fuse + rerank + tempRef + mat;

      setResults(finalResults);
      setLatencyMs({
        query_processing: qProc,
        query_encoding: qEnc,
        candidate_retrieval: cRet,
        fusion: fuse,
        reranking: rerank,
        temporal_refinement: tempRef,
        materialization: mat,
        total: total
      });
      setIsLoading(false);
    }, 600);
  };

  const handleReset = () => {
    setQuery('');
    setResults([]);
    setIsLoading(false);
    setError(null);
    setLatencyMs(null);
    setTopK(20);
    setTemperature(0.2);
    setFilter('');
    setSearchMode('accurate');
  };

  return (
    <div className="app-wrapper">
      {/* Main Page Layout Container */}
      <main className="app-container">
        {/* Top Search Input Box */}
        <section className="search-section">
          <QueryInput
            query={query}
            setQuery={setQuery}
            onSearch={handleSearch}
          />
        </section>

        {/* Two-Column Workspace Layout */}
        <div className="workspace-layout">
          {/* Left Panel: ToolBox Parameters */}
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            temperature={temperature}
            setTemperature={setTemperature}
            filter={filter}
            setFilter={setFilter}
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            onReset={handleReset}
          />

          {/* Right Panel: FramesBox */}
          <FramesBox
            results={results}
            isLoading={isLoading}
            error={error}
            latencyMs={latencyMs}
            onFrameClick={setSelectedFrame}
          />
        </div>
      </main>

      {/* Large Image Lightbox Popup Modal */}
      {selectedFrame && (
        <ImageModal
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}
    </div>
  );
}

export default App;

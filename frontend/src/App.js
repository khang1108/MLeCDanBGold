import React, { useState } from 'react';
import QueryInput from './components/QueryInput';
import ToolBox from './components/ToolBox';
import FramesBox from './components/FramesBox';
import ImageModal from './components/ImageModal';
import { searchFrames } from './api/search';

function App() {
  const [query, setQuery] = useState('');
  const [selectedFrame, setSelectedFrame] = useState(null);

  const [topK, setTopK] = useState(20);
  const [searchMode, setSearchMode] = useState('accurate');

  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);

  const handleSearch = async (q) => {
    if (!q || q.trim() === '') {
      setResults([]);
      setLatencyMs(null);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    setLatencyMs(null);

    try {
      const response = await searchFrames({
        query: q,
        topK,
        searchMode,
      });
      setResults(response.results);
      setLatencyMs(response.latency_ms);
    } catch (requestError) {
      setResults([]);
      setError(requestError.message || 'Could not contact the search API');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setQuery('');
    setResults([]);
    setIsLoading(false);
    setError(null);
    setLatencyMs(null);
    setTopK(20);
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

import React, { useState } from 'react';
import QueryInput from './components/QueryInput';
import ToolBox from './components/ToolBox';
import FramesBox from './components/FramesBox';
import ImageModal from './components/ImageModal';

function App() {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFrame, setSelectedFrame] = useState(null);

  // Parameter states
  const [topK, setTopK] = useState(20);
  const [temperature, setTemperature] = useState(0.2);
  const [filter, setFilter] = useState('');

  const handleSearch = (q) => {
    setSearchQuery(q);
  };

  const handleReset = () => {
    setQuery('');
    setSearchQuery('');
    setTopK(20);
    setTemperature(0.2);
    setFilter('');
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
            onReset={handleReset}
          />

          {/* Right Panel: FramesBox */}
          <FramesBox
            searchQuery={searchQuery}
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

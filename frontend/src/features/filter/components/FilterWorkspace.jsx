import React, { useCallback, useEffect, useRef, useState } from 'react';
import { filterFrames } from '../../../api/filter';
import FrameCard from '../../frames/components/FrameCard';
import SubmissionWorktree from '../../submission/components/SubmissionWorktree';
import { useSubmissionDialog } from '../../submission/contexts/SubmissionDialogContext';
import { displayVideoId } from '../../frames/videoSource';
import FilterForm from './FilterForm';
import FilterPagination from './FilterPagination';
import { DEFAULT_FRAMES_PER_PAGE, resolveFramesPerPage } from '../filterPagination';


const MatchedFrame = ({ frame, onFrameClick, onSubmit }) => (
  <div className="filter-result-card">
    <FrameCard
      frame={frame}
      imageLoading="eager"
      onClick={() => onFrameClick?.(frame)}
      onSubmit={onSubmit}
    />
    <div className="filter-match-list">
      {Object.entries(frame.matches || {}).map(([source, text]) => (
        <p className="filter-match-text" key={source} title={text}>
          <strong>{source}</strong>: {text}
        </p>
      ))}
    </div>
  </div>
);


const FilterResults = ({
  results,
  totalResults,
  availableSources,
  hasFiltered,
  error,
  onFrameClick,
  onSubmit,
  containerRef,
  currentPage,
  totalPages,
  isLoading,
  onPageChange,
}) => {
  if (error) {
    return (
      <section ref={containerRef} className="frames-container filter-results" aria-label="Filter results">
        <div className="error-alert" role="alert">{error}</div>
      </section>
    );
  }

  if (!hasFiltered || !results.length) {
    return (
      <section ref={containerRef} className="frames-container filter-results" aria-label="Filter results">
        <div className="frames-empty-state filter-empty-state">
          <p className="body-md frames-empty-text">
            {hasFiltered ? 'No matching frames' : 'Search text evidence directly'}
          </p>
          <p className="caption frames-empty-subtext">
            {hasFiltered
              ? 'Try another keyword or remove the folder/video scope.'
              : 'One keyword checks Title, Caption, OCR, ASR, and Objects.'}
          </p>
        </div>
        {hasFiltered && totalPages > 0 && (
          <FilterPagination
            currentPage={currentPage}
            totalPages={totalPages}
            isLoading={isLoading}
            onPageChange={onPageChange}
          />
        )}
      </section>
    );
  }

  return (
    <section ref={containerRef} className="frames-container filter-results" aria-label="Filter results">
      <div className="filter-result-toolbar">
        <div className="filter-result-summary">
          <strong>{totalResults}</strong> matches
          {availableSources.length ? ` · ${availableSources.join(', ')}` : ''}
        </div>
      </div>
      <div className="frames-scroll-region">
        <div className="frames-grid">
          {results.map((frame) => (
            <MatchedFrame
              key={frame.frame_id}
              frame={frame}
              onFrameClick={onFrameClick}
              onSubmit={onSubmit}
            />
          ))}
        </div>
      </div>
      <FilterPagination
        currentPage={currentPage}
        totalPages={totalPages}
        isLoading={isLoading}
        onPageChange={onPageChange}
      />
    </section>
  );
};


/** Search raw corpus text with optional free-text folder and video scopes. */
const FilterWorkspace = ({ isActive = true, onFrameClick }) => {
  const [query, setQuery] = useState('');
  const [folderId, setFolderId] = useState('');
  const [videoId, setVideoId] = useState('');
  const [applied, setApplied] = useState(null);
  const [results, setResults] = useState([]);
  const [availableSources, setAvailableSources] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [pageId, setPageId] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [hasFiltered, setHasFiltered] = useState(false);
  const [isFiltering, setIsFiltering] = useState(false);
  const [error, setError] = useState(null);
  const requestRef = useRef(null);
  const resultsViewportRef = useRef(null);
  const { requestSubmission } = useSubmissionDialog();

  useEffect(() => () => requestRef.current?.abort(), []);

  const pageSize = useCallback(() => resolveFramesPerPage('auto', {
    width: resultsViewportRef.current?.clientWidth,
    height: resultsViewportRef.current?.clientHeight,
  }), []);

  const requestPage = useCallback(async (parameters) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsFiltering(true);
    setHasFiltered(true);
    setError(null);

    try {
      const response = await filterFrames({ ...parameters, signal: controller.signal });
      setResults(response.results || []);
      setAvailableSources(response.available_sources || []);
      setTotalResults(response.total_results || 0);
      setPageId(response.page_id);
      setTotalPages(response.total_pages);
    } catch (requestError) {
      if (requestError.name !== 'AbortError') {
        setResults([]);
        setError(requestError.message || 'Failed to contact filter API');
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsFiltering(false);
      }
    }
  }, []);

  const handleFilter = useCallback((event) => {
    event.preventDefault();
    if (isFiltering || !query.trim()) return;
    const parameters = {
      query,
      folderId,
      videoId,
      framesPerPage: pageSize() || DEFAULT_FRAMES_PER_PAGE,
      pageId: 1,
    };
    setApplied(parameters);
    requestPage(parameters);
  }, [folderId, isFiltering, pageSize, query, requestPage, videoId]);

  const handlePageChange = useCallback((nextPage) => {
    if (!applied || isFiltering || nextPage < 1 || nextPage > totalPages) return;
    requestPage({ ...applied, pageId: nextPage });
  }, [applied, isFiltering, requestPage, totalPages]);

  const handleFrameSubmit = useCallback((frame) => {
    requestSubmission({
      line: `${displayVideoId(frame.video_id)},${frame.frame_idx}`,
      source: 'KIS frame',
    });
  }, [requestSubmission]);

  return (
    <div className="adhoc-workspace filter-workspace">
      <div className="filter-workspace-body">
        <aside className="adhoc-sidebar filter-sidebar">
          <h3 className="adhoc-sidebar-title">Filter Scope</h3>
          <div className="filter-scope-card">
            <label className="filter-scope-field" htmlFor="filter-folder-id">
              <span className="filter-scope-label">Folder</span>
              <input
                id="filter-folder-id"
                aria-label="Folder"
                className="input-text filter-scope-input"
                value={folderId}
                onChange={(event) => setFolderId(event.target.value)}
                placeholder="folder_id"
              />
            </label>
            <label className="filter-scope-field" htmlFor="filter-video-id">
              <span className="filter-scope-label">Video</span>
              <input
                id="filter-video-id"
                aria-label="Video"
                className="input-text filter-scope-input"
                value={videoId}
                onChange={(event) => setVideoId(event.target.value)}
                placeholder="video_id"
              />
            </label>
          </div>
          {isActive && <SubmissionWorktree />}
        </aside>

        <div className="filter-main-column">
          <FilterForm
            query={query}
            onChange={setQuery}
            onSubmit={handleFilter}
            onReset={() => setQuery('')}
            isLoading={isFiltering}
          />
          <div className="filter-results-shell">
            {isFiltering && <div className="filter-loading" role="status">Searching text…</div>}
            <FilterResults
              results={results}
              totalResults={totalResults}
              availableSources={availableSources}
              hasFiltered={hasFiltered}
              error={error}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
              containerRef={resultsViewportRef}
              currentPage={pageId}
              totalPages={totalPages}
              isLoading={isFiltering}
              onPageChange={handlePageChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};


export default FilterWorkspace;

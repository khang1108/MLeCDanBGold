import React, { useCallback, useEffect, useRef, useState } from 'react';
import { filterFrames } from '../../../api/filter';
import FrameCard from '../../frames/components/FrameCard';
import SubmissionWorktree from '../../submission/components/SubmissionWorktree';
import { useSubmissionDialog } from '../../submission/contexts/SubmissionDialogContext';
import { displayVideoId } from '../../frames/videoSource';
import FilterForm from './FilterForm';
import FilterPagination from './FilterPagination';
import { FRAMES_PER_PAGE } from '../filterPagination';


const createInitialFilterValues = () => ({
  title: '',
  asr: '',
  caption: '',
  ocr: '',
  objects: [{ id: 'object-1', value: '' }],
});


const MatchedFrame = ({ frame, onFrameClick, onSubmit }) => (
  <div className="filter-result-card">
    <FrameCard
      frame={frame}
      imageLoading="eager"
      onClick={() => onFrameClick?.(frame)}
      onSubmit={onSubmit}
    />
    {Object.keys(frame.matches || {}).length > 0 && (
      <div className="filter-match-list">
        {Object.entries(frame.matches).map(([source, text]) => (
          <p className="filter-match-text" key={source} title={text}>
            <strong>{source}</strong>: {text}
          </p>
        ))}
      </div>
    )}
  </div>
);


const FilterResults = ({
  results,
  totalResults,
  hasFiltered,
  error,
  onFrameClick,
  onSubmit,
  currentPage,
  totalPages,
  isLoading,
  onPageChange,
}) => {
  if (error) {
    return (
      <section className="frames-container filter-results" aria-label="Filter results">
        <div className="error-alert" role="alert">{error}</div>
      </section>
    );
  }

  if (!hasFiltered) {
    return (
      <section className="frames-container filter-results" aria-label="Filter results">
        <div className="frames-empty-state filter-empty-state">
          <div className="filter-empty-icon" aria-hidden="true">🎯</div>
          <p className="body-md frames-empty-text">Welcome to HCMAI Frame Search</p>
          <p className="caption frames-empty-subtext">
            Enter source-specific keywords above to query the video corpus.
          </p>
        </div>
      </section>
    );
  }

  if (!results.length) {
    return (
      <section className="frames-container filter-results" aria-label="Filter results">
        <div className="frames-empty-state filter-empty-state">
          <div className="filter-empty-icon" aria-hidden="true">🔍</div>
          <p className="body-md frames-empty-text">No matching frames</p>
          <p className="caption frames-empty-subtext">
            Try adjusting your Title, ASR, Caption, OCR, or Object filters.
          </p>
        </div>
        <FilterPagination
          currentPage={currentPage}
          totalPages={totalPages}
          isLoading={isLoading}
          onPageChange={onPageChange}
        />
      </section>
    );
  }

  return (
    <section className="frames-container filter-results" aria-label="Filter results">
      <div className="filter-result-toolbar">
        <div className="filter-result-summary">
          <strong>{totalResults}</strong> matching frame{totalResults === 1 ? '' : 's'}
          {' · '}{FRAMES_PER_PAGE} per page
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


/** Own the source-specific Filter form and backend-owned result pages. */
const FilterWorkspace = ({ isActive = true, onFrameClick }) => {
  const [filters, setFilters] = useState(createInitialFilterValues);
  const [appliedFilters, setAppliedFilters] = useState(null);
  const [folderId, setFolderId] = useState('');
  const [videoId, setVideoId] = useState('');
  const [appliedScope, setAppliedScope] = useState(null);
  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [pageId, setPageId] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [hasFiltered, setHasFiltered] = useState(false);
  const [isFiltering, setIsFiltering] = useState(false);
  const [error, setError] = useState(null);
  const requestRef = useRef(null);
  const { requestSubmission } = useSubmissionDialog();

  useEffect(() => () => requestRef.current?.abort(), []);

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
      setTotalResults(response.total_results || 0);
      setPageId(response.page_id);
      setTotalPages(response.total_pages || 0);
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
    if (isFiltering) return;

    const parameters = { filters, folderId, videoId, pageId: 1 };
    setAppliedFilters(filters);
    setAppliedScope({ folderId, videoId });
    requestPage(parameters);
  }, [filters, folderId, isFiltering, requestPage, videoId]);

  const handlePageChange = useCallback((nextPage) => {
    if (!appliedFilters || isFiltering || nextPage < 1 || nextPage > totalPages) return;
    requestPage({
      filters: appliedFilters,
      folderId: appliedScope?.folderId,
      videoId: appliedScope?.videoId,
      pageId: nextPage,
    });
  }, [appliedFilters, appliedScope, isFiltering, requestPage, totalPages]);

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
            values={filters}
            onChange={setFilters}
            onSubmit={handleFilter}
            onReset={() => setFilters(createInitialFilterValues())}
            isLoading={isFiltering}
          />
          <div className="filter-results-shell">
            {isFiltering && <div className="filter-loading" role="status">Filtering evidence…</div>}
            <FilterResults
              results={results}
              totalResults={totalResults}
              hasFiltered={hasFiltered}
              error={error}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
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

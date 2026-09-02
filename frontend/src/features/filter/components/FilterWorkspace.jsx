import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { filterFrames } from '../../../api/filter';
import FrameCard from '../../frames/components/FrameCard';
import SubmissionWorktree from '../../submission/components/SubmissionWorktree';
import { useSubmission } from '../../submission/contexts/SubmissionContext';
import { displayVideoId } from '../../frames/videoSource';
import FilterForm from './FilterForm';
import FilterPagination from './FilterPagination';
import FilterPageSize from './FilterPageSize';
import FolderScopeCombobox from './FolderScopeCombobox';
import { DEFAULT_FRAMES_PER_PAGE, resolveFramesPerPage } from '../filterPagination';
import {
  FILTER_FOLDER_IDS,
  filterResultsByScope,
  getFrameFolderId,
  normalizeFolderId,
} from '../filterUtils';

const createInitialFilterValues = () => ({
  title: '',
  asr: '',
  caption: '',
  ocr: '',
  objects: [{ id: 'object-1', value: '' }],
});

const FilterResults = ({
  results,
  hasFiltered,
  error,
  folderId,
  selectedVideoId,
  onFrameClick,
  onSubmit,
  containerRef,
  currentPage,
  totalPages,
  isLoading,
  onPageChange,
  pageSizeMode,
  onPageSizeChange,
}) => {
  const pagination = (
    <FilterPagination
      currentPage={currentPage}
      totalPages={totalPages}
      isLoading={isLoading}
      onPageChange={onPageChange}
    />
  );
  const pageSizeControl = (
    <FilterPageSize
      value={pageSizeMode}
      onChange={onPageSizeChange}
      disabled={isLoading}
    />
  );

  const renderFrame = (frame) => {
    return (
      <FrameCard
        key={frame.frame_id}
        frame={frame}
        imageLoading="eager"
        onClick={() => onFrameClick?.(frame)}
        onSubmit={onSubmit}
      />
    );
  };

  if (error) {
    return (
      <section
        ref={containerRef}
        className="frames-container filter-results"
        aria-label="Filter results"
      >
        <div className="error-alert" role="alert">
          <div className="error-details">
            <h4 className="error-title">Filter Connection Error</h4>
            <p className="error-message">{error}</p>
          </div>
        </div>
      </section>
    );
  }

  if (!hasFiltered) {
    return (
      <section
        ref={containerRef}
        className="frames-container filter-results"
        aria-label="Filter results"
      >
        <div className="frames-empty-state filter-empty-state">
          <p className="body-md frames-empty-text">Welcome to HCMAI Frame Search</p>
          <p className="caption frames-empty-subtext">
            Enter a natural language question or keywords above to query the video corpus.
          </p>
        </div>
      </section>
    );
  }

  if (!results.length) {
    return (
      <section
        ref={containerRef}
        className="frames-container filter-results"
        aria-label="Filter results"
      >
        <div className="filter-result-toolbar">
          <div className="filter-result-summary">No frames match the current scope.</div>
          {pageSizeControl}
        </div>
        <div className="frames-empty-state filter-empty-state">
          <p className="body-md frames-empty-text">No matching frames</p>
        </div>
        {pagination}
      </section>
    );
  }

  return (
    <section
      ref={containerRef}
      className="frames-container filter-results"
      aria-label="Filter results"
    >
      <div className="filter-result-toolbar">
        <div className="filter-result-summary">
          <strong>{results.length}</strong> frame{results.length === 1 ? '' : 's'}
          {folderId ? ` · ${folderId}` : ''}
          {selectedVideoId ? ` · ${displayVideoId(selectedVideoId)}` : ''}
        </div>
        {pageSizeControl}
      </div>
      <div className="frames-scroll-region">
        <div className="frames-grid">{results.map(renderFrame)}</div>
      </div>
      {pagination}
    </section>
  );
};

/**
 * Independent metadata-filter page. It owns filter/scope/detail state while
 * reusing the Query viewer and submission workflow at the application level.
 */
const FilterWorkspace = ({ isActive = true, onFrameClick }) => {
  const [filters, setFilters] = useState(createInitialFilterValues);
  const [activeFolder, setActiveFolder] = useState(null);
  const [selectedVideoId, setSelectedVideoId] = useState('');
  const [results, setResults] = useState([]);
  const [appliedFilters, setAppliedFilters] = useState(createInitialFilterValues);
  const [appliedScope, setAppliedScope] = useState({ folderId: null, videoId: '' });
  const [appliedFramesPerPage, setAppliedFramesPerPage] = useState(DEFAULT_FRAMES_PER_PAGE);
  const [hasFiltered, setHasFiltered] = useState(false);
  const [isFiltering, setIsFiltering] = useState(false);
  const [error, setError] = useState(null);
  const [pageId, setPageId] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [pageSizeMode, setPageSizeMode] = useState('auto');
  const requestRef = useRef(null);
  const resultsViewportRef = useRef(null);
  const { requestSubmission } = useSubmission();

  const folderIds = FILTER_FOLDER_IDS;
  const videoIds = useMemo(() => {
    const scopedResults = filterResultsByScope(results, { folderId: activeFolder });
    return Array.from(new Set(
      scopedResults.map((frame) => frame.video_id).filter(Boolean),
    ));
  }, [activeFolder, results]);

  useEffect(() => () => {
    requestRef.current?.abort();
  }, []);

  const resolveCurrentPageSize = useCallback((mode) => {
    const element = resultsViewportRef.current;
    return resolveFramesPerPage(mode, {
      width: element?.clientWidth,
      height: element?.clientHeight,
    });
  }, []);

  const requestFilterPage = useCallback(async ({
    requestedFilters,
    requestedFolderId,
    requestedVideoId,
    requestedPage,
    requestedFramesPerPage,
    resetPagination = false,
  }) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsFiltering(true);
    setHasFiltered(true);
    setError(null);
    setResults([]);
    setPageId(requestedPage);
    if (resetPagination) setTotalPages(1);

    try {
      const response = await filterFrames({
        filters: requestedFilters,
        folderId: requestedFolderId,
        videoId: requestedVideoId,
        framesPerPage: requestedFramesPerPage,
        pageId: requestedPage,
        signal: controller.signal,
      });
      setResults(response.results || []);
      setTotalPages(response.total_pages);
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setError(requestError.message || 'Failed to contact filter API');
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

    // A new filter is a new result set, so it must always start on page 1.
    const nextFramesPerPage = resolveCurrentPageSize(pageSizeMode);
    setAppliedFilters(filters);
    setAppliedScope({ folderId: activeFolder, videoId: selectedVideoId });
    setAppliedFramesPerPage(nextFramesPerPage);
    requestFilterPage({
      requestedFilters: filters,
      requestedFolderId: activeFolder,
      requestedVideoId: selectedVideoId,
      requestedPage: 1,
      requestedFramesPerPage: nextFramesPerPage,
      resetPagination: true,
    });
  }, [activeFolder, filters, isFiltering, pageSizeMode, requestFilterPage, resolveCurrentPageSize, selectedVideoId]);

  const handlePageSizeChange = useCallback((nextMode) => {
    setPageSizeMode(nextMode);
    const nextFramesPerPage = resolveCurrentPageSize(nextMode);
    if (!hasFiltered || isFiltering) return;

    setAppliedFramesPerPage(nextFramesPerPage);
    requestFilterPage({
      requestedFilters: appliedFilters,
      requestedFolderId: appliedScope.folderId,
      requestedVideoId: appliedScope.videoId,
      requestedPage: 1,
      requestedFramesPerPage: nextFramesPerPage,
      resetPagination: true,
    });
  }, [appliedFilters, appliedScope, hasFiltered, isFiltering, requestFilterPage, resolveCurrentPageSize]);

  const handlePageChange = useCallback((nextPage) => {
    if (isFiltering || nextPage === pageId
        || nextPage < 1 || nextPage > totalPages) return;

    requestFilterPage({
      requestedFilters: appliedFilters,
      requestedFolderId: appliedScope.folderId,
      requestedVideoId: appliedScope.videoId,
      requestedPage: nextPage,
      requestedFramesPerPage: appliedFramesPerPage,
    });
  }, [appliedFilters, appliedFramesPerPage, appliedScope, isFiltering, pageId, requestFilterPage, totalPages]);

  const handleReset = useCallback(() => {
    setFilters(createInitialFilterValues());
  }, []);

  const handleFolderChange = useCallback((folderId) => {
    const nextFolder = folderId.trim();
    if (selectedVideoId && nextFolder && !results.some((frame) => (
      frame.video_id === selectedVideoId
      && getFrameFolderId(frame) === normalizeFolderId(nextFolder)
    ))) {
      setSelectedVideoId('');
    }
    setActiveFolder(nextFolder || null);
  }, [results, selectedVideoId]);

  const handleFrameSubmit = useCallback((frame) => {
    const videoId = displayVideoId(frame.video_id);
    requestSubmission({
      line: `${videoId},${frame.frame_idx}`,
      source: 'KIS frame',
    });
  }, [requestSubmission]);

  return (
    <div className="adhoc-workspace filter-workspace">
      <div className="filter-workspace-body">
        <aside className="adhoc-sidebar filter-sidebar">
          <label className="filter-scope-field" htmlFor="filter-folder-id">
            <span>Folder</span>
            <FolderScopeCombobox
              value={activeFolder || ''}
              options={folderIds}
              onChange={handleFolderChange}
            />
          </label>

          <label className="filter-scope-field" htmlFor="filter-video-id">
            <span>Video</span>
            {hasFiltered && videoIds.length > 0 ? (
              <FolderScopeCombobox
                inputId="filter-video-id"
                scopeLabel="video"
                placeholder="video_id"
                value={selectedVideoId}
                options={videoIds}
                onChange={setSelectedVideoId}
              />
            ) : (
              <input
                id="filter-video-id"
                className="input-text filter-scope-input"
                type="text"
                value={selectedVideoId}
                onChange={(event) => setSelectedVideoId(event.target.value)}
                placeholder="video_id"
                autoComplete="off"
              />
            )}
          </label>

          {isActive && <SubmissionWorktree />}
        </aside>

        <div className="filter-main-column">
          <FilterForm
            values={filters}
            onChange={setFilters}
            onSubmit={handleFilter}
            onReset={handleReset}
            isLoading={isFiltering}
          />
          <div className="filter-results-shell">
            {isFiltering && <div className="filter-loading" role="status">Loading filter results…</div>}
            <FilterResults
              results={results}
              hasFiltered={hasFiltered}
              error={error}
              folderId={activeFolder}
              selectedVideoId={selectedVideoId}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
              containerRef={resultsViewportRef}
              currentPage={pageId}
              totalPages={totalPages}
              isLoading={isFiltering}
              onPageChange={handlePageChange}
              pageSizeMode={pageSizeMode}
              onPageSizeChange={handlePageSizeChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterWorkspace;

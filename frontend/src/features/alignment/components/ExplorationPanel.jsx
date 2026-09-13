import React from "react";
import { keyframeUrl } from "../../../api/keyframes";

const validTime = (value) => Number.isFinite(value) && value >= 0;

const formatTime = (value) => (validTime(value) ? `${(value / 1000).toFixed(3)} s` : "—");

const ExplorationPanel = ({
  events = [], session,
  pending = false, error,
  onApprove, onDecline, onUndo, onSearchRange, onBack, onSeek,
  readCurrentTimeMs,
}) => {
  const view = session?.view;
  const [selectedEvent, setSelectedEvent] = React.useState(0);
  const [draft, setDraft] = React.useState({ start: null, end: null });
  const [searchDraft, setSearchDraft] = React.useState({ start: null, end: null });

  React.useEffect(() => {
    setSelectedEvent(0);
    setDraft({ start: null, end: null });
  }, [events.length, session?.handle]);

  if (!session || !view) return null;
  const eventList = Array.isArray(view.events) ? view.events : events;
  const paths = Array.isArray(view.paths) ? view.paths : [];
  const currentPath = paths[selectedEvent] || paths[0];
  const approvedRanges = view.conditions?.approved_ranges || view.conditions?.confirmed || [];
  const declinedRanges = view.conditions?.declined_ranges || view.conditions?.rejected || [];
  const intervalValid = validTime(draft.start) && validTime(draft.end) && draft.start <= draft.end;
  const searchValid = validTime(searchDraft.start)
    && validTime(searchDraft.end) && searchDraft.start <= searchDraft.end;
  const capture = (end) => {
    const seconds = Number(readCurrentTimeMs?.());
    if (!validTime(seconds)) return;
    setDraft((previous) => ({ ...previous, [end]: Math.round(seconds * 1000) }));
  };
  const selectEvent = (index) => {
    setSelectedEvent(index);
    setDraft({ start: null, end: null });
  };
  const updateSearch = (end, value) => {
    const milliseconds = Number(value);
    setSearchDraft((previous) => ({ ...previous, [end]: validTime(milliseconds) ? Math.round(milliseconds) : null }));
  };
  const renderPath = (path, index) => {
    const frameId = path?.frame_ids?.[index];
    const timestamp = path?.timestamps_ms?.[index];
    if (!frameId && !validTime(timestamp)) return null;
    return (
      <button type="button" className="exploration-moment" key={`${frameId || "frame"}-${index}`} onClick={() => onSeek?.(timestamp)}>
        {frameId && <img src={keyframeUrl(frameId)} alt={`Moment ${frameId}`} />}
        <span>Updated moment · {formatTime(timestamp)}</span>
      </button>
    );
  };

  return (
    <section className="exploration-panel" aria-label="Explore">
      <div className="exploration-panel-header"><h2>Explore</h2>{onBack && <button type="button" onClick={onBack}>Back to results</button>}</div>
      <p>Applies to the selected event and time range.</p>
      <div className="exploration-events" aria-label="Event">
        {eventList.map((event, index) => <button type="button" key={`${event}-${index}`} onClick={() => selectEvent(index)}>E{index + 1}</button>)}
      </div>
      <p className="exploration-selected-event">Event: {eventList[selectedEvent] || "—"}</p>
      <div className="exploration-range-fields">
        <span>Start: {formatTime(draft.start)}</span>
        <span>End: {formatTime(draft.end)}</span>
      </div>
      <div className="exploration-draft-actions">
        <button type="button" onClick={() => capture("start")}>Use current time as start</button>
        <button type="button" onClick={() => capture("end")}>Use current time as end</button>
      </div>
      <div className="exploration-feedback-actions">
        <button type="button" disabled={!intervalValid || pending} onClick={() => onApprove?.({ event_index: selectedEvent, interval: [draft.start, draft.end] })}>Approve</button>
        <button type="button" disabled={!intervalValid || pending} onClick={() => onDecline?.({ event_index: selectedEvent, interval: [draft.start, draft.end] })}>Decline</button>
      </div>
      <div className="exploration-search-range">
        <h3>Search range</h3>
        <input aria-label="Search range start" type="number" min="0" value={searchDraft.start ?? ""} onChange={(event) => updateSearch("start", event.target.value)} />
        <input aria-label="Search range end" type="number" min="0" value={searchDraft.end ?? ""} onChange={(event) => updateSearch("end", event.target.value)} />
        <button type="button" disabled={!searchValid || pending} onClick={() => onSearchRange?.({ interval: [searchDraft.start, searchDraft.end] })}>Search this range</button>
      </div>
      {view.can_undo && <button type="button" disabled={pending} onClick={onUndo}>Undo</button>}
      {error && <p role="alert">{error}</p>}
      {view.conditions && <div className="exploration-conditions">Conditions: {JSON.stringify(view.conditions)}</div>}
      {approvedRanges.length > 0 && <div>Approved range: {JSON.stringify(approvedRanges)}</div>}
      {declinedRanges.length > 0 && <div>Declined ranges: {JSON.stringify(declinedRanges)}</div>}
      {currentPath && <div className="exploration-path">{currentPath.frame_ids?.map((_, index) => renderPath(currentPath, index))}</div>}
    </section>
  );
};

export default ExplorationPanel;

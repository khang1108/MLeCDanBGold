/** Adapt a history snapshot to the same result UI used by live Query.

Replay never invokes KIS or TRAKE retrieval. The snapshot supplies the same
result and metadata fields returned by live search, so ImageModal opens from
the snapshot without a second frame-detail request.
*/
import React, { useCallback } from 'react';
import FramesBox from '../../frames/components/FramesBox';
import TrakeResults from '../../search/components/TrakeResults';
import { activityStateForFrame } from '../queryHistory';

const ReplayResults = ({
  resultSnapshot,
  frameActivity,
  onFrameClick,
  onFrameSubmit,
  onSubmit,
  onPathSubmit,
  onTrakeSubmit,
}) => {
  const frameSubmit = onFrameSubmit || onSubmit;
  const pathSubmit = onPathSubmit || onTrakeSubmit;

  const openFrame = useCallback((frame, submissionMode) => {
    onFrameClick?.(frame, submissionMode);
  }, [onFrameClick]);

  const getFrameClassName = useCallback(
    (frameOrId) => activityStateForFrame(
      typeof frameOrId === 'string' ? frameOrId : frameOrId.frame_id,
      frameActivity,
    ),
    [frameActivity],
  );

  if (Array.isArray(resultSnapshot?.results)) {
    return (
      <FramesBox
        results={resultSnapshot.results}
        isLoading={false}
        error={null}
        latencyMs={resultSnapshot.latency ?? null}
        warnings={resultSnapshot.warnings || []}
        events={resultSnapshot.events || []}
        getFrameClassName={getFrameClassName}
        onFrameClick={(frame) => openFrame(frame, 'kis')}
        onSubmit={frameSubmit}
      />
    );
  }

  if (Array.isArray(resultSnapshot?.paths)) {
    return (
      <TrakeResults
        events={resultSnapshot.events || []}
        paths={resultSnapshot.paths}
        warnings={resultSnapshot.warnings || []}
        error={null}
        hasSearched
        getFrameClassName={getFrameClassName}
        onFrameClick={(frame) => openFrame(frame, 'none')}
        onTrakeSubmit={pathSubmit}
      />
    );
  }

  return <div className="workspace-empty-copy">This history snapshot cannot be replayed.</div>;
};

export default ReplayResults;

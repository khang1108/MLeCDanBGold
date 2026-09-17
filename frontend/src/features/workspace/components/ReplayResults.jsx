/** Adapt a KIS history snapshot to the same result UI used by live Query.

Replay never invokes retrieval. The snapshot supplies the same result and
metadata fields returned by live search, so the inspector opens without a
second frame-detail request. Legacy path snapshots remain unsupported.
*/
import React, { useCallback, useMemo } from 'react';
import FramesBox from '../../frames/components/FramesBox';
import { activityStateForFrame } from '../queryHistory';

const ReplayResults = ({
  resultSnapshot,
  frameActivity,
  onFrameClick,
}) => {
  const replayEvents = useMemo(
    () => (Array.isArray(resultSnapshot?.intent?.events)
      ? resultSnapshot.intent.events.map((e) => (typeof e === 'string' ? e : e.text))
      : (resultSnapshot?.events || [])),
    [resultSnapshot?.intent?.events, resultSnapshot?.events],
  );

  const openFrame = useCallback((frame) => {
    onFrameClick?.({
      ...frame,
      events: frame.events || replayEvents,
    });
  }, [onFrameClick, replayEvents]);

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
        events={replayEvents}
        getFrameClassName={getFrameClassName}
        onFrameClick={openFrame}
      />
    );
  }

  return <div className="workspace-empty-copy">This history snapshot cannot be replayed.</div>;
};

export default ReplayResults;

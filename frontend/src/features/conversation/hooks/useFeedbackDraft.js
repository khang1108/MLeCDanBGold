import { useEffect, useMemo, useState } from 'react';
import { emptyFeedback, feedbackFromSession, sameFeedback, toggleFeedback } from '../utils/feedback';

// Holds local card clicks until the user submits the full feedback snapshot.
export const useFeedbackDraft = (session, isPending) => {
  const [draftFeedback, setDraftFeedback] = useState(emptyFeedback);
  const committedFeedback = useMemo(() => feedbackFromSession(session?.feedback), [session]);

  useEffect(() => setDraftFeedback(committedFeedback), [committedFeedback]);

  const feedbackDirty = Boolean(session) && !sameFeedback(draftFeedback, committedFeedback);
  const toggle = (frameId, decision) => {
    if (!session || isPending) return;
    setDraftFeedback((current) => toggleFeedback(current, frameId, decision));
  };
  const stateFor = (frameId) => {
    if (draftFeedback.accepted_frame_ids.includes(frameId)) return 'promising';
    return draftFeedback.rejected_frame_ids.includes(frameId) ? 'rejected' : null;
  };

  return { committedFeedback, draftFeedback, feedbackDirty, stateFor, toggle };
};
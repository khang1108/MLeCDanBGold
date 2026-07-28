// Keeps the UI feedback draft aligned with the FrameFeedback contract.
export const emptyFeedback = () => ({ accepted_frame_ids: [], rejected_frame_ids: [] });

export const feedbackFromSession = (feedback) => ({
  accepted_frame_ids: Array.isArray(feedback?.accepted_frame_ids) ? [...feedback.accepted_frame_ids] : [],
  rejected_frame_ids: Array.isArray(feedback?.rejected_frame_ids) ? [...feedback.rejected_frame_ids] : [],
});

const sameIds = (left, right) => left.length === right.length
  && left.every((value, index) => value === right[index]);

export const sameFeedback = (left, right) => (
  sameIds(left.accepted_frame_ids, right.accepted_frame_ids)
  && sameIds(left.rejected_frame_ids, right.rejected_frame_ids)
);

// A frame can have one decision at most; clicking an active decision clears the draft.
export const toggleFeedback = (feedback, frameId, decision) => {
  const ownKey = decision === 'promising' ? 'accepted_frame_ids' : 'rejected_frame_ids';
  const otherKey = decision === 'promising' ? 'rejected_frame_ids' : 'accepted_frame_ids';
  const active = feedback[ownKey].includes(frameId);
  return {
    [ownKey]: active ? feedback[ownKey].filter((id) => id !== frameId) : [...feedback[ownKey], frameId],
    [otherKey]: feedback[otherKey].filter((id) => id !== frameId),
  };
};
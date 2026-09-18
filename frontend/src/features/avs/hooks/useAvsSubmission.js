import { useCallback, useState } from 'react';
import { submitDresAnswers } from '../../../api/submissions';

/**
 * Failure-safe batch submission hook for AVS workspace.
 * Submits frozen batches of temporal answers with definite state transitions:
 * RECORDED, NOT_RECORDED, UNKNOWN, or IDLE with TASK_SCOPE_MISMATCH error.
 */
export const useAvsSubmission = ({
  userId = '',
  selectedTask = null,
  taskScopeKey = null,
  onSessionRejected,
} = {}) => {
  const [status, setStatus] = useState('IDLE');
  const [outcome, setOutcome] = useState(null);
  const [error, setError] = useState(null);

  const submitBatch = useCallback(async (batch) => {
    if (!batch || !Array.isArray(batch.answers) || batch.answers.length === 0) {
      return null;
    }

    setStatus('SUBMITTING');
    setError(null);

    try {
      const result = await submitDresAnswers({
        userId,
        expectedTaskScopeKey: taskScopeKey,
        evaluationId: selectedTask?.evaluationId,
        taskName: selectedTask?.taskName,
        answers: batch.answers,
      });

      const nextStatus = result?.state || 'UNKNOWN';
      setStatus(nextStatus);
      setOutcome(result);

      if (nextStatus === 'NOT_RECORDED' && result?.reason === 'DRES_AUTH_REJECTED') {
        onSessionRejected?.();
      }

      return result;
    } catch (err) {
      if (err?.code === 'TASK_SCOPE_MISMATCH' || err?.status === 409) {
        setStatus('IDLE');
        setError(err);
        return { error: err };
      }

      const fallbackOutcome = {
        state: 'UNKNOWN',
        recorded: null,
        message: err?.message || 'Network error while contacting DRES',
      };
      setStatus('UNKNOWN');
      setOutcome(fallbackOutcome);
      setError(err);
      return fallbackOutcome;
    }
  }, [userId, selectedTask?.evaluationId, selectedTask?.taskName, taskScopeKey, onSessionRejected]);

  const retryUnknown = useCallback(async (batch) => {
    return submitBatch(batch);
  }, [submitBatch]);

  const resetOutcome = useCallback(() => {
    setStatus('IDLE');
    setOutcome(null);
    setError(null);
  }, []);

  return {
    status,
    outcome,
    error,
    submitBatch,
    retryUnknown,
    resetOutcome,
  };
};

import { useCallback, useMemo, useRef, useState } from 'react';
import { getCurrentDresTask, submitDresAnswer } from '../../../api/submissions';

/**
 * Hook managing text-based answer submission and immediate DRES verdict verification for QA/VQA tasks.
 */
export const useQaSubmission = ({ userId, selectedTask, onSessionRejected } = {}) => {
  const [draft, setDraft] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [lastOutcome, setLastOutcome] = useState(null);
  const [historyByTask, setHistoryByTask] = useState({});

  const submittingRef = useRef(false);

  const taskKey = useMemo(() => {
    if (!selectedTask?.evaluationId || !selectedTask?.taskName) return null;
    return `${selectedTask.evaluationId}:${selectedTask.taskName}`;
  }, [selectedTask?.evaluationId, selectedTask?.taskName]);

  const history = useMemo(() => {
    if (!taskKey) return [];
    return historyByTask[taskKey] || [];
  }, [historyByTask, taskKey]);

  const submit = useCallback(async (textOverride) => {
    const participantId = typeof userId === 'string' ? userId.trim() : '';
    if (!participantId) {
      setError('Connect a VBS participant before submitting an answer.');
      return;
    }
    if (!selectedTask?.evaluationId || !selectedTask?.taskName) {
      setError('Select an active QA task before submitting.');
      return;
    }

    const textToSubmit = (typeof textOverride === 'string' ? textOverride : draft).trim();
    if (!textToSubmit) {
      setError('Enter a non-empty text answer.');
      return;
    }

    if (submittingRef.current) return;
    submittingRef.current = true;
    setIsSubmitting(true);
    setError('');

    try {
      const liveTask = await getCurrentDresTask(participantId, {
        evaluationId: selectedTask.evaluationId,
        taskName: selectedTask.taskName,
      });

      const taskScopeKey = liveTask?.task_scope_key;
      if (!taskScopeKey) {
        throw new Error('Could not resolve active DRES task scope.');
      }

      const outcome = await submitDresAnswer({
        userId: participantId,
        expectedTaskScopeKey: taskScopeKey,
        evaluationId: selectedTask.evaluationId,
        taskName: selectedTask.taskName,
        answer: { kind: 'TEXT', text: textToSubmit },
      });

      const timestamp = Date.now();
      const submissionRecord = {
        id: `sub_${timestamp}_${Math.random().toString(36).slice(2, 7)}`,
        text: textToSubmit,
        state: outcome?.state || 'UNKNOWN',
        verdict: outcome?.verdict || null,
        message: outcome?.message || '',
        reason: outcome?.reason || null,
        timestamp,
      };

      setLastOutcome(submissionRecord);

      const currentKey = `${selectedTask.evaluationId}:${selectedTask.taskName}`;
      setHistoryByTask((prev) => ({
        ...prev,
        [currentKey]: [submissionRecord, ...(prev[currentKey] || [])],
      }));

      if (outcome?.state === 'RECORDED' && outcome?.verdict === 'CORRECT') {
        setDraft('');
      }

      if (outcome?.reason === 'DRES_AUTH_REJECTED') {
        await onSessionRejected?.(participantId);
      }
    } catch (err) {
      const message = err?.message || 'Submission failed.';
      setError(message);
      if (err?.status === 401) {
        await onSessionRejected?.(participantId);
      }
    } finally {
      submittingRef.current = false;
      setIsSubmitting(false);
    }
  }, [userId, selectedTask, draft, onSessionRejected]);

  const clearHistory = useCallback(() => {
    if (!taskKey) return;
    setHistoryByTask((prev) => {
      const next = { ...prev };
      delete next[taskKey];
      return next;
    });
  }, [taskKey]);

  return {
    draft,
    setDraft,
    isSubmitting,
    error,
    setError,
    lastOutcome,
    history,
    submit,
    clearHistory,
  };
};

export default useQaSubmission;

import { useCallback, useRef, useState } from 'react';
import { getCurrentDresTask, submitDresAnswer } from '../../../api/submissions';
import { formatTemporalAnswer, parseAnswerLine } from '../answerFormat';

const recordedOutcomes = new Set(['RECORDED', 'NOT_RECORDED', 'UNKNOWN']);

/** Hold one task-scoped draft in React memory and forward it exactly once. */
export const useDirectSubmission = ({ userId, selectedTask, onSessionRejected } = {}) => {
  const [dialog, setDialog] = useState(null);
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState('');
  const openingRef = useRef(false);
  const submittingRef = useRef(false);
  const dialogSequenceRef = useRef(0);

  const open = useCallback(async ({ videoId, startMs, endMs } = {}) => {
    const participantId = typeof userId === 'string' ? userId.trim() : '';
    if (!participantId) {
      setOpenError('Connect a VBS participant before submitting an answer.');
      return;
    }
    if (openingRef.current || submittingRef.current) return;

    openingRef.current = true;
    setOpening(true);
    setOpenError('');
    try {
      const task = await getCurrentDresTask(participantId, {
        evaluationId: selectedTask?.evaluationId,
        taskName: selectedTask?.taskName,
      });
      const taskScopeKey = task?.task_scope_key;
      if (typeof taskScopeKey !== 'string' || !taskScopeKey.trim()) {
        throw new Error('The backend did not provide a current DRES task scope.');
      }
      const initialValue = formatTemporalAnswer({ videoId, startMs, endMs });
      const id = ++dialogSequenceRef.current;
      setDialog({
        id,
        userId: participantId,
        expectedTaskScopeKey: taskScopeKey,
        evaluationId: selectedTask?.evaluationId || task.evaluation_id,
        taskName: selectedTask?.taskName || task.task_name,
        task,
        value: initialValue,
        outcome: null,
        error: '',
      });
    } catch (error) {
      setOpenError(error?.message || 'Could not load the active DRES task.');
    } finally {
      openingRef.current = false;
      setOpening(false);
    }
  }, [userId, selectedTask]);

  const updateValue = useCallback((value) => {
    setDialog((current) => current ? { ...current, value, error: '' } : current);
  }, []);

  const close = useCallback(() => {
    setDialog(null);
  }, []);

  const submit = useCallback(async (rawValue = dialog?.value) => {
    if (!dialog || submittingRef.current) return;

    let answer;
    try {
      answer = parseAnswerLine(rawValue);
    } catch (error) {
      setDialog((current) => current?.id === dialog.id
        ? { ...current, error: error?.message || 'Enter one non-blank answer line.' }
        : current);
      return;
    }

    submittingRef.current = true;
    setDialog((current) => current?.id === dialog.id
      ? { ...current, error: '', outcome: null, submitting: true }
      : current);
    try {
      const outcome = await submitDresAnswer({
        userId: dialog.userId,
        expectedTaskScopeKey: dialog.expectedTaskScopeKey,
        evaluationId: dialog.evaluationId,
        taskName: dialog.taskName,
        answer,
      });
      if (!outcome || !recordedOutcomes.has(outcome.state)) {
        throw new Error('The backend returned an invalid DRES submission outcome.');
      }
      if (outcome.state === 'RECORDED') {
        setDialog((current) => current?.id === dialog.id ? null : current);
      } else {
        setDialog((current) => current?.id === dialog.id
          ? { ...current, submitting: false, outcome, error: '' }
          : current);
        if (outcome.reason === 'DRES_AUTH_REJECTED') {
          await onSessionRejected?.(dialog.userId);
        }
      }
    } catch (error) {
      setDialog((current) => current?.id === dialog.id
        ? { ...current, submitting: false, outcome: null, error: error?.message || 'Submission failed.' }
        : current);
    } finally {
      submittingRef.current = false;
    }
  }, [dialog, onSessionRejected]);

  return {
    dialog,
    opening,
    openError,
    open,
    updateValue,
    submit,
    close,
  };
};

export default useDirectSubmission;

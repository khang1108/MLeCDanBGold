import { useCallback, useState } from 'react';
import {
  getMiniChallengeCurrentTask,
  listMiniChallengeEvaluations,
  submitMiniChallengeFrame,
} from '../../../api/minichallenge';

export const useMiniChallenge = () => {
  const [session, setSession] = useState('');
  const [evaluations, setEvaluations] = useState([]);
  const [evaluationId, setEvaluationId] = useState('');
  const [currentTask, setCurrentTask] = useState(null);
  const [answer, setAnswer] = useState('');
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [submittingFrameId, setSubmittingFrameId] = useState(null);
  const selectedEvaluation = evaluations.find((item) => item.id === evaluationId);
  const matchingTemplate = selectedEvaluation?.taskTemplates?.find(
    (template) => template.taskGroup === currentTask?.taskGroup
      && template.taskType === currentTask?.taskType,
  );
  const submissionTaskName = matchingTemplate?.name
    || selectedEvaluation?.taskTemplates?.[0]?.name
    || currentTask?.name
    || '';

  const loadTask = useCallback(async (token, id) => {
    setCurrentTask(null);
    if (!id) return;
    try {
      setCurrentTask(await getMiniChallengeCurrentTask(token, id));
    } catch (requestError) {
      setError(requestError.message || 'Could not load the current task.');
    }
  }, []);

  const connect = useCallback(async () => {
    const token = session.trim();
    if (!token || isLoading) return;
    setIsLoading(true);
    setError(null);
    setNotice(null);
    setCurrentTask(null);
    try {
      const values = await listMiniChallengeEvaluations(token);
      setEvaluations(values);
      const selected = values.find((item) => item.status === 'ACTIVE') || values[0];
      const selectedId = selected?.id || '';
      setEvaluationId(selectedId);
      if (selectedId) {
        await loadTask(token, selectedId);
      } else {
        setNotice('No evaluation is visible for this session.');
      }
    } catch (requestError) {
      setEvaluations([]);
      setEvaluationId('');
      setError(requestError.message || 'Could not load evaluations.');
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, loadTask, session]);

  const selectEvaluation = useCallback(async (id) => {
    setEvaluationId(id);
    setError(null);
    setNotice(null);
    setIsLoading(true);
    await loadTask(session.trim(), id);
    setIsLoading(false);
  }, [loadTask, session]);

  const submitFrame = useCallback(async (frame) => {
    if (!currentTask || !submissionTaskName || !evaluationId
      || !session.trim() || submittingFrameId) {
      return null;
    }
    setSubmittingFrameId(frame.frame_id);
    setError(null);
    setNotice(null);
    try {
      const result = await submitMiniChallengeFrame({
        session,
        evaluationId,
        frameId: frame.frame_id,
        taskName: submissionTaskName,
        text: answer,
      });
      setNotice(`${result.submission}: ${result.description}`);
      return result;
    } catch (requestError) {
      setError(requestError.message || 'Submission failed.');
      return null;
    } finally {
      setSubmittingFrameId(null);
    }
  }, [answer, currentTask, evaluationId, session, submissionTaskName, submittingFrameId]);

  return {
    session,
    setSession,
    evaluations,
    evaluationId,
    currentTask,
    submissionTaskName,
    answer,
    setAnswer,
    error,
    notice,
    isLoading,
    submittingFrameId,
    connect,
    selectEvaluation,
    submitFrame,
  };
};

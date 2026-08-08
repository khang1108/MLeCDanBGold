import { useCallback, useEffect, useState } from 'react';
import {
  getMiniChallengeConfig,
  getMiniChallengeCurrentTask,
  listMiniChallengeEvaluations,
  loginMiniChallenge,
  submitMiniChallengeFrame,
} from '../../../api/minichallenge';

export const useMiniChallenge = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [session, setSession] = useState('');
  const [userRole, setUserRole] = useState(null);
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
      const task = await getMiniChallengeCurrentTask(token, id);
      setCurrentTask(task);
    } catch (requestError) {
      setError(requestError.message || 'Could not load the current task.');
    }
  }, []);

  const fetchEvaluations = useCallback(async (token) => {
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
    return values;
  }, [loadTask]);

  useEffect(() => {
    let cancelled = false;
    const initFromConfig = async () => {
      try {
        const config = await getMiniChallengeConfig();
        if (cancelled) return;
        if (config?.session_id) {
          setSession(config.session_id);
          if (config.username) setUsername(config.username);
          if (config.role) setUserRole(config.role);
          await fetchEvaluations(config.session_id);
        }
      } catch {
        // Backend config is optional or server still initializing
      }
    };
    initFromConfig();
    return () => {
      cancelled = true;
    };
  }, [fetchEvaluations]);

  const login = useCallback(async (customUser, customPass) => {
    const user = (customUser ?? username).trim();
    const pass = customPass ?? password;
    if (!user || !pass || isLoading) return;
    setIsLoading(true);
    setError(null);
    setNotice(null);
    try {
      const auth = await loginMiniChallenge(user, pass);
      setSession(auth.sessionId);
      setUserRole(auth.role || 'PARTICIPANT');
      setNotice(`Logged in as ${auth.username}`);
      await fetchEvaluations(auth.sessionId);
    } catch (requestError) {
      setError(requestError.message || 'Login failed.');
    } finally {
      setIsLoading(false);
    }
  }, [fetchEvaluations, isLoading, password, username]);

  const connect = useCallback(async () => {
    const token = session.trim();
    if (!token || isLoading) return;
    setIsLoading(true);
    setError(null);
    setNotice(null);
    setCurrentTask(null);
    try {
      await fetchEvaluations(token);
      setNotice('Connected to DRES evaluation.');
    } catch (requestError) {
      setEvaluations([]);
      setEvaluationId('');
      setError(requestError.message || 'Could not load evaluations.');
    } finally {
      setIsLoading(false);
    }
  }, [fetchEvaluations, isLoading, session]);

  const selectEvaluation = useCallback(async (id) => {
    setEvaluationId(id);
    setError(null);
    setNotice(null);
    setIsLoading(true);
    await loadTask(session.trim(), id);
    setIsLoading(false);
  }, [loadTask, session]);

  const refreshTask = useCallback(async () => {
    if (!session.trim() || !evaluationId || isLoading) return;
    setIsLoading(true);
    setError(null);
    await loadTask(session.trim(), evaluationId);
    setIsLoading(false);
  }, [evaluationId, isLoading, loadTask, session]);

  const logout = useCallback(() => {
    setSession('');
    setUserRole(null);
    setEvaluations([]);
    setEvaluationId('');
    setCurrentTask(null);
    setAnswer('');
    setError(null);
    setNotice(null);
  }, []);

  const submitFrame = useCallback(async (frame, overrideAnswer = null) => {
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
        text: overrideAnswer !== null ? overrideAnswer : answer,
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
    username,
    setUsername,
    password,
    setPassword,
    session,
    setSession,
    userRole,
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
    login,
    connect,
    logout,
    refreshTask,
    selectEvaluation,
    submitFrame,
  };
};


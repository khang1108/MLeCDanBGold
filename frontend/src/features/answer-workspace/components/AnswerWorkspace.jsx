import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  resolveSubmissionAttempt,
  submitAnswerCandidate,
  submitAvsAnswers,
} from '../../../api/answerWorkspace';
import { resolveFrameAtTimestamp } from '../../../api/frames';
import ImageModal from '../../frames/components/ImageModal';
import AnswerCandidateDialog from './AnswerCandidateDialog';
import { useAnswerWorkspace } from '../contexts/AnswerWorkspaceContext';

const frameAnswer = (candidate) => `${candidate.video_id},${candidate.timestamp_ms},${candidate.timestamp_ms}`;
const EMPTY_CANDIDATES = [];

const submissionLocked = (context) => !context.connectedUserId
  || !context.isConnected
  || !context.workspace
  || Boolean(context.workspace.pending_submission)
  || Boolean(context.workspace.task_scope_mismatch)
  || Boolean(context.pendingAction);

const AnswerConfirmation = ({ title, answer, confirmLabel, onConfirm, onCancel, busy = false }) => (
  <div className="answer-dialog-backdrop">
    <section className="answer-confirmation-dialog" role="dialog" aria-modal="true" aria-label={title}>
      <header className="answer-dialog-header">
        <div>
          <span className="answer-dialog-eyebrow">Review before sending to DRES</span>
          <h2>{title}</h2>
        </div>
      </header>
      <p className="answer-confirmation-copy">The answer snapshot is fixed for this request.</p>
      <div className="answer-confirmation-value">Answer: {answer}</div>
      <footer className="answer-dialog-actions">
        <button type="button" className="btn-secondary" onClick={onCancel} disabled={busy}>Cancel</button>
        <button type="button" className="btn-primary" onClick={onConfirm} disabled={busy}>
          {busy ? 'Submitting…' : confirmLabel}
        </button>
      </footer>
    </section>
  </div>
);

/** Shared task-scoped editor and DRES submission surface for retrieval workspaces. */
const AnswerWorkspace = ({ isActive = true }) => {
  const context = useAnswerWorkspace();
  const {
    connectedUserId, workspace, candidates, isConnected, connectionError,
    pendingAction, addFrame, addText, updateFrame, updateText, remove,
    clear, clearAndSwitchTask, setAvsEnabled, refreshWorkspace,
  } = context;
  const [draft, setDraft] = useState(null);
  const [confirmation, setConfirmation] = useState(null);
  const [showClearConfirmation, setShowClearConfirmation] = useState(false);
  const [showSwitchConfirmation, setShowSwitchConfirmation] = useState(false);
  const [switchScope, setSwitchScope] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [viewer, setViewer] = useState(null);
  const [isResolvingViewer, setIsResolvingViewer] = useState(false);
  const viewerRequestRef = useRef(null);
  const rowRefs = useRef(new Map());

  const avsEnabled = Boolean(workspace?.avs_enabled);
  const locked = submissionLocked(context);
  const candidatesInOrder = candidates || EMPTY_CANDIDATES;
  const frameCandidates = useMemo(
    () => candidatesInOrder.filter((candidate) => candidate.kind === 'FRAME'),
    [candidatesInOrder],
  );
  const visibleCandidates = avsEnabled
    ? frameCandidates
    : candidatesInOrder;
  const eligibleAvsCandidates = frameCandidates.filter((candidate) => !candidate.submitted_at_ms
    && !candidate.submitted_by_user_id && !candidate.dres_status);
  const pendingSubmission = workspace?.pending_submission;
  const taskMismatch = Boolean(workspace?.task_scope_mismatch);

  useEffect(() => () => viewerRequestRef.current?.abort(), []);

  useEffect(() => {
    if (!isActive) return undefined;
    const focusCandidate = (event) => {
      const candidateId = event.detail?.candidateId;
      const row = rowRefs.current.get(candidateId);
      if (!row) return;
      row.scrollIntoView?.({ block: 'nearest' });
      row.focus();
    };
    window.addEventListener('answer-workspace-focus-candidate', focusCandidate);
    return () => window.removeEventListener('answer-workspace-focus-candidate', focusCandidate);
  }, [isActive]);

  const openEdit = useCallback((candidate) => {
    if (locked) return;
    setError('');
    setDraft(candidate.kind === 'FRAME'
      ? {
        kind: 'FRAME',
        isEditing: true,
        candidateId: candidate.candidate_id,
        expectedCandidateRevision: candidate.revision,
        initialValue: {
          kind: 'FRAME', videoId: candidate.video_id, timestampMs: candidate.timestamp_ms,
        },
      }
      : {
        kind: 'TEXT',
        isEditing: true,
        candidateId: candidate.candidate_id,
        expectedCandidateRevision: candidate.revision,
        initialValue: { kind: 'TEXT', text: candidate.text },
      });
  }, [locked]);

  const saveDraft = useCallback(async (value) => {
    if (locked) return;
    setBusy(true);
    setError('');
    try {
      if (draft.isEditing && draft.kind === 'FRAME') {
        await updateFrame({
          candidateId: draft.candidateId,
          expectedCandidateRevision: draft.expectedCandidateRevision,
          videoId: value.videoId,
          timestampMs: value.timestampMs,
        });
      } else if (draft.isEditing) {
        await updateText({
          candidateId: draft.candidateId,
          expectedCandidateRevision: draft.expectedCandidateRevision,
          text: value.text,
        });
      } else if (value.kind === 'FRAME') {
        await addFrame({ videoId: value.videoId, timestampMs: value.timestampMs });
      } else {
        await addText({ text: value.text });
      }
      setDraft(null);
    } catch (mutationError) {
      setError(mutationError.message || 'Could not update the shared answer workspace');
    } finally {
      setBusy(false);
    }
  }, [addFrame, addText, draft, locked, updateFrame, updateText]);

  const startSingleSubmission = useCallback((candidate) => {
    if (locked || candidate.submitted_at_ms || candidate.dres_status) return;
    const kind = candidate.kind === 'FRAME' ? 'KIS' : 'VQA';
    const answer = candidate.kind === 'FRAME' ? frameAnswer(candidate) : candidate.text;
    setError('');
    setConfirmation(Object.freeze({
      kind,
      userId: connectedUserId,
      taskScopeKey: workspace.task_scope_key,
      expectedWorkspaceRevision: workspace.revision,
      mode: avsEnabled,
      candidateId: candidate.candidate_id,
      expectedCandidateRevision: candidate.revision,
      answer,
    }));
  }, [avsEnabled, connectedUserId, locked, workspace]);

  const startAvsSubmission = useCallback(() => {
    if (locked || !avsEnabled || eligibleAvsCandidates.length === 0) return;
    const frozenCandidates = Object.freeze(eligibleAvsCandidates.map((candidate) => Object.freeze({
      candidateId: candidate.candidate_id,
      expectedRevision: candidate.revision,
      answer: frameAnswer(candidate),
    })));
    setError('');
    setConfirmation(Object.freeze({
      kind: 'AVS',
      userId: connectedUserId,
      taskScopeKey: workspace.task_scope_key,
      expectedWorkspaceRevision: workspace.revision,
      candidates: frozenCandidates,
      mode: avsEnabled,
      answer: frozenCandidates.map((candidate) => candidate.answer).join(' → '),
    }));
  }, [avsEnabled, connectedUserId, eligibleAvsCandidates, locked, workspace]);

  const confirmSubmission = useCallback(async () => {
    if (!confirmation || locked) return;
    setBusy(true);
    setError('');
    setStatusMessage('');
    try {
      let result;
      if (confirmation.kind === 'AVS') {
        result = await submitAvsAnswers({
          userId: confirmation.userId,
          taskScopeKey: confirmation.taskScopeKey,
          expectedWorkspaceRevision: confirmation.expectedWorkspaceRevision,
          candidates: confirmation.candidates.map(({ candidateId, expectedRevision }) => ({
            candidateId, expectedRevision,
          })),
        });
      } else {
        result = await submitAnswerCandidate({
          kind: confirmation.kind === 'KIS' ? 'FRAME' : 'TEXT',
          userId: confirmation.userId,
          taskScopeKey: confirmation.taskScopeKey,
          expectedWorkspaceRevision: confirmation.expectedWorkspaceRevision,
          candidateId: confirmation.candidateId,
          expectedCandidateRevision: confirmation.expectedCandidateRevision,
        });
      }
      setStatusMessage(result?.message || (result?.accepted
        ? 'DRES accepted the submission.'
        : `DRES status: ${result?.state || 'submitted'}.`));
      await refreshWorkspace?.();
      setConfirmation(null);
    } catch (submitError) {
      setError(submitError.message || 'The DRES submission result could not be confirmed.');
      try {
        await refreshWorkspace?.();
      } catch (refreshError) {
        setError(`${submitError.message || 'Submission failed'}; workspace refresh failed: ${refreshError.message}`);
      }
    } finally {
      setBusy(false);
    }
  }, [confirmation, locked, refreshWorkspace]);

  const handleResolveAttempt = useCallback(async (outcome) => {
    if (!pendingSubmission || pendingSubmission.state !== 'UNKNOWN' || !connectedUserId) return;
    setBusy(true);
    setError('');
    try {
      await resolveSubmissionAttempt({
        userId: connectedUserId,
        attemptId: pendingSubmission.attempt_id,
        outcome,
      });
      setStatusMessage(`DRES attempt marked ${outcome === 'accepted' ? 'accepted' : 'not accepted'}.`);
      await refreshWorkspace?.();
    } catch (resolveError) {
      setError(resolveError.message || 'Could not resolve the DRES submission attempt.');
    } finally {
      setBusy(false);
    }
  }, [connectedUserId, pendingSubmission, refreshWorkspace]);

  const handleOpenViewer = useCallback(async (candidate) => {
    viewerRequestRef.current?.abort();
    const controller = new AbortController();
    viewerRequestRef.current = controller;
    setIsResolvingViewer(true);
    setError('');
    try {
      const frame = await resolveFrameAtTimestamp({
        videoId: candidate.video_id,
        timestampMs: candidate.timestamp_ms,
        signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setViewer({ frame, initialTimestampMs: candidate.timestamp_ms });
      }
    } catch (viewerError) {
      if (viewerError.name !== 'AbortError') setError(viewerError.message || 'Could not open this answer in the viewer.');
    } finally {
      if (viewerRequestRef.current === controller) {
        viewerRequestRef.current = null;
        setIsResolvingViewer(false);
      }
    }
  }, []);

  const handleDelete = useCallback(async (candidate) => {
    if (locked) return;
    setError('');
    try {
      await remove({ candidateId: candidate.candidate_id, expectedCandidateRevision: candidate.revision });
    } catch (mutationError) {
      setError(mutationError.message || 'Could not remove this answer.');
    }
  }, [locked, remove]);

  const handleClear = useCallback(async () => {
    if (locked) return;
    setBusy(true);
    setError('');
    try {
      await clear();
      setShowClearConfirmation(false);
    } catch (mutationError) {
      setError(mutationError.message || 'Could not clear the answer workspace.');
    } finally {
      setBusy(false);
    }
  }, [clear, locked]);

  const handleSwitchTask = useCallback(async () => {
    if (!connectedUserId || !isConnected || pendingSubmission || pendingAction) return;
    setBusy(true);
    setError('');
    try {
      await clearAndSwitchTask(switchScope || undefined);
      setShowSwitchConfirmation(false);
      setSwitchScope(null);
    } catch (mutationError) {
      setError(mutationError.message || 'Could not switch the answer workspace task.');
    } finally {
      setBusy(false);
    }
  }, [clearAndSwitchTask, connectedUserId, isConnected, pendingAction, pendingSubmission, switchScope]);

  const [vqaText, setVqaText] = useState('');

  const beginManualFrame = () => {
    if (!locked) setDraft({ kind: 'FRAME', isEditing: false, initialValue: { kind: 'FRAME' } });
  };

  return (
    <>
      <section className="answer-workspace-panel" aria-label="Answer workspace">
        <header className="answer-workspace-header">
          <div className="answer-workspace-title-row">
            <div>
              <span className="answer-dialog-eyebrow">Shared with your team</span>
              <h3>Answer Workspace</h3>
            </div>
            <span className={`answer-connection-indicator ${isConnected ? 'is-connected' : 'is-disconnected'}`} aria-label="Workspace connection">
              <span className="answer-connection-dot" />{isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
          <div className="answer-workspace-toolbar">
            <span className="answer-count">{candidatesInOrder.length} {candidatesInOrder.length === 1 ? 'answer' : 'answers'}</span>
            <button
              type="button"
              className="answer-clear-button"
              disabled={locked || candidatesInOrder.length === 0}
              onClick={() => setShowClearConfirmation(true)}
            >
              Clear workspace
            </button>
          </div>
        </header>

        <div className="answer-mode-row">
          <div>
            <strong>AVS mode</strong>
            <p>{avsEnabled ? 'Submit an ordered frame sequence.' : 'Submit each KIS frame or VQA answer separately.'}</p>
          </div>
          <input
            type="checkbox"
            role="switch"
            aria-label="AVS mode"
            checked={avsEnabled}
            disabled={locked}
            onChange={(event) => setAvsEnabled(event.target.checked).catch((modeError) => setError(modeError.message))}
          />
        </div>

        {!avsEnabled && (
          <div className="answer-add-text-row">
            <label htmlFor="answer-vqa-input">VQA answer</label>
            <textarea
              id="answer-vqa-input"
              aria-label="VQA answer"
              rows={2}
              placeholder="Type an answer"
              disabled={locked}
              value={vqaText}
              onChange={(event) => setVqaText(event.target.value)}
            />
            <button
              type="button"
              className="btn-secondary answer-add-text-button"
              disabled={locked || !vqaText.trim()}
              onClick={() => {
                const text = vqaText.trim();
                if (!text) return;
                setVqaText('');
                setDraft({ kind: 'TEXT', isEditing: false, initialValue: { kind: 'TEXT', text } });
              }}
            >
              Add VQA answer
            </button>
          </div>
        )}

        {avsEnabled && (
          <button
            type="button"
            className="btn-secondary answer-add-frame-button"
            aria-label="Add frame to answer workspace"
            disabled={locked}
            onClick={beginManualFrame}
          >
            ＋ Add frame to answer workspace
          </button>
        )}

        {taskMismatch && (
          <div className="answer-task-mismatch" role="alert">
            <strong>Task changed on the DRES session</strong>
            <p>Saved answers belong to {workspace.task_name} ({workspace.evaluation_id}); the active task is {workspace.active_task_name} ({workspace.active_evaluation_id}).</p>
            <p>Clear the old task answers before switching the shared workspace.</p>
            <button
              type="button"
              className="btn-secondary"
              disabled={!connectedUserId || !isConnected || Boolean(pendingSubmission) || Boolean(pendingAction)}
              onClick={() => {
                setSwitchScope(Object.freeze({
                  expectedWorkspaceRevision: workspace.revision,
                  oldEvaluationId: workspace.evaluation_id,
                  oldTaskScopeKey: workspace.task_scope_key,
                  oldTaskName: workspace.task_name,
                  targetEvaluationId: workspace.active_evaluation_id,
                  targetTaskScopeKey: workspace.active_task_scope_key,
                  targetTaskName: workspace.active_task_name,
                }));
                setShowSwitchConfirmation(true);
              }}
            >
              Clear and switch task
            </button>
          </div>
        )}

        {pendingSubmission && (
          <div className="answer-pending-state" role="status">
            {pendingSubmission.state === 'UNKNOWN'
              ? 'DRES submission outcome is unknown. Resolve it before editing or submitting more answers.'
              : 'DRES submission is forwarding. Workspace actions are paused.'}
            {pendingSubmission.state === 'UNKNOWN' && (
              <div className="answer-resolution-actions">
                <button type="button" className="btn-secondary" disabled={busy} onClick={() => handleResolveAttempt('accepted')}>
                  Resolve as accepted
                </button>
                <button type="button" className="btn-secondary" disabled={busy} onClick={() => handleResolveAttempt('not_accepted')}>
                  Resolve as not accepted
                </button>
              </div>
            )}
          </div>
        )}

        {connectionError && <p className="answer-connection-error" role="status">{connectionError}</p>}
        {!connectedUserId && <p className="answer-empty-note">Connect a participant ID to edit and share answers.</p>}
        {!isConnected && connectedUserId && <p className="answer-empty-note">Reconnecting to the shared workspace…</p>}
        {error && <p className="answer-inline-error" role="alert">{error}</p>}
        {statusMessage && <p className="answer-success-message" role="status">{statusMessage}</p>}

        <div className="answer-candidate-list" aria-label="Shared answers">
          {visibleCandidates.length === 0 ? (
            <p className="answer-empty-note">No answers yet. Add a frame or type a VQA answer.</p>
          ) : visibleCandidates.map((candidate) => {
            const answer = candidate.kind === 'FRAME' ? frameAnswer(candidate) : candidate.text;
            const submitted = Boolean(candidate.submitted_at_ms || candidate.submitted_by_user_id || candidate.dres_status);
            return (
              <article
                className={`answer-candidate-row ${submitted ? 'is-submitted' : ''}`}
                key={candidate.candidate_id}
                ref={(node) => {
                  if (node) rowRefs.current.set(candidate.candidate_id, node);
                  else rowRefs.current.delete(candidate.candidate_id);
                }}
                tabIndex={-1}
              >
                <button
                  type="button"
                  className="answer-candidate-value"
                  aria-label={`Edit ${candidate.kind} answer ${answer}`}
                  disabled={locked}
                  onClick={() => openEdit(candidate)}
                >
                  <span>{answer}</span>
                </button>
                {submitted && <span className="answer-submitted-badge">Submitted</span>}
                <div className="answer-candidate-actions">
                  {candidate.kind === 'FRAME' && (
                    <button
                      type="button"
                      className="answer-icon-button"
                      aria-label="Open answer in viewer"
                      title="Open answer in viewer"
                      disabled={isResolvingViewer}
                      onClick={() => handleOpenViewer(candidate)}
                    >
                      ▶
                    </button>
                  )}
                  {!avsEnabled && (
                    <button
                      type="button"
                      className="answer-icon-button answer-submit-button"
                      aria-label={candidate.kind === 'FRAME' ? 'Submit KIS answer' : 'Submit VQA answer'}
                      title={candidate.kind === 'FRAME' ? 'Submit KIS answer' : 'Submit VQA answer'}
                      disabled={locked || submitted}
                      onClick={() => startSingleSubmission(candidate)}
                    >
                      ⇧
                    </button>
                  )}
                  <button
                    type="button"
                    className="answer-icon-button answer-delete-button"
                    aria-label="Delete answer"
                    title="Delete answer"
                    disabled={locked}
                    onClick={() => handleDelete(candidate)}
                  >
                    ×
                  </button>
                </div>
              </article>
            );
          })}
        </div>

        {avsEnabled && (
          <button
            type="button"
            className="btn-primary answer-submit-all-button"
            disabled={locked || eligibleAvsCandidates.length === 0}
            onClick={startAvsSubmission}
          >
            Submit all {eligibleAvsCandidates.length} AVS answers
          </button>
        )}
      </section>

      {draft && draft.kind !== 'quick-text' && (
        <AnswerCandidateDialog
          kind={draft.kind}
          initialValue={draft.initialValue}
          isEditing={draft.isEditing}
          onSave={saveDraft}
          onCancel={() => setDraft(null)}
        />
      )}
      {confirmation && (
        <AnswerConfirmation
          title={`Confirm ${confirmation.kind} submission`}
          answer={confirmation.answer}
          confirmLabel="Confirm and submit"
          onConfirm={confirmSubmission}
          onCancel={() => setConfirmation(null)}
          busy={busy}
        />
      )}
      {showClearConfirmation && (
        <AnswerConfirmation
          title="Confirm clear answers"
          answer={`Clear all ${candidatesInOrder.length} answers from this shared workspace?`}
          confirmLabel="Confirm and clear answers"
          onConfirm={handleClear}
          onCancel={() => setShowClearConfirmation(false)}
          busy={busy}
        />
      )}
      {showSwitchConfirmation && (
        <AnswerConfirmation
          title="Confirm task switch"
          answer={`Clear ${switchScope?.oldTaskName} answers and switch to ${switchScope?.targetTaskName}?`}
          confirmLabel="Confirm and clear old-task answers"
          onConfirm={handleSwitchTask}
          onCancel={() => {
            setShowSwitchConfirmation(false);
            setSwitchScope(null);
          }}
          busy={busy}
        />
      )}
      {viewer && (
        <ImageModal
          frame={viewer.frame}
          initialTimestampMs={viewer.initialTimestampMs}
          onClose={() => setViewer(null)}
        />
      )}
    </>
  );
};

export default AnswerWorkspace;

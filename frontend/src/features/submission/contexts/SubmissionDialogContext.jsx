/**
 * Own the ephemeral submission picker/editor workflow.
 *
 * Shared files remain exclusively in SubmissionContext. This provider keeps
 * only the open dialog, draft, pending submission intent, and conflict UI.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { markFramesSubmitted } from '../../../api/workspace';
import SubmissionFileModal from '../components/SubmissionFileModal';
import { useSubmission } from './SubmissionContext';

const SubmissionDialogContext = createContext(null);

const emptyDialog = {
  mode: null,
  fileName: null,
  pendingIntent: null,
  baseRevision: null,
  remoteConflict: null,
};

const appendLine = (content, line) => {
  if (!line) return content;
  return content.trim() ? `${content.trimEnd()}\n${line}` : line;
};

export const SubmissionDialogProvider = ({ children }) => {
  const {
    files,
    pendingFileNames,
    updateFile,
    validateFile,
    deleteFile,
  } = useSubmission();
  const [dialog, setDialog] = useState(emptyDialog);
  const [draft, setDraft] = useState('');
  const [baseContent, setBaseContent] = useState('');
  const [mutation, setMutation] = useState(null);
  const [error, setError] = useState(null);
  const mountedRef = useRef(true);
  const mutationRef = useRef(null);

  useEffect(() => {
    // StrictMode replays effect cleanup during development. Restore this flag
    // on every setup so a confirmed WebSocket mutation is not discarded.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const editorFile = useMemo(
    () => files.find((file) => file.name === dialog.fileName) || null,
    [dialog.fileName, files],
  );

  const resetDialog = useCallback(() => {
    if (mutationRef.current) return;
    setDialog(emptyDialog);
    setDraft('');
    setBaseContent('');
    setMutation(null);
    setError(null);
  }, []);

  const closeDialog = useCallback(() => {
    setDialog(emptyDialog);
    setDraft('');
    setBaseContent('');
    setError(null);
  }, []);

  const finishSuccessfulEditorOperation = useCallback(() => {
    // A confirmed save or validation should always dismiss the editor. This
    // deliberately bypasses resetDialog's in-flight guard after clearing the
    // ref, so both completion paths have identical close behavior.
    mutationRef.current = null;
    setMutation(null);
    closeDialog();
  }, [closeDialog]);

  const requestSubmission = useCallback((intent) => {
    setError(null);
    setDialog({
      ...emptyDialog,
      mode: 'picker',
      pendingIntent: intent || null,
    });
  }, []);

  const openEditor = useCallback((name) => {
    const file = files.find((item) => item.name === name);
    if (!file) return;
    setError(null);
    setDraft(file.content);
    setBaseContent(file.content);
    setDialog({
      ...emptyDialog,
      mode: 'editor',
      fileName: file.name,
      baseRevision: file.revision,
    });
  }, [files]);

  const selectFile = useCallback((name) => {
    const file = files.find((item) => item.name === name);
    if (!file) return;
    const pendingIntent = dialog.pendingIntent;
    setDraft(appendLine(file.content, pendingIntent?.line));
    setBaseContent(file.content);
    setDialog((current) => ({
      ...current,
      mode: 'editor',
      fileName: file.name,
      baseRevision: file.revision,
    }));
  }, [dialog.pendingIntent, files]);

  const markHistoryAfterCommit = useCallback((committedFile, pendingIntent) => {
    const history = pendingIntent?.history;
    if (!history) return;
    const patch = {
      queryId: history.queryId,
      submissionFileName: committedFile.name,
      submissionLine: pendingIntent.line,
      frameIds: history.frameIds,
    };

    // The file update is complete once its WebSocket confirmation arrives.
    // Keep the editor responsive while the independent history PATCH finishes.
    void Promise.resolve()
      .then(() => markFramesSubmitted(patch))
      .then(() => {
        window.dispatchEvent(new CustomEvent('hcmai:history-changed', { detail: patch }));
      })
      .catch((patchError) => {
        console.error('Could not record submission history', patchError);
      });
  }, []);

  const saveDraft = useCallback(async ({ validateAfter = false } = {}) => {
    if (!editorFile || mutationRef.current) return;
    const pendingIntent = dialog.pendingIntent;
    const operation = { kind: validateAfter ? 'save-and-validate' : 'save', name: editorFile.name };
    mutationRef.current = operation;
    setMutation(operation.kind);
    setError(null);
    try {
      const committed = await updateFile({
        name: editorFile.name,
        content: draft,
        expectedRevision: dialog.baseRevision,
      });
      if (!mountedRef.current) return;
      setBaseContent(committed.content);
      setDraft(committed.content);
      setDialog((current) => ({ ...current, baseRevision: committed.revision }));
      if (validateAfter) {
        setMutation('validate');
        const validated = await validateFile({
          name: committed.name,
          expectedRevision: committed.revision,
          isValidated: true,
        });
        if (!mountedRef.current) return;
        setBaseContent(validated.content);
        setDraft(validated.content);
        setDialog((current) => ({ ...current, baseRevision: validated.revision }));
      }
      finishSuccessfulEditorOperation();
      markHistoryAfterCommit(committed, pendingIntent);
    } catch (saveError) {
      mutationRef.current = null;
      setMutation(null);
      if (saveError?.latestFile) {
        setDialog((current) => ({ ...current, remoteConflict: saveError.latestFile }));
      }
      setError(saveError.message || 'Could not save submission file');
    }
  }, [dialog.baseRevision, dialog.pendingIntent, draft, editorFile, finishSuccessfulEditorOperation, markHistoryAfterCommit, updateFile, validateFile]);

  const handleValidate = useCallback(() => {
    if (!editorFile || (editorFile.is_validated && draft === baseContent)) return;
    if (draft !== baseContent) saveDraft({ validateAfter: true });
    else {
      mutationRef.current = { kind: 'validate', name: editorFile.name };
      setMutation('validate');
      validateFile({ name: editorFile.name, expectedRevision: editorFile.revision, isValidated: true })
        .then((committed) => {
          if (!mountedRef.current) return;
          setBaseContent(committed.content);
          setDraft(committed.content);
          finishSuccessfulEditorOperation();
        })
        .catch((validateError) => {
          mutationRef.current = null;
          setMutation(null);
          if (validateError?.latestFile) {
            setDialog((current) => ({ ...current, remoteConflict: validateError.latestFile }));
          }
          setError(validateError.message || 'Could not validate submission file');
        });
    }
  }, [baseContent, draft, editorFile, finishSuccessfulEditorOperation, saveDraft, validateFile]);

  const handleDelete = useCallback(() => {
    if (!editorFile || mutationRef.current) return;
    const operation = { kind: 'delete', name: editorFile.name };
    mutationRef.current = operation;
    setMutation('delete');
    deleteFile({ name: editorFile.name, expectedRevision: editorFile.revision })
      .then(() => {
        mutationRef.current = null;
        setMutation(null);
        resetDialog();
      })
      .catch((deleteError) => {
        mutationRef.current = null;
        setMutation(null);
        if (deleteError?.latestFile) {
          setDialog((current) => ({ ...current, remoteConflict: deleteError.latestFile }));
        }
        setError(deleteError.message || 'Could not delete submission file');
      });
  }, [deleteFile, editorFile, resetDialog]);

  useEffect(() => {
    if (dialog.mode !== 'editor' || !editorFile || mutationRef.current) return;
    if (editorFile.revision <= dialog.baseRevision) return;
    if (draft === baseContent) {
      setBaseContent(editorFile.content);
      setDraft(editorFile.content);
      setDialog((current) => ({ ...current, baseRevision: editorFile.revision }));
    } else {
      setDialog((current) => ({ ...current, remoteConflict: editorFile }));
    }
  }, [baseContent, dialog.baseRevision, dialog.mode, draft, editorFile]);

  const loadConflict = useCallback(() => {
    const conflict = dialog.remoteConflict;
    if (!conflict) return;
    setBaseContent(conflict.content);
    setDraft(conflict.content);
    setDialog((current) => ({
      ...current,
      baseRevision: conflict.revision,
      remoteConflict: null,
    }));
    setError(null);
  }, [dialog.remoteConflict]);

  const rebaseConflict = useCallback(() => {
    const conflict = dialog.remoteConflict;
    if (!conflict) return;
    setDialog((current) => ({ ...current, baseRevision: conflict.revision, remoteConflict: null }));
    setError(null);
  }, [dialog.remoteConflict]);

  const value = useMemo(() => ({
    requestSubmission,
    openEditor,
    mode: dialog.mode,
    files,
    editorFile,
    pendingIntent: dialog.pendingIntent,
    draft,
    isMutating: Boolean(mutation),
    mutation,
    pendingFileNames,
    error,
    remoteConflict: dialog.remoteConflict,
    selectFile,
    setDraft,
    saveDraft,
    handleValidate,
    handleDelete,
    resetDialog,
    loadConflict,
    rebaseConflict,
  }), [
    dialog.mode,
    dialog.pendingIntent,
    dialog.remoteConflict,
    draft,
    editorFile,
    error,
    files,
    handleDelete,
    handleValidate,
    loadConflict,
    mutation,
    openEditor,
    pendingFileNames,
    rebaseConflict,
    requestSubmission,
    resetDialog,
    saveDraft,
    selectFile,
  ]);

  return (
    <SubmissionDialogContext.Provider value={value}>
      {children}
      <SubmissionFileModal
        mode={dialog.mode}
        files={files}
        editorFile={editorFile}
        pendingLine={dialog.pendingIntent?.line}
        draft={draft}
        onDraftChange={setDraft}
        onSelectFile={selectFile}
        onSave={() => saveDraft()}
        onValidate={handleValidate}
        onDelete={handleDelete}
        onClose={closeDialog}
        isMutating={Boolean(mutation)}
        remoteConflict={dialog.remoteConflict}
        onLoadConflict={loadConflict}
        onRebaseConflict={rebaseConflict}
        error={error}
      />
    </SubmissionDialogContext.Provider>
  );
};

export const useSubmissionDialog = () => {
  const context = useContext(SubmissionDialogContext);
  if (context) return context;
  return {
    requestSubmission: () => {},
    openEditor: () => {},
  };
};

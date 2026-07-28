import React from 'react';
import ConversationToolbar from './ConversationToolbar';
import ConversationMessages from './ConversationMessages';
import ConversationDebug from './ConversationDebug';
import QueryInput from '../../search-controls/components/QueryInput';

// Assembles the narrow conversation column from focused presentation components.
const ConversationPanel = ({ toolbar, session, sessionError, isPending, debug, query, setQuery, onSubmit, canSubmit }) => <aside className="conversation-panel">
  <ConversationToolbar {...toolbar} sessionId={session?.session_id} isPending={isPending} />
  <ConversationMessages session={session} sessionError={sessionError} isPending={isPending} />
  <ConversationDebug session={session} {...debug} />
  <QueryInput query={query} setQuery={setQuery} onSubmit={onSubmit} isSubmitting={isPending} canSubmit={canSubmit} />
</aside>;

export default ConversationPanel;
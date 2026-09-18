import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

const mockSession = {
  connectedUserId: 'team-a',
  draftUserId: 'team-a',
  invalidateSession: jest.fn(),
  evaluations: [],
  selectedTask: {
    evaluationId: 'eval-1',
    evaluationName: 'VBS',
    taskName: 'AVS task',
    taskGroup: 'AVS',
    taskType: 'AVS',
    duration: 300,
  },
  setSelectedTask: jest.fn(),
};

jest.mock('./features/vbs/contexts/VbsSessionContext', () => ({
  VbsSessionProvider: ({ children }) => children,
  useVbsSession: () => mockSession,
}));

jest.mock('./features/search', () => ({
  SearchWorkspace: ({ workspaceMode, onToggleMode }) => (
    <div>
      <div>Unified search workspace (mode: {workspaceMode})</div>
      <button type="button" onClick={() => onToggleMode?.('KIS')}>Switch to KIS</button>
      <button type="button" onClick={() => onToggleMode?.('AVS')}>Switch to AVS</button>
    </div>
  ),
}));

jest.mock('./features/health', () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {} }),
}));

jest.mock('./features/event-trail', () => ({
  useEventTrail: () => ({
    session: null,
    pending: false,
    error: '',
    close: jest.fn(),
  }),
}));

jest.mock('./features/submission', () => ({
  SubmissionDialog: () => null,
  useDirectSubmission: () => ({ dialog: null, openError: '', opening: false }),
}));

test('routes an AVS task to the unified workspace with workspaceMode AVS', () => {
  render(<App />);

  expect(screen.getByText('Unified search workspace (mode: AVS)')).toBeTruthy();
});

test('allows toggling between KIS and AVS mode in unified workspace', () => {
  render(<App />);

  expect(screen.getByText('Unified search workspace (mode: AVS)')).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'Switch to KIS' }));
  expect(screen.getByText('Unified search workspace (mode: KIS)')).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'Switch to AVS' }));
  expect(screen.getByText('Unified search workspace (mode: AVS)')).toBeTruthy();
});

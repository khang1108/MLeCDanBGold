import React from 'react';
import { render, screen } from '@testing-library/react';
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

jest.mock('./features/avs', () => ({
  AvsWorkspace: () => <div>Dedicated AVS workspace</div>,
}));

jest.mock('./features/search', () => ({
  SearchWorkspace: () => <div>KIS search workspace</div>,
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

test('routes an AVS task to the dedicated workspace', () => {
  render(<App />);

  expect(screen.getByText('Dedicated AVS workspace')).toBeTruthy();
  expect(screen.queryByText('KIS search workspace')).toBeNull();
});

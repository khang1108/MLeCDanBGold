import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import {
  VbsSessionProvider,
  useVbsSession,
} from './VbsSessionContext';
import { connectVbsSession, getVbsSessionStatus, getVbsEvaluations } from '../../../api/vbs';

jest.mock('../../../api/vbs', () => ({
  connectVbsSession: jest.fn(),
  disconnectVbsSession: jest.fn(),
  getVbsSessionStatus: jest.fn(),
  getVbsEvaluations: jest.fn().mockResolvedValue([]),
}));

const Probe = () => {
  const session = useVbsSession();
  window.vbsSessionProbe = session;
  return (
    <div>
      <output data-testid="state">{session.connectionState}</output>
      <output data-testid="draft">{session.draftUserId}</output>
      <output data-testid="connected">{session.connectedUserId}</output>
      <button type="button" onClick={() => session.connect()}>Connect participant</button>
      <button type="button" onClick={() => session.invalidateSession('team-a')}>Invalidate cached session</button>
    </div>
  );
};

beforeEach(() => {
  localStorage.clear();
  jest.clearAllMocks();
  getVbsEvaluations.mockResolvedValue([]);
});

afterEach(() => { delete window.vbsSessionProbe; });

test('keeps the draft local until the backend handshake succeeds', async () => {
  let resolveConnect;
  connectVbsSession.mockReturnValueOnce(new Promise((resolve) => { resolveConnect = resolve; }));
  render(<VbsSessionProvider><Probe /></VbsSessionProvider>);
  act(() => window.vbsSessionProbe.setDraftUserId('team-a'));

  await act(async () => { window.vbsSessionProbe.connect(); });
  expect(screen.getByTestId('state').textContent).toBe('connecting');
  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  await act(async () => {
    resolveConnect({ user_id: 'team-a', connected: true });
  });

  expect(screen.getByTestId('connected').textContent).toBe('team-a');
  expect(localStorage.getItem('hcmai_user_id')).toBe('team-a');
  expect(localStorage.length).toBe(1);
});

test('revalidates a stored identity and reconnects if the cached backend session is absent', async () => {
  localStorage.setItem('hcmai_user_id', 'team-a');
  getVbsSessionStatus.mockResolvedValueOnce({ user_id: 'team-a', connected: false });
  connectVbsSession.mockResolvedValueOnce({ user_id: 'team-a', connected: true });

  render(<VbsSessionProvider><Probe /></VbsSessionProvider>);

  expect(screen.getByTestId('state').textContent).toBe('connecting');
  await waitFor(() => expect(screen.getByTestId('connected').textContent).toBe('team-a'));
  expect(getVbsSessionStatus).toHaveBeenCalledWith('team-a');
  expect(connectVbsSession).toHaveBeenCalledWith('team-a');
});

test('retains a previously handshaken identity for history when reconnect fails', async () => {
  localStorage.setItem('hcmai_user_id', 'team-a');
  getVbsSessionStatus.mockRejectedValueOnce(new Error('backend unavailable'));

  render(<VbsSessionProvider><Probe /></VbsSessionProvider>);

  await waitFor(() => expect(screen.getByTestId('state').textContent).toBe('error'));
  expect(screen.getByTestId('connected').textContent).toBe('');
  expect(screen.getByTestId('draft').textContent).toBe('team-a');
  expect(localStorage.getItem('hcmai_user_id')).toBe('team-a');
});

test('invalidates a rejected participant session and requires a later explicit connect', async () => {
  localStorage.setItem('hcmai_user_id', 'team-a');
  getVbsSessionStatus.mockResolvedValueOnce({ user_id: 'team-a', connected: true });
  render(<VbsSessionProvider><Probe /></VbsSessionProvider>);

  await waitFor(() => expect(screen.getByTestId('connected').textContent).toBe('team-a'));
  expect(connectVbsSession).not.toHaveBeenCalled();
  act(() => window.vbsSessionProbe.invalidateSession('team-a'));

  expect(screen.getByTestId('connected').textContent).toBe('');
  expect(screen.getByTestId('state').textContent).toBe('editing');
  expect(localStorage.getItem('hcmai_user_id')).toBeNull();
  expect(connectVbsSession).not.toHaveBeenCalled();
});

test('fetches evaluations and sets the default selected task upon connection', async () => {
  connectVbsSession.mockResolvedValueOnce({ user_id: 'team-a', connected: true });
  getVbsEvaluations.mockResolvedValueOnce([
    {
      id: 'eval-123',
      name: 'Evaluation Run 1',
      status: 'ACTIVE',
      taskTemplates: [
        { name: 'task-1', taskGroup: 'KIS', taskType: 'KIS', duration: 300 },
        { name: 'task-2', taskGroup: 'KIS', taskType: 'KIS', duration: 300 },
      ],
    },
  ]);

  render(<VbsSessionProvider><Probe /></VbsSessionProvider>);
  act(() => window.vbsSessionProbe.setDraftUserId('team-a'));
  await act(async () => { await window.vbsSessionProbe.connect(); });

  await waitFor(() => {
    expect(window.vbsSessionProbe.evaluations).toHaveLength(1);
    expect(window.vbsSessionProbe.selectedTask).toEqual({
      evaluationId: 'eval-123',
      evaluationName: 'Evaluation Run 1',
      taskName: 'task-1',
      taskGroup: 'KIS',
      taskType: 'KIS',
      duration: 300,
    });
  });
});


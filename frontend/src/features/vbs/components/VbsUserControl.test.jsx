import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import VbsUserControl from './VbsUserControl';
import { VbsSessionProvider } from '../contexts/VbsSessionContext';
import { connectVbsSession, disconnectVbsSession } from '../../../api/vbs';

jest.mock('../../../api/vbs', () => ({
  connectVbsSession: jest.fn(),
  disconnectVbsSession: jest.fn(),
  getVbsSessionStatus: jest.fn(),
}));

beforeEach(() => {
  localStorage.clear();
  jest.clearAllMocks();
  connectVbsSession.mockResolvedValue({ user_id: 'team-a', connected: true });
  disconnectVbsSession.mockResolvedValue({ user_id: 'team-a', connected: false });
});

test('OK disconnects and returns keyboard focus to the unlocked User ID input', async () => {
  const inputRef = React.createRef();
  render(
    <VbsSessionProvider>
      <VbsUserControl inputRef={inputRef} />
    </VbsSessionProvider>,
  );

  const userId = screen.getByLabelText('User ID');
  fireEvent.change(userId, { target: { value: 'team-a' } });
  fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
  await screen.findByRole('button', { name: 'OK' });

  fireEvent.click(screen.getByRole('button', { name: 'OK' }));
  await waitFor(() => expect(userId.disabled).toBe(false));
  await waitFor(() => expect(document.activeElement).toBe(userId));
  expect(disconnectVbsSession).toHaveBeenCalledWith('team-a');
});

test('shows DRES connection and result-log status without exposing session tokens', async () => {
  render(
    <VbsSessionProvider>
      <VbsUserControl />
    </VbsSessionProvider>,
  );

  const status = screen.getByRole('status', { name: 'DRES status' });
  expect(status.textContent).toBe('Disconnected');
  fireEvent.change(screen.getByLabelText('User ID'), { target: { value: 'team-a' } });
  fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
  expect(status.textContent).toBe('Connecting');
  await screen.findByRole('button', { name: 'OK' });
  expect(status.textContent).toBe('Connected');

  act(() => window.dispatchEvent(new CustomEvent('hcmai:dres-log-status', {
    detail: {
      userId: 'team-a',
      status: 'sent',
      session: 'private-session-token',
      evaluationToken: 'private-evaluation-token',
    },
  })));
  expect(status.textContent).toBe('Log sent');
  expect(document.body.textContent).not.toContain('private-session-token');
  expect(document.body.textContent).not.toContain('private-evaluation-token');

  act(() => window.dispatchEvent(new CustomEvent('hcmai:dres-log-status', {
    detail: { userId: 'team-b', status: 'failed' },
  })));
  expect(status.textContent).toBe('Log sent');
  act(() => window.dispatchEvent(new CustomEvent('hcmai:dres-log-status', {
    detail: { userId: 'team-a', status: 'failed' },
  })));
  expect(status.textContent).toBe('Last log failed');

  fireEvent.click(screen.getByRole('button', { name: 'OK' }));
  await waitFor(() => expect(status.textContent).toBe('Disconnected'));
});

import React from 'react';
import { render, screen } from '@testing-library/react';
import AppHeader from './AppHeader';
import { VbsSessionProvider } from '../vbs/contexts/VbsSessionContext';

test('uses the VBS 2027 competition title and no tab buttons', () => {
  render(
    <VbsSessionProvider>
      <AppHeader
        isHealthy
        healthData={{}}
        onOpenDocs={jest.fn()}
      />
    </VbsSessionProvider>,
  );

  expect(screen.getByRole('heading', { name: 'VBS 2027 Video Retrieval' })).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Query' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Workspace' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Database' })).toBeNull();
});


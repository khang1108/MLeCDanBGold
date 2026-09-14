import React from 'react';
import { render, screen } from '@testing-library/react';
import AppHeader from './AppHeader';
import { VbsSessionProvider } from '../vbs/contexts/VbsSessionContext';

test('uses the VBS 2027 competition title', () => {
  render(
    <VbsSessionProvider>
      <AppHeader
        isHealthy
        healthData={{}}
        vimMode="NORMAL"
        onToggleVimMode={jest.fn()}
        onOpenDocs={jest.fn()}
        activePage="query"
        onSelectPage={jest.fn()}
      />
    </VbsSessionProvider>,
  );

  expect(screen.getByRole('heading', { name: 'VBS 2027 Video Retrieval' })).toBeTruthy();
});

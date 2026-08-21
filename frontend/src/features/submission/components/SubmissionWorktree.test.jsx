import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SubmissionWorktree from './SubmissionWorktree';
import { SubmissionProvider } from '../contexts/SubmissionContext';
import * as searchApi from '../../../api/search';

jest.mock('../../../api/search');

describe('SubmissionWorktree component', () => {
  beforeEach(() => {
    window.localStorage.clear();
    jest.clearAllMocks();
  });

  test('renders upload state when no submission files are loaded', () => {
    render(
      <SubmissionProvider>
        <SubmissionWorktree />
      </SubmissionProvider>
    );

    expect(screen.getByText('Submission Files')).toBeTruthy();
    expect(screen.getByText('No Query Files')).toBeTruthy();
    expect(screen.getByRole('button', { name: /upload query files/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /select folder/i })).toBeTruthy();
  });

  test('handles uploading query files and displays .csv files in tree', async () => {
    searchApi.uploadQueryFiles.mockResolvedValueOnce([
      { id: 'query_1.csv', name: 'query_1.csv', originalName: 'query_1.txt', content: '' },
      { id: 'query_2.csv', name: 'query_2.csv', originalName: 'query_2.txt', content: '' },
    ]);

    render(
      <SubmissionProvider>
        <SubmissionWorktree />
      </SubmissionProvider>
    );

    const fileInput = screen.getByTestId('query-file-input');
    const mockFile1 = new File(['sample query 1'], 'query_1.txt', { type: 'text/plain' });
    const mockFile2 = new File(['sample query 2'], 'query_2.txt', { type: 'text/plain' });

    fireEvent.change(fileInput, { target: { files: [mockFile1, mockFile2] } });

    await waitFor(() => {
      expect(screen.getByText('query_1.csv')).toBeTruthy();
      expect(screen.getByText('query_2.csv')).toBeTruthy();
    });

    expect(screen.getByText('submissions/')).toBeTruthy();
    expect(screen.getByRole('button', { name: /submit to backend \(2\)/i })).toBeTruthy();
  });

  test('submits files to backend', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'query_1.csv', name: 'query_1.csv', content: 'L01_V001,100' },
      ])
    );
    searchApi.submitCsvFiles.mockResolvedValueOnce({ status: 'ok' });

    render(
      <SubmissionProvider>
        <SubmissionWorktree />
      </SubmissionProvider>
    );

    expect(screen.getByText('query_1.csv')).toBeTruthy();
    const submitBtn = screen.getByRole('button', { name: /submit to backend/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(searchApi.submitCsvFiles).toHaveBeenCalledWith([
        { name: 'query_1.csv', content: 'L01_V001,100' },
      ]);
    });
  });
});

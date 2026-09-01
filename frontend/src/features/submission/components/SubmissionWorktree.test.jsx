import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SubmissionWorktree from './SubmissionWorktree';
import { SubmissionProvider } from '../contexts/SubmissionContext';
import * as submissionArchive from '../submissionArchive';

const renderWorktree = (props = {}) => render(
  <SubmissionProvider>
    <SubmissionWorktree {...props} />
  </SubmissionProvider>,
);

describe('SubmissionWorktree component', () => {
  beforeEach(() => {
    window.localStorage.clear();
    jest.clearAllMocks();
  });

  test('renders upload state when no submission files are loaded', () => {
    renderWorktree();

    expect(screen.getByText('Submission Files')).toBeTruthy();
    expect(screen.getByText('No Query Files')).toBeTruthy();
    expect(screen.getByRole('button', { name: /upload query files/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /select folder/i })).toBeTruthy();
  });

  test('creates local CSV targets from uploaded query files', async () => {
    renderWorktree();
    const fileInput = screen.getByTestId('query-file-input');
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(['sample query 1'], 'query_1.txt', { type: 'text/plain' }),
          new File(['sample query 2'], 'query_2.txt', { type: 'text/plain' }),
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText('query_1.csv')).toBeTruthy();
      expect(screen.getByText('query_2.csv')).toBeTruthy();
    });

    expect(screen.getByText('submissions/')).toBeTruthy();
    expect(screen.getByRole('button', { name: /download csv zip \(0\)/i }).disabled).toBe(true);
  });

  test('uses the browser file picker API without a backend request', async () => {
    const originalPicker = window.showOpenFilePicker;
    const pickedFile = new File(['query'], 'query.txt', { type: 'text/plain' });
    window.showOpenFilePicker = jest.fn().mockResolvedValue([
      { getFile: jest.fn().mockResolvedValue(pickedFile) },
    ]);

    try {
      renderWorktree();
      fireEvent.click(screen.getByRole('button', { name: /upload query files/i }));

      expect(await screen.findByText('query.csv')).toBeTruthy();
      expect(window.showOpenFilePicker).toHaveBeenCalledTimes(1);
    } finally {
      if (originalPicker) window.showOpenFilePicker = originalPicker;
      else delete window.showOpenFilePicker;
    }
  });

  test('reads nested files selected through the browser directory picker', async () => {
    const originalPicker = window.showDirectoryPicker;
    const pickedFile = new File(['query'], 'nested-query.txt', { type: 'text/plain' });
    const directoryHandle = {
      values: async function* values() {
        yield { kind: 'file', getFile: jest.fn().mockResolvedValue(pickedFile) };
      },
    };
    window.showDirectoryPicker = jest.fn().mockResolvedValue(directoryHandle);

    try {
      renderWorktree();
      fireEvent.click(screen.getByRole('button', { name: /select folder/i }));

      expect(await screen.findByText('nested-query.csv')).toBeTruthy();
      expect(window.showDirectoryPicker).toHaveBeenCalledWith({ mode: 'read' });
    } finally {
      if (originalPicker) window.showDirectoryPicker = originalPicker;
      else delete window.showDirectoryPicker;
    }
  });

  test('submit request opens the picker, appends the BTC row, and opens the editor', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'query_1.csv', name: 'query_1.csv', content: '' },
        { id: 'other.csv', name: 'other.csv', content: '' },
      ]),
    );

    renderWorktree({
      submissionRequest: { line: 'L21_V001,17794', source: 'KIS/TRAKE frame' },
    });

    expect(await screen.findByRole('dialog', { name: /choose a csv file/i })).toBeTruthy();
    const search = screen.getByRole('textbox', { name: /search submission files/i });
    fireEvent.change(search, { target: { value: 'query_1' } });
    fireEvent.keyDown(search, { key: 'Enter', code: 'Enter' });

    const editor = await screen.findByRole('textbox', { name: /edit query_1\.csv content/i });
    expect(editor.value).toBe('L21_V001,17794');
  });

  test('keeps the highlighted picker file visible while navigating with arrows', async () => {
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const scrollIntoView = jest.fn();
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'one.csv', name: 'one.csv', content: '' },
        { id: 'two.csv', name: 'two.csv', content: '' },
      ]),
    );

    try {
      renderWorktree({
        submissionRequest: { line: 'L21_V001,17794', source: 'KIS/TRAKE frame' },
      });
      const search = await screen.findByRole('textbox', { name: /search submission files/i });
      fireEvent.keyDown(search, { key: 'ArrowDown', code: 'ArrowDown' });

      await waitFor(() => expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' }));
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
    }
  });

  test('Enter saves and closes the editor, while double-click opens a file', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'query_1.csv', name: 'query_1.csv', content: 'L21_V001,100' },
      ]),
    );

    renderWorktree();
    fireEvent.doubleClick(screen.getByText('query_1.csv'));

    const editor = await screen.findByRole('textbox', { name: /edit query_1\.csv content/i });
    fireEvent.change(editor, { target: { value: 'L21_V001,200' } });
    fireEvent.keyDown(editor, { key: 'Enter', code: 'Enter' });

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(JSON.parse(window.localStorage.getItem('hcmai.submission.files'))[0].content)
      .toBe('L21_V001,200');
  });

  test('Escape closes an open file editor without saving', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'query_1.csv', name: 'query_1.csv', content: 'L21_V001,100' },
      ]),
    );

    renderWorktree();
    fireEvent.doubleClick(screen.getByText('query_1.csv'));
    const editor = await screen.findByRole('textbox', { name: /edit query_1\.csv content/i });
    fireEvent.change(editor, { target: { value: 'L21_V001,999' } });
    fireEvent.keyDown(editor, { key: 'Escape', code: 'Escape' });

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(JSON.parse(window.localStorage.getItem('hcmai.submission.files'))[0].content)
      .toBe('L21_V001,100');
  });

  test('Escape closes the file picker without bubbling to the parent modal', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'query_1.csv', name: 'query_1.csv', content: '' },
      ]),
    );
    const parentEscapeHandler = jest.fn();
    window.addEventListener('keydown', parentEscapeHandler);

    try {
      renderWorktree({
        submissionRequest: { line: 'L21_V001,17794', source: 'Frame inspector' },
      });
      const search = await screen.findByRole('textbox', { name: /search submission files/i });

      fireEvent.keyDown(search, { key: 'Escape', code: 'Escape' });

      await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
      expect(parentEscapeHandler).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('keydown', parentEscapeHandler);
    }
  });

  test('downloads only non-empty CSV files as one ZIP', async () => {
    window.localStorage.setItem(
      'hcmai.submission.files',
      JSON.stringify([
        { id: 'filled.csv', name: 'filled.csv', content: 'L21_V001,100' },
        { id: 'empty.csv', name: 'empty.csv', content: '  \n' },
      ]),
    );
    jest.spyOn(submissionArchive, 'downloadCsvArchive').mockReturnValue(true);

    renderWorktree();
    fireEvent.click(screen.getByRole('button', { name: /download csv zip \(1\)/i }));

    expect(submissionArchive.downloadCsvArchive).toHaveBeenCalledWith([
      { id: 'filled.csv', name: 'filled.csv', content: 'L21_V001,100' },
    ]);
    expect((await screen.findByRole('status')).textContent).toMatch(/submissions\.zip/i);
  });
});

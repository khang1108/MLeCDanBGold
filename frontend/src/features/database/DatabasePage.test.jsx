import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import DatabasePage from './DatabasePage';
import * as dbApi from '../../api/database';

jest.mock('../../api/database');

describe('DatabasePage Component', () => {
  const mockTables = {
    tables: [
      {
        name: 'query_history',
        row_count: 2,
        columns: [
          { name: 'query_id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'query_text', type: 'TEXT', nullable: false, primary_key: false },
        ],
      },
      {
        name: 'submission_files',
        row_count: 1,
        columns: [
          { name: 'name', type: 'TEXT', nullable: false, primary_key: true },
        ],
      },
    ],
  };

  const mockRows = {
    table: 'query_history',
    page: 1,
    page_size: 25,
    total_rows: 2,
    total_pages: 1,
    rows: [
      { query_id: 'q-1', query_text: 'red bus' },
      { query_id: 'q-2', query_text: 'blue car' },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('fetches tables and default rows when activated', async () => {
    dbApi.fetchDatabaseTables.mockResolvedValueOnce(mockTables);
    dbApi.fetchDatabaseRows.mockResolvedValueOnce(mockRows);

    render(<DatabasePage isActive={true} />);

    expect(screen.getByText('Loading database tables...')).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText('query_history (2 rows)')).toBeTruthy();
      expect(screen.getByText('submission_files (1 rows)')).toBeTruthy();
    });

    await waitFor(() => {
      expect(screen.getByText('red bus')).toBeTruthy();
      expect(screen.getByText('blue car')).toBeTruthy();
    });

    expect(dbApi.fetchDatabaseTables).toHaveBeenCalledTimes(1);
    expect(dbApi.fetchDatabaseRows).toHaveBeenCalledWith('query_history', { page: 1, pageSize: 25 });
  });

  test('switches table when clicking another tab', async () => {
    dbApi.fetchDatabaseTables.mockResolvedValueOnce(mockTables);
    dbApi.fetchDatabaseRows
      .mockResolvedValueOnce(mockRows)
      .mockResolvedValueOnce({
        table: 'submission_files',
        page: 1,
        page_size: 25,
        total_rows: 1,
        total_pages: 1,
        rows: [{ name: 'results.csv' }],
      });

    render(<DatabasePage isActive={true} />);

    await waitFor(() => {
      expect(screen.getByText('submission_files (1 rows)')).toBeTruthy();
    });

    fireEvent.click(screen.getByText('submission_files (1 rows)'));

    await waitFor(() => {
      expect(screen.getByText('results.csv')).toBeTruthy();
    });

    expect(dbApi.fetchDatabaseRows).toHaveBeenLastCalledWith('submission_files', { page: 1, pageSize: 25 });
  });

  test('handles database error gracefully', async () => {
    dbApi.fetchDatabaseTables.mockRejectedValueOnce(new Error('Workspace database is not configured'));

    render(<DatabasePage isActive={true} />);

    await waitFor(() => {
      expect(screen.getByText(/Workspace database is not configured/)).toBeTruthy();
    });
  });
});

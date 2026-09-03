import { fetchDatabaseTables, fetchDatabaseRows, executeDatabaseQuery } from './database';

jest.mock('./client', () => {
  const actual = jest.requireActual('./client');
  return { ...actual, requestJson: jest.fn() };
});

import { requestJson } from './client';

beforeEach(() => {
  requestJson.mockReset();
});

test('fetchDatabaseTables queries /api/v1/database/tables', async () => {
  const mockPayload = {
    tables: [
      {
        name: 'query_history',
        row_count: 5,
        columns: [{ name: 'query_id', type: 'TEXT', nullable: false, primary_key: true }],
      },
    ],
  };
  requestJson.mockResolvedValueOnce(mockPayload);

  const result = await fetchDatabaseTables();
  expect(requestJson).toHaveBeenCalledWith('/api/v1/database/tables', { signal: undefined });
  expect(result).toEqual(mockPayload);
});

test('fetchDatabaseRows formats query parameters and table name', async () => {
  const mockRows = {
    table: 'query_history',
    page: 2,
    page_size: 10,
    total_rows: 15,
    total_pages: 2,
    rows: [{ query_id: 'q-1', query_text: 'sample query' }],
  };
  requestJson.mockResolvedValueOnce(mockRows);

  const result = await fetchDatabaseRows('query_history', { page: 2, pageSize: 10 });
  expect(requestJson).toHaveBeenCalledWith(
    '/api/v1/database/tables/query_history/rows?page=2&page_size=10',
    { signal: undefined },
  );
  expect(result).toEqual(mockRows);
});

test('fetchDatabaseRows validates required tableName', async () => {
  await expect(fetchDatabaseRows('')).rejects.toThrow('tableName is required');
  expect(requestJson).not.toHaveBeenCalled();
});

test('executeDatabaseQuery sends POST /api/v1/database/execute', async () => {
  const mockResponse = {
    query: 'SELECT 1',
    columns: ['1'],
    rows: [{ '1': 1 }],
    rows_affected: 0,
    execution_time_ms: 1.5,
    is_mutation: false,
  };
  requestJson.mockResolvedValueOnce(mockResponse);

  const result = await executeDatabaseQuery('SELECT 1', { maxRows: 50 });
  expect(requestJson).toHaveBeenCalledWith('/api/v1/database/execute', {
    method: 'POST',
    body: { query: 'SELECT 1', max_rows: 50 },
    signal: undefined,
  });
  expect(result).toEqual(mockResponse);
});

test('executeDatabaseQuery rejects blank query', async () => {
  await expect(executeDatabaseQuery('   ')).rejects.toThrow('query is required');
  expect(requestJson).not.toHaveBeenCalled();
});


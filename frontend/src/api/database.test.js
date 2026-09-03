import { fetchDatabaseTables, fetchDatabaseRows } from './database';

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

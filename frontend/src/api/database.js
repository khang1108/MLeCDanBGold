/** Safe read-only HTTP client wrappers for the workspace SQLite database. */
import { requestJson } from './client';

/** Fetch available allowlisted SQLite tables and their column schemas. */
export const fetchDatabaseTables = async ({ signal } = {}) => {
  return requestJson('/api/v1/database/tables', { signal });
};

/** Fetch a paginated slice of raw SQLite row records from a specified table. */
export const fetchDatabaseRows = async (tableName, { page = 1, pageSize = 25, signal } = {}) => {
  if (!tableName || typeof tableName !== 'string' || !tableName.trim()) {
    throw new Error('tableName is required and must be a non-empty string');
  }

  const searchParams = new URLSearchParams({
    page: String(Math.max(1, Math.floor(page))),
    page_size: String(Math.min(100, Math.max(1, Math.floor(pageSize)))),
  });

  const path = `/api/v1/database/tables/${encodeURIComponent(tableName.trim())}/rows?${searchParams.toString()}`;
  return requestJson(path, { signal });
};

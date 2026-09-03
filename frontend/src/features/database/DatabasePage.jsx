/** SQLite Database Browser page for inspecting workspace tables and records. */
import React, { useCallback, useEffect, useState } from 'react';
import { fetchDatabaseTables, fetchDatabaseRows, executeDatabaseQuery } from '../../api/database';
import TableSelector from './components/TableSelector';
import TableMetadata from './components/TableMetadata';
import DataGrid from './components/DataGrid';
import PaginationControls from './components/PaginationControls';
import SqlQueryEditor from './components/SqlQueryEditor';

export const DatabasePage = ({ isActive = false }) => {
  const [tables, setTables] = useState([]);
  const [selectedTableName, setSelectedTableName] = useState('');
  const [rowsData, setRowsData] = useState(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [isLoadingTables, setIsLoadingTables] = useState(false);
  const [isLoadingRows, setIsLoadingRows] = useState(false);
  const [error, setError] = useState(null);

  // SQL Query execution state
  const [customQueryResult, setCustomQueryResult] = useState(null);
  const [isExecutingSql, setIsExecutingSql] = useState(false);
  const [sqlStats, setSqlStats] = useState(null);
  const [sqlError, setSqlError] = useState(null);

  const loadTables = useCallback(async () => {
    setIsLoadingTables(true);
    setError(null);
    try {
      const response = await fetchDatabaseTables();
      const loadedTables = response?.tables || [];
      setTables(loadedTables);
      if (loadedTables.length > 0) {
        setSelectedTableName((current) => current || loadedTables[0].name);
      }
    } catch (err) {
      setError(err?.message || 'Failed to load SQLite tables');
    } finally {
      setIsLoadingTables(false);
    }
  }, []);

  const loadRows = useCallback(async (tableName, targetPage, targetPageSize) => {
    if (!tableName) return;
    setIsLoadingRows(true);
    setError(null);
    try {
      const response = await fetchDatabaseRows(tableName, {
        page: targetPage,
        pageSize: targetPageSize,
      });
      setRowsData(response);
    } catch (err) {
      setError(err?.message || `Failed to load rows for ${tableName}`);
    } finally {
      setIsLoadingRows(false);
    }
  }, []);

  useEffect(() => {
    if (isActive && tables.length === 0 && !isLoadingTables && !error) {
      loadTables();
    }
  }, [isActive, tables.length, isLoadingTables, error, loadTables]);

  useEffect(() => {
    if (isActive && selectedTableName && !customQueryResult) {
      loadRows(selectedTableName, page, pageSize);
    }
  }, [isActive, selectedTableName, page, pageSize, customQueryResult, loadRows]);

  const handleSelectTable = (tableName) => {
    setCustomQueryResult(null);
    if (tableName === selectedTableName) return;
    setSelectedTableName(tableName);
    setPage(1);
  };

  const handleChangePage = (newPage) => {
    setPage(newPage);
  };

  const handleChangePageSize = (newPageSize) => {
    setPageSize(newPageSize);
    setPage(1);
  };

  const handleExecuteSql = async (queryText) => {
    setIsExecutingSql(true);
    setSqlError(null);
    try {
      const result = await executeDatabaseQuery(queryText);
      setSqlStats({
        execution_time_ms: result.execution_time_ms,
        rows_count: result.rows?.length || 0,
        rows_affected: result.rows_affected || 0,
        is_mutation: result.is_mutation,
      });

      if (result.is_mutation) {
        setCustomQueryResult(null);
        await loadTables();
        if (selectedTableName) {
          await loadRows(selectedTableName, page, pageSize);
        }
      } else {
        setCustomQueryResult(result);
      }
    } catch (err) {
      setSqlError(err?.message || 'Failed to execute SQL query');
    } finally {
      setIsExecutingSql(false);
    }
  };

  const handleBackToTable = () => {
    setCustomQueryResult(null);
  };

  const selectedTable = tables.find((t) => t.name === selectedTableName) || null;

  return (
    <div className="database-page">
      <header className="db-header">
        <h2 className="db-title">SQLite Database Browser</h2>
        <p className="db-subtitle">
          Inspect allowlisted workspace database tables, column schemas, and execute raw SQL queries.
        </p>
      </header>

      <SqlQueryEditor
        onExecute={handleExecuteSql}
        isExecuting={isExecutingSql}
        stats={sqlStats}
        error={sqlError}
      />

      {error && (
        <div className="db-error-state" role="alert">
          {error}
        </div>
      )}

      {isLoadingTables ? (
        <div className="db-loading-state">Loading database tables...</div>
      ) : customQueryResult ? (
        <div className="db-custom-results">
          <div className="db-custom-header">
            <span className="db-custom-title">
              Query Results: <code>{customQueryResult.query}</code>
            </span>
            <button
              type="button"
              className="db-btn"
              onClick={handleBackToTable}
            >
              Back to Table View
            </button>
          </div>
          <DataGrid
            columns={customQueryResult.columns.map((name) => ({ name }))}
            rows={customQueryResult.rows}
            isLoading={false}
          />
        </div>
      ) : tables.length > 0 ? (
        <>
          <TableSelector
            tables={tables}
            selectedTableName={selectedTableName}
            onSelectTable={handleSelectTable}
            disabled={isLoadingRows}
          />

          {selectedTable && <TableMetadata table={selectedTable} />}

          <DataGrid
            columns={selectedTable?.columns || []}
            rows={rowsData?.rows || []}
            isLoading={isLoadingRows}
          />

          {rowsData && (
            <PaginationControls
              page={rowsData.page}
              pageSize={rowsData.page_size}
              totalRows={rowsData.total_rows}
              totalPages={rowsData.total_pages}
              onChangePage={handleChangePage}
              onChangePageSize={handleChangePageSize}
              disabled={isLoadingRows}
            />
          )}
        </>
      ) : !error ? (
        <div className="db-empty-state">No database tables available.</div>
      ) : null}
    </div>
  );
};

export default DatabasePage;

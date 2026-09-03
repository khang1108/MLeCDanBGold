/** SQLite Database Browser page for inspecting workspace tables and records. */
import React, { useCallback, useEffect, useState } from 'react';
import { fetchDatabaseTables, fetchDatabaseRows } from '../../api/database';
import TableSelector from './components/TableSelector';
import TableMetadata from './components/TableMetadata';
import DataGrid from './components/DataGrid';
import PaginationControls from './components/PaginationControls';

export const DatabasePage = ({ isActive = false }) => {
  const [tables, setTables] = useState([]);
  const [selectedTableName, setSelectedTableName] = useState('');
  const [rowsData, setRowsData] = useState(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [isLoadingTables, setIsLoadingTables] = useState(false);
  const [isLoadingRows, setIsLoadingRows] = useState(false);
  const [error, setError] = useState(null);

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
    if (isActive && selectedTableName) {
      loadRows(selectedTableName, page, pageSize);
    }
  }, [isActive, selectedTableName, page, pageSize, loadRows]);

  const handleSelectTable = (tableName) => {
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

  const selectedTable = tables.find((t) => t.name === selectedTableName) || null;

  return (
    <div className="database-page">
      <header className="db-header">
        <h2 className="db-title">SQLite Database Browser</h2>
        <p className="db-subtitle">
          Inspect allowlisted workspace database tables, column schemas, and paginated records.
        </p>
      </header>

      {error && (
        <div className="db-error-state" role="alert">
          {error}
        </div>
      )}

      {isLoadingTables ? (
        <div className="db-loading-state">Loading database tables...</div>
      ) : (
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
      )}
    </div>
  );
};

export default DatabasePage;

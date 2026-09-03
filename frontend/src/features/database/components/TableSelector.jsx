import React from 'react';

export const TableSelector = ({ tables = [], selectedTableName, onSelectTable, disabled = false }) => {
  if (!tables.length) return null;

  return (
    <div className="db-table-selector" role="tablist" aria-label="Database tables">
      {tables.map((table) => {
        const isSelected = table.name === selectedTableName;
        return (
          <button
            key={table.name}
            type="button"
            role="tab"
            aria-selected={isSelected}
            disabled={disabled}
            className={`db-table-tab ${isSelected ? 'active' : ''}`}
            onClick={() => onSelectTable(table.name)}
          >
            {table.name} ({table.row_count} rows)
          </button>
        );
      })}
    </div>
  );
};

export default TableSelector;

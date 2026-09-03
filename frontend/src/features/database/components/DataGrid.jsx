import React from 'react';

const formatCellValue = (value) => {
  if (value === null || value === undefined) {
    return <span className="db-cell-null">null</span>;
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
};

export const DataGrid = ({ columns = [], rows = [], isLoading = false }) => {
  if (isLoading) {
    return <div className="db-loading-state">Loading table data...</div>;
  }

  const columnKeys = columns.length > 0
    ? columns.map((col) => col.name)
    : (rows.length > 0 ? Object.keys(rows[0]) : []);

  if (columnKeys.length === 0 || rows.length === 0) {
    return <div className="db-empty-state">No rows found in this table.</div>;
  }

  return (
    <div className="db-grid-container">
      <table className="db-grid-table">
        <thead>
          <tr>
            {columnKeys.map((colName) => (
              <th key={colName} scope="col">
                {colName}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columnKeys.map((colName) => (
                <td key={colName}>{formatCellValue(row[colName])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DataGrid;

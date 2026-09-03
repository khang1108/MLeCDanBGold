import React from 'react';

export const TableMetadata = ({ table }) => {
  if (!table || !table.columns?.length) return null;

  const columnDescriptions = table.columns.map((col) => {
    const attributes = [col.type || 'TEXT'];
    if (col.primary_key) attributes.push('PK');
    return `${col.name} (${attributes.join(', ')})`;
  });

  return (
    <div className="db-table-metadata">
      <span className="db-metadata-label">Columns: </span>
      <span className="db-metadata-content">{columnDescriptions.join(' | ')}</span>
    </div>
  );
};

export default TableMetadata;

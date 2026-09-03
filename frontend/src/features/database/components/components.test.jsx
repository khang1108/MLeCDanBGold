import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import TableSelector from './TableSelector';
import TableMetadata from './TableMetadata';
import DataGrid from './DataGrid';
import PaginationControls from './PaginationControls';

describe('Database Sub-Components (No Icons, Plain Text UI)', () => {
  test('TableSelector renders table list and invokes onSelectTable on click', () => {
    const tables = [
      { name: 'query_history', row_count: 5 },
      { name: 'submission_files', row_count: 2 },
    ];
    const onSelect = jest.fn();

    render(<TableSelector tables={tables} selectedTableName="query_history" onSelectTable={onSelect} />);

    expect(screen.getByText('query_history (5 rows)')).toBeTruthy();
    const subBtn = screen.getByText('submission_files (2 rows)');
    expect(subBtn).toBeTruthy();

    fireEvent.click(subBtn);
    expect(onSelect).toHaveBeenCalledWith('submission_files');
  });

  test('TableMetadata displays formatted schema columns and primary keys', () => {
    const table = {
      name: 'query_history',
      row_count: 5,
      columns: [
        { name: 'query_id', type: 'TEXT', nullable: false, primary_key: true },
        { name: 'query_text', type: 'TEXT', nullable: true, primary_key: false },
      ],
    };

    render(<TableMetadata table={table} />);
    expect(screen.getByText(/query_id \(TEXT, PK\)/)).toBeTruthy();
    expect(screen.getByText(/query_text \(TEXT\)/)).toBeTruthy();
  });

  test('DataGrid renders table header, cells and stringifies object/array values', () => {
    const columns = [
      { name: 'id', type: 'INTEGER' },
      { name: 'meta', type: 'JSON' },
    ];
    const rows = [
      { id: 1, meta: { tag: 'test' } },
      { id: 2, meta: null },
    ];

    render(<DataGrid columns={columns} rows={rows} isLoading={false} />);
    expect(screen.getByText('id')).toBeTruthy();
    expect(screen.getByText('meta')).toBeTruthy();
    expect(screen.getByText('1')).toBeTruthy();
    expect(screen.getByText('{"tag":"test"}')).toBeTruthy();
    expect(screen.getByText('null')).toBeTruthy();
  });

  test('DataGrid renders empty message when there are no rows', () => {
    render(<DataGrid columns={[{ name: 'id' }]} rows={[]} isLoading={false} />);
    expect(screen.getByText('No rows found in this table.')).toBeTruthy();
  });

  test('PaginationControls triggers previous, next and page size changes', () => {
    const onChangePage = jest.fn();
    const onChangePageSize = jest.fn();

    render(
      <PaginationControls
        page={2}
        pageSize={25}
        totalRows={50}
        totalPages={2}
        onChangePage={onChangePage}
        onChangePageSize={onChangePageSize}
      />
    );

    expect(screen.getByText('Page 2 of 2 (50 total rows)')).toBeTruthy();

    const prevBtn = screen.getByText('Previous');
    expect(prevBtn.disabled).toBe(false);
    fireEvent.click(prevBtn);
    expect(onChangePage).toHaveBeenCalledWith(1);

    const nextBtn = screen.getByText('Next');
    expect(nextBtn.disabled).toBe(true);

    const select = screen.getByLabelText('Rows per page:');
    fireEvent.change(select, { target: { value: '50' } });
    expect(onChangePageSize).toHaveBeenCalledWith(50);
  });
});

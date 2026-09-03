# Design Spec: SQLite Database Browser Page

- **Date:** 2026-09-03
- **Status:** Approved
- **Scope:** Frontend new page for inspecting SQLite database via existing backend endpoints.

---

## 1. Context & Motivation

The backend provides two read-only endpoints for inspecting the application's workspace SQLite database:
1. `GET /api/v1/database/tables`: Lists allowlisted tables, their schemas (columns, data types, nullability, primary key flag), and row counts.
2. `GET /api/v1/database/tables/{table_name}/rows?page={page}&page_size={page_size}`: Returns a paginated slice of raw row dictionaries along with pagination metadata (`total_rows`, `total_pages`).

Currently, the frontend application provides pages for `Query`, `Filter`, and `Workspace`, but lacks an interface to inspect the underlying SQLite database records.

---

## 2. Requirements & Constraints

- **Minimalist & Simple:** Focused solely on exploring table schemas and records cleanly.
- **Zero Icons:** Strictly no icons (no SVGs, no icon fonts, no unicode/emoji icons). All buttons and indicators use plain readable text.
- **Pure White Background:** Clean, high-contrast white background (`#ffffff`) conforming to the user's explicit preference.
- **Responsive & Safe Data Rendering:** Tables display arbitrary SQLite column keys and values safely, stringifying complex JSON values or arrays without throwing runtime rendering errors.
- **Pagination Support:** Users can navigate pages (Previous, Next) and change page size (10, 25, 50, 100 rows per page).
- **Graceful Error & Empty State Handling:** Clear textual messages for loading states, empty tables, and backend unavailability (e.g. HTTP 503 when the database is not configured).

---

## 3. Architecture & File Structure

```text
frontend/
├── src/
│   ├── api/
│   │   ├── database.js               # API client functions for SQLite endpoints
│   │   └── database.test.js          # Unit tests for API client functions
│   ├── features/
│   │   ├── database/
│   │   │   ├── components/
│   │   │   │   ├── TableSelector.jsx # Button list of available tables
│   │   │   │   ├── TableMetadata.jsx # Schema information (columns, types, PKs)
│   │   │   │   ├── DataGrid.jsx      # Clean HTML table for rows
│   │   │   │   └── PaginationControls.jsx # Text-based pagination buttons & dropdown
│   │   │   ├── DatabasePage.jsx      # Feature container & state management
│   │   │   ├── DatabasePage.test.jsx # React unit tests for page behavior
│   │   │   └── index.js              # Feature barrel export
│   │   └── header/
│   │       └── AppHeader.jsx         # Add 'Database' to navigation items
│   ├── styles/
│   │   ├── database.css              # White-background styling without icons
│   │   └── index.css                 # Import database.css
│   └── App.jsx                       # Mount DatabasePage in workspace panel
```

---

## 4. Detailed Component Design

### 4.1. API Client (`src/api/database.js`)
- `fetchDatabaseTables(signal)`:
  - Invokes `requestJson('/api/v1/database/tables', { signal })`.
  - Returns `{ tables: Array<{ name: string, row_count: number, columns: Array<{ name: string, type: string, nullable: boolean, primary_key: boolean }> }> }`.
- `fetchDatabaseRows(tableName, { page = 1, pageSize = 25, signal } = {})`:
  - Builds query string: `?page=${encodeURIComponent(page)}&page_size=${encodeURIComponent(pageSize)}`.
  - Invokes `requestJson(`/api/v1/database/tables/${encodeURIComponent(tableName)}/rows${queryString}`, { signal })`.
  - Returns `{ table: string, page: number, page_size: number, total_rows: number, total_pages: number, rows: Array<Record<string, any>> }`.

### 4.2. UI Components (`src/features/database/`)

#### `TableSelector.jsx`
- Renders available tables as plain text buttons.
- Highlights active table with subtle text emphasis and border (no colored icons or badges).
- Displays text label: `${table.name} (${table.row_count} rows)`.
- Accessible with `aria-pressed={isActive}`.

#### `TableMetadata.jsx`
- Renders schema summary for currently selected table.
- Plain text line: `Columns: col1 (INTEGER, PK), col2 (TEXT), col3 (TEXT, NULL)`.

#### `DataGrid.jsx`
- Formats rows into an HTML `<table>`.
- Header (`<thead>`): Column names derived from table schema columns (or dynamic row keys as fallback).
- Body (`<tbody>`): Row cells.
  - If value is `null` or `undefined`, renders a muted `null`.
  - If value is `boolean`, renders `true` or `false`.
  - If value is an object or array, renders formatted `JSON.stringify(val)`.
  - Otherwise renders string / number representation.
- Empty state: Plain text message `"No rows found in this table."` when `rows.length === 0`.
- Wrapped in a container with horizontal scroll (`overflow-x: auto`) for wide datasets.

#### `PaginationControls.jsx`
- Plain text buttons: `[Previous]` (disabled on page 1) and `[Next]` (disabled on last page or when `page >= total_pages`).
- Status text: `Page {page} of {totalPages || 1} ({totalRows} total rows)`.
- Page size selector: Standard `<select>` with options `[10, 25, 50, 100]`.

#### `DatabasePage.jsx`
- States:
  - `tables`: Array of available tables.
  - `selectedTableName`: Currently selected table name.
  - `rowsData`: Current page result `{ page, page_size, total_rows, total_pages, rows }`.
  - `page`: Current page number (integer, default 1).
  - `pageSize`: Current page size (integer, default 25).
  - `isLoadingTables`: Boolean.
  - `isLoadingRows`: Boolean.
  - `error`: Error message string or null.
- Lifecycle:
  - When mounted and `isActive` is true, loads tables if not already loaded.
  - When `selectedTableName`, `page`, or `pageSize` changes, loads rows.
  - Resets `page` to 1 when user switches tables or changes `pageSize`.

### 4.3. Styling (`src/styles/database.css`)
- Target: `.database-page-container`.
- Background: `#ffffff`.
- Text color: `#1a1a1a` / `var(--color-ink)`.
- Borders: `1px solid #e6e6e6` / `var(--color-hairline)`.
- No pseudo-elements (`::before`, `::after`) rendering icons or glyphs.
- Clean typography and spacing conforming to `tokens.css`.

---

## 5. Integration into App Shell

1. **`AppHeader.jsx`**:
   Update navigation items array:
   ```javascript
   [
     ['query', 'Query'],
     ['filter', 'Filter'],
     ['workspace', 'Workspace'],
     ['database', 'Database'],
   ]
   ```
2. **`App.jsx`**:
   Add workspace panel section:
   ```jsx
   <div className="workspace-panel" hidden={activePage !== 'database'}>
     <DatabasePage isActive={activePage === 'database'} />
   </div>
   ```
3. **`styles/index.css`**:
   Include `@import './database.css';`.

---

## 6. Testing & Quality Assurance

- **Unit Tests (`src/api/database.test.js`)**:
  - `fetchDatabaseTables` requests `/api/v1/database/tables` and parses response.
  - `fetchDatabaseRows` formats query params correctly and calls `/api/v1/database/tables/{tableName}/rows`.
- **Component Tests (`src/features/database/DatabasePage.test.jsx`)**:
  - Renders loading indicator text when fetching tables.
  - Renders table selector buttons and displays metadata.
  - Fetches and displays rows when a table is clicked.
  - Handles page navigation and page size change.
  - Displays readable error message if fetch fails.
- **Zero Icon Verification**:
  - Verify that no `<svg>`, `<i>`, icon classes, or icon emojis exist in `features/database` or `styles/database.css`.
- **Build Verification**:
  - Run `npm test` in `frontend/` to confirm all existing and new tests pass.

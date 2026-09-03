# Design Spec: SQLite Query Execution (Backend & Frontend)

- **Date:** 2026-09-03
- **Status:** Approved
- **Scope:** Full-stack feature to execute arbitrary SQL queries against the workspace SQLite database and display results in the Database tab.

---

## 1. Context & Motivation

The Database tab currently allows users to inspect predefined allowlisted tables (`query_history`, `submission_files`) and paginate their contents. The user wants the ability to input raw SQL queries, transmit them to the backend, execute them against SQLite, and display the output directly on screen.

---

## 2. Requirements & Constraints

- **Execution Capabilities:** Supports both read queries (`SELECT`, `PRAGMA`, `EXPLAIN`) and data-modifying queries (`INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`).
- **Safe Feedback & Error Reporting:** Syntax errors or constraint violations in SQLite return an informative HTTP 400 with the exact SQLite error detail, displayed clearly to the user in a red alert container.
- **Result Presentation:**
  - For queries returning rows (`SELECT`), returns columns and row arrays (capped at `max_rows`, default 100).
  - For mutation queries (`INSERT`, `UPDATE`, `DELETE`), returns `rows_affected` and execution duration.
  - Automatic refresh of table metadata and row counts if a mutation query succeeds.
- **UI Constraints:**
  - Pure white background (`#ffffff`).
  - Strictly zero icons (no SVG, no font icons, no glyph/emoji icons).
  - Clean textarea with monospace font for SQL input, text buttons `[Execute SQL]` and `[Clear]`.

---

## 3. Backend Architecture

### 3.1. HTTP Contracts (`src/hcmai/api/contracts/database.py`)

```python
class DatabaseQueryRequest(BaseModel):
    """Input payload for executing an arbitrary SQL query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    max_rows: int = Field(default=100, ge=1, le=500)


class DatabaseQueryResponse(BaseModel):
    """Execution output from a raw SQL query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, JsonValue]] = Field(default_factory=list)
    rows_affected: int = 0
    execution_time_ms: float = Field(ge=0.0)
    is_mutation: bool = False
```

### 3.2. Data Store Service (`src/hcmai/api/history.py`)

Method `execute_query(self, query: str, *, max_rows: int = 100) -> DatabaseQueryResponse`:
- Uses `with self._connection() as connection:` to open a transaction supporting both reads and writes.
- Records start time and executes `cursor = connection.execute(query)`.
- If `cursor.description` is not None (query produces tabular data):
  - Extracts column names `[col[0] for col in cursor.description]`.
  - Fetches up to `max_rows` rows.
  - Constructs row dictionaries.
  - `is_mutation = False`.
- If `cursor.description` is None (query was a write/DDL operation):
  - `rows_affected = cursor.rowcount if cursor.rowcount >= 0 else 0`.
  - `is_mutation = True`.
- Computes `execution_time_ms = (time.perf_counter() - start_time) * 1000`.
- Catches `sqlite3.Error` and raises `ValueError(str(error))` which the router catches and turns into HTTP 400.

### 3.3. HTTP Router (`src/hcmai/api/routers/database.py`)

- `POST /api/v1/database/execute`:
  - Request body: `DatabaseQueryRequest`.
  - Response model: `DatabaseQueryResponse`.
  - Runs in threadpool.
  - On `ValueError` or `sqlite3.Error`, raises `HTTPException(status_code=400, detail=str(error))`.

---

## 4. Frontend Architecture

### 4.1. API Client (`frontend/src/api/database.js`)

- `executeDatabaseQuery(query, { maxRows = 100, signal } = {})`:
  - Validates `query` is a non-empty string.
  - Calls `requestJson('/api/v1/database/execute', { method: 'POST', body: { query, max_rows: maxRows }, signal })`.

### 4.2. UI Components (`frontend/src/features/database/`)

#### `components/SqlQueryEditor.jsx`
- Textarea with monospace font, placeholder (e.g. `SELECT * FROM query_history LIMIT 10;`).
- Action bar:
  - Text button: `[Execute SQL]` (disabled while executing or if query is blank).
  - Text button: `[Clear]`.
- Execution summary text:
  - If read: `Returned {rows.length} row(s) in {execution_time_ms} ms.`
  - If mutation: `Query executed successfully. {rows_affected} row(s) affected in {execution_time_ms} ms.`
- Error banner: Clean red alert text if execution fails.

#### `DatabasePage.jsx` Updates
- Mounts `SqlQueryEditor` above or alongside the table viewer.
- When an SQL query is executed:
  - If it returns rows, updates the active view to display the query result rows and dynamic columns in `DataGrid`.
  - If it was a mutation, triggers `loadTables()` to refresh table tabs and counts.
- Navigation back to table view: User can click any table tab to return to viewing standard table rows.

### 4.3. Styling (`frontend/src/styles/database.css`)
- `.db-sql-editor`: Container with border, padding, white background.
- `.db-sql-textarea`: Monospace, white background, hairline border, no resize or vertical-only resize.
- `.db-sql-actions`: Flex row with text buttons and execution stats.
- Zero icons.

---

## 5. Verification & Testing

1. **Backend Tests (`tests/api/test_database_routes.py`)**:
   - `SELECT` query returns columns, rows, and timing.
   - `INSERT` query adds a row, reports `rows_affected`, and `SELECT` confirms row exists.
   - Invalid SQL syntax (e.g. `SELCT *`) returns HTTP 400 with SQLite syntax error detail.
2. **Frontend Tests (`frontend/src/api/database.test.js`)**:
   - `executeDatabaseQuery` sends correct POST payload to `/api/v1/database/execute`.
3. **Frontend Component Tests (`frontend/src/features/database/DatabasePage.test.jsx`)**:
   - Typing in SQL editor and clicking `[Execute SQL]` calls API and renders custom query results in `DataGrid`.
   - Mutation query triggers table reload.
   - Failed query renders error message.

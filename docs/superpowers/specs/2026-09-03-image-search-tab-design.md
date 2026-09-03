# Design Specification: Frontend Image Search Tab

**Date:** 2026-09-03  
**Status:** Approved  
**Topic:** Multimodal Video Retrieval - Frontend Image Search Tab  

---

## 1. Context & User Goal

The HCMAI retrieval backend provides an image search endpoint (`POST /api/v1/search/image`) leveraging the SigLIP2 visual vector index. Users need to search for video keyframes directly using an uploaded reference image.

The frontend currently provides tabs for `Query` (text KIS/TRAKE), `Filter`, `Workspace`, and `Database`.

**Goal:**
- Add an **"Image Search"** tab to the navigation bar.
- Replace the text query textarea with an image upload button/area (supporting click-to-browse and drag-and-drop for `.jpg`, `.jpeg`, `.png`, `.webp`, with preview thumbnail, file details, and clear action).
- Keep the workspace structure, options sidebar, and result presentation identical to the existing KIS/TRAKE `SearchWorkspace` (`FramesBox`, latency breakdown, frame inspector with KIS submission, session activity tracking, and history logging).

---

## 2. Backend Contracts & Integration

### Endpoint: `POST /api/v1/search/image`
- **Content-Type**: `multipart/form-data`
- **Request Fields**:
  - `image`: Binary file (`UploadFile`), media type in `{image/jpeg, image/png, image/webp}`. Max size bounded by backend configuration.
  - `top_k`: Form integer `1 <= top_k <= 100` (default: `20`).
- **Response Model (`ImageSearchResponse`)**:
  ```json
  {
    "results": [
      {
        "frame_id": "L01_V001/0001",
        "video_id": "L01_V001",
        "frame_idx": 1,
        "timestamp_ms": 1000,
        "score": 0.85,
        "frame_ids": ["L01_V001/0001"],
        "timestamps_ms": [1000],
        "metadata": {
          "title": null,
          "caption": "...",
          "ocr": "...",
          "objects": ["..."],
          "asr": null
        },
        "fps": 25.0
      }
    ],
    "latency": {
      "query_ms": 12.3,
      "retrieval_ms": 45.6,
      "alignment_ms": 0.0,
      "materialization_ms": 8.1,
      "total_ms": 66.0
    }
  }
  ```

---

## 3. Frontend Architecture & Component Design

### 3.1. Navigation & App Shell
- **`AppHeader.jsx`**:
  - Add `['image-search', 'Image Search']` to the workspace navigation list right after `['query', 'Query']`.
- **`App.jsx`**:
  - Import `ImageSearchWorkspace`.
  - Add workspace panel `<div className="workspace-panel" hidden={activePage !== 'image-search'}>` containing `<ImageSearchWorkspace ... />`.
  - Wire frame clicks, inspector submission modal, User ID focus, and history refresh identical to `SearchWorkspace`.

### 3.2. API Client Layer
- **`frontend/src/api/client.js`**:
  - Export `requestFormData(path, formData, { signal, headers })`:
    - Sends `fetch(`${API_BASE_URL}${path}`, { method: 'POST', body: formData, signal, headers })`.
    - Omits explicit `Content-Type` so the browser assigns the multipart boundary.
    - Handles status check, JSON parsing, error message extraction identically to `requestJson`.
- **`frontend/src/api/search.js`**:
  - Export `searchFramesByImage({ imageFile, topK, signal })`:
    - Appends `image` and `top_k` to `FormData`.
    - Invokes `requestFormData('/api/v1/search/image', formData, { signal })`.
    - Validates response arrays and latency contract.
    - Normalizes latency using `normalizeSearchLatency`.
    - Returns `{ results, latency, warnings }`.

### 3.3. `ImageSearchWorkspace.jsx`
- **Location**: `frontend/src/features/search/components/ImageSearchWorkspace.jsx`
- **Top Row (Upload area)**:
  - Hidden `<input type="file" accept="image/jpeg,image/png,image/webp" />`.
  - Upload container styled to match `.search-query-row`:
    - If no file selected: upload button/area displaying an upload icon, "Choose or drop an image (JPEG, PNG, WebP)".
    - If file selected: shows compact preview thumbnail, file name, formatted file size, and a "Clear / Change" button.
    - Actions:
      - "Search" button (disabled when searching or no image selected).
      - "New Search" button (resets upload, results, latency, error).
- **Sidebar Options (`ToolBox`)**:
  - Top-K numeric input.
  - `SubmissionWorktree` included when active.
  - `showRetrievalSources={false}` passed to `ToolBox` (since image search queries the visual vector index directly).
- **Results (`FramesBox`)**:
  - Displays `GifLoaderOverlay` while searching.
  - Latency banner showing total time and stage breakdown (`query_ms`, `retrieval_ms`, `materialization_ms`).
  - Result grid with `FrameCard` components.
  - Click frame opens `ImageModal` with `submissionMode: 'kis'`.
  - Submit frame submits line `${vid},${frame.frame_idx}` to `requestSubmission`.
- **History & Session**:
  - Creates query history via `createQueryHistory` with `query_text: "[Image] " + file.name` and snapshot from `buildKisSnapshot`.
  - Tracks viewed frames with `markFrameViewed`.
  - Highlights viewed/submitted frames via `activityStateForFrame`.

### 3.4. Styling (`controls.css`)
- Clean image upload dropzone matching HCMAI tokens:
  - Border radius `var(--rounded-md)`, hairline border `var(--color-hairline)`.
  - Interactive states with focus rings and hover styling.
  - Compact image thumbnail preview (`height: 38px`, `width: 38px`, `object-fit: cover`).

---

## 4. Testing & Verification Plan

### Automated Tests
- **Frontend API Tests (`search.test.js`)**:
  - Test `searchFramesByImage` constructs `FormData` correctly and calls `/api/v1/search/image`.
  - Test validation error and network error handling.
- **Component Tests (`ImageSearchWorkspace.test.jsx`)**:
  - Render empty upload state.
  - Selecting/dropping an image displays preview and enables the Search button.
  - Submitting search triggers `searchFramesByImage` and renders results in `FramesBox`.
  - Frame click triggers `onFrameClick` with KIS submission mode.
  - "New Search" resets the state.
- **App Shell Integration (`App.test.jsx`)**:
  - Verify "Image Search" nav button switches `activePage` to `image-search`.
- **Backend API Tests (`tests/api/test_search_routes.py`)**:
  - Existing tests for `POST /api/v1/search/image` pass.

### Manual Verification
- Launch backend and frontend.
- Navigate to "Image Search" tab.
- Upload an image file (e.g., JPEG or PNG).
- Verify Top-K control, click "Search", confirm results render with latency banner and frame cards.
- Click a frame to inspect in `ImageModal` and test submission action.

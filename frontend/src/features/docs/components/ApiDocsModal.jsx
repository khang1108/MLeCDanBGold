import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../../../api/client';

const ENDPOINTS = [
  {
    category: 'System & Diagnostics',
    items: [
      {
        method: 'GET',
        path: '/health',
        title: 'System Health & Capabilities',
        desc: 'Returns online status, total indexed frames, evidence stores (caption, OCR, ASR), and model capabilities.',
        curl: `curl -X GET "${API_BASE_URL}/health"`,
      },
    ],
  },
  {
    category: 'Multimodal Frame Retrieval',
    items: [
      {
        method: 'POST',
        path: '/api/v1/kis/search',
        title: 'KIS Semantic Search & Revision',
        desc: 'Runs progressive multimodal KIS retrieval via explicit semantic operations (initial_resolve, patch_events, global_rewrite, search_only) over canonical video frames. Returns resolved intent, events, entities, temporal metadata, and ranked frames.',
        curl: `curl -i -X POST "${API_BASE_URL}/api/v1/kis/search" \\
  -H "Content-Type: application/json" \\
  -H "X-VBS-User-ID: team-a" \\
  -d '{"base_intent": null, "expected_revision": 0, "operation": {"kind": "initial_resolve", "text": "a red car approaches. It turns left."}, "use_dense": true, "use_bm25": true, "top_k": 20}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/kis/assets',
        title: 'KIS Query Image Upload',
        desc: 'Uploads a content-addressed JPEG/PNG/WebP query image and returns its canonical asset reference.',
        curl: `curl -i -X POST "${API_BASE_URL}/api/v1/kis/assets" \\
  -F "file=@example.jpg;type=image/jpeg"`,
      },
      {
        method: 'GET',
        path: '/api/v1/kis/assets/{asset_id}',
        title: 'KIS Query Image Asset',
        desc: 'Retrieves a previously stored query image asset by its SHA-256 digest reference.',
        curl: `curl -i -X GET "${API_BASE_URL}/api/v1/kis/assets/sha256:..."`,
      },
      {
        method: 'POST',
        path: '/api/v1/search/image',
        title: 'Image-to-Frame Search',
        desc: 'Searches keyframes from an image upload. The response carries the same DRES log-status header as text search.',
        curl: `curl -i -X POST "${API_BASE_URL}/api/v1/search/image" \\
  -H "X-VBS-User-ID: team-a" \\
  -F "image=@query.jpg;type=image/jpeg" \\
  -F "top_k=20"`,
      },
      {
        method: 'POST',
        path: '/api/v1/filter',
        title: 'Evidence Filter',
        desc: 'Returns a backend-paginated page of matching frames and logs a FILTER result event for the connected participant.',
        curl: `curl -i -X POST "${API_BASE_URL}/api/v1/filter" \\
  -H "Content-Type: application/json" \\
  -H "X-VBS-User-ID: team-a" \\
  -d '{"metadata_filters": {"caption": "red car"}, "folder_id": null, "video_id": null, "frames_per_pages": 20, "page_id": 1}'`,
      },
    ],
  },
  {
    category: 'Frame Assets & Evidence',
    items: [
      {
        method: 'GET',
        path: '/api/v1/keyframes/{frame_id}',
        title: 'Canonical Keyframe',
        desc: 'Fetches the canonical keyframe image for an internal frame ID.',
        curl: `curl -X GET "${API_BASE_URL}/api/v1/keyframes/FRAME_ID_HERE"`,
      },
      {
        method: 'GET',
        path: '/api/v1/frames/{frame_id}/metadata',
        title: 'Specialist Evidence Metadata',
        desc: 'Returns deterministic Caption, OCR, Object, and ASR timeline evidence.',
        curl: `curl -X GET "${API_BASE_URL}/api/v1/frames/FRAME_ID_HERE/metadata"`,
      },
    ],
  },
  {
    category: 'VBS Sessions',
    items: [
      {
        method: 'POST',
        path: '/api/v1/vbs/session/connect',
        title: 'Connect a mapped participant',
        desc: 'Connects using only the VBS user ID. DRES credentials and session values are resolved and retained by the backend.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/session/connect" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "team-a"}'`,
      },
      {
        method: 'GET',
        path: '/api/v1/vbs/session/{user_id}',
        title: 'Check or clear participant connection',
        desc: 'GET reports a safe connected flag; DELETE evicts that backend session. Neither response returns credentials or a DRES session.',
        curl: `curl -i "${API_BASE_URL}/api/v1/vbs/session/team-a"`,
      },
      {
        method: 'DELETE',
        path: '/api/v1/vbs/session/{user_id}',
        title: 'Disconnect a participant',
        desc: 'Unlocks the browser after the backend removes this participant’s private DRES session.',
        curl: `curl -X DELETE "${API_BASE_URL}/api/v1/vbs/session/team-a"`,
      },
    ],
  },
  {
    category: 'Private DRES Submissions',
    items: [
      {
        method: 'GET',
        path: '/api/v1/vbs/task/{user_id}',
        title: 'Read the participant’s active task scope',
        desc: 'Requires an already-connected participant ID. Returns safe task metadata and an opaque task_scope_key; the DRES session and credentials remain private to the backend.',
        curl: `curl "${API_BASE_URL}/api/v1/vbs/task/team-a"`,
      },
      {
        method: 'POST',
        path: '/api/v1/vbs/submit',
        title: 'Submit one KIS, AVS, or VQA answer',
        desc: 'KIS and AVS accept one temporal answer; VQA accepts one text answer. Include the task_scope_key returned by task lookup. The backend validates scope and answer kind, then sends once through this participant’s private DRES session.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/submit" -H "Content-Type: application/json" -d '{"user_id": "team-a", "expected_task_scope_key": "dres-task-v1:<copy-from-task-lookup>", "answer": {"kind": "TEMPORAL", "video_id": "L21_V001", "start_ms": 12345, "end_ms": 12345}}'`,
      },
    ],
  },
];

const ApiDocsModal = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState('swagger');
  const [copiedIndex, setCopiedIndex] = useState(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleCopy = (curl, index) => {
    navigator.clipboard?.writeText(curl);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="modal-overlay" onClick={onClose} style={{ zIndex: 9999 }}>
      <div
        className="api-docs-modal-card"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="api-docs-header">
          <div className="api-docs-title-group">
            <span className="api-docs-tag">FASTAPI SPECS</span>
            <h2 className="api-docs-title">Interactive API Documentation</h2>
          </div>

          <div className="api-docs-quick-links">
            <a
              href={`${API_BASE_URL}/docs`}
              target="_blank"
              rel="noreferrer"
              className="api-docs-link-btn"
              title="Open Swagger UI in new tab"
            >
              Swagger /docs ↗
            </a>
            <a
              href={`${API_BASE_URL}/redoc`}
              target="_blank"
              rel="noreferrer"
              className="api-docs-link-btn"
              title="Open ReDoc in new tab"
            >
              ReDoc ↗
            </a>
            <a
              href={`${API_BASE_URL}/openapi.json`}
              target="_blank"
              rel="noreferrer"
              className="api-docs-link-btn"
              title="Open OpenAPI JSON schema"
            >
              openapi.json ↗
            </a>
            <button
              type="button"
              className="modal-close-btn"
              onClick={onClose}
              title="Close [Esc]"
              style={{ position: 'static', marginLeft: '8px' }}
            >
              ✕
            </button>
          </div>
        </div>

        <div className="api-docs-tab-bar">
          <button
            type="button"
            className={`api-docs-tab ${activeTab === 'swagger' ? 'active' : ''}`}
            onClick={() => setActiveTab('swagger')}
          >
            Live Swagger UI
          </button>
          <button
            type="button"
            className={`api-docs-tab ${activeTab === 'reference' ? 'active' : ''}`}
            onClick={() => setActiveTab('reference')}
          >
            Endpoints Quick Reference
          </button>
        </div>

        <div className="api-docs-body">
          {activeTab === 'swagger' ? (
            <div className="api-docs-iframe-wrapper">
              <iframe
                src={`${API_BASE_URL}/docs`}
                title="FastAPI Swagger UI"
                className="api-docs-iframe"
              />
            </div>
          ) : (
            <div className="api-docs-reference-container">
              <aside className="api-docs-log-status" aria-label="DRES result logging status">
                <strong>DRES result logging</strong>
                <span>
                  Successful text, image, and Filter retrievals send one DRES
                  <code> QueryResultLog</code> with the active session of the user who searched.
                  Read <code>X-DRES-Log-Status</code> as <code>sent</code>, <code>failed</code>, or
                  {' '}<code>skipped</code> (no active session). A logging failure does not replace
                  the retrieval response. Credentials and DRES sessions stay on the backend.
                </span>
              </aside>
              {ENDPOINTS.map((category) => (
                <div key={category.category} className="api-docs-category-group">
                  <h3 className="api-docs-category-title">{category.category}</h3>
                  <div className="api-docs-cards-list">
                    {category.items.map((endpoint, itemIdx) => {
                      const uniqueKey = `${category.category}-${itemIdx}`;
                      const isCopied = copiedIndex === uniqueKey;
                      return (
                        <div key={`${endpoint.method}-${endpoint.path}`} className="api-docs-endpoint-card">
                          <div className="api-docs-endpoint-header">
                            <div className="api-docs-path-group">
                              <span className={`api-method-badge ${endpoint.method.toLowerCase()}`}>
                                {endpoint.method}
                              </span>
                              <span className="api-endpoint-path">{endpoint.path}</span>
                            </div>
                            <button
                              type="button"
                              className={`btn-utility api-copy-curl-btn ${isCopied ? 'copied' : ''}`}
                              onClick={() => handleCopy(endpoint.curl, uniqueKey)}
                              title="Copy cURL Command"
                            >
                              {isCopied ? '✓ Copied' : 'Copy cURL'}
                            </button>
                          </div>
                          <p className="api-endpoint-desc">{endpoint.desc}</p>
                          <pre className="api-endpoint-code">
                            <code>{endpoint.curl}</code>
                          </pre>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ApiDocsModal;

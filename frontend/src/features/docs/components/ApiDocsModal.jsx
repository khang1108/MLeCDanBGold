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
        desc: 'Runs semantic KIS retrieval over canonical video frames with clue history. Returns canonical resolved intent, events, entities, temporal metadata, and ranked frames.',
        curl: `curl -i -X POST "${API_BASE_URL}/api/v1/kis/search" \\
  -H "Content-Type: application/json" \\
  -H "X-VBS-User-ID: team-a" \\
  -d '{"inputs": [{"text": "a red car approaches. It turns left."}], "use_dense": true, "use_bm25": true, "top_k": 20}'`,
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
    category: 'Shared Answer Workspace',
    items: [
      {
        method: 'GET',
        path: '/api/v1/answer-workspace',
        title: 'Load the task-scoped answer workspace',
        desc: 'Requires X-VBS-User-ID. Live collaborators receive revisioned snapshots over /api/v1/answer-workspace/ws?user_id=team-a. The snapshot includes a human-readable task_name and an opaque task_scope_key used to guard mutations.',
        curl: `curl -H "X-VBS-User-ID: team-a" "${API_BASE_URL}/api/v1/answer-workspace"`,
      },
    ],
  },
  {
    category: 'KIS, VQA, and AVS Submissions',
    items: [
      {
        method: 'POST',
        path: '/api/v1/vbs/submit/kis',
        title: 'Submit one KIS frame answer',
        desc: 'Forwards one frozen FRAME candidate and expected revisions. Use the task_scope_key returned by the workspace as an optimistic scope guard; it is not a DRES task ID. The backend resolves the canonical media ID and exact timestamp before sending to DRES.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/submit/kis" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "team-a", "task_scope_key": "dres-task-v1:<copy-from-workspace>", "expected_workspace_revision": 4, "candidate_id": "candidate-1", "expected_revision": 2}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/vbs/submit/vqa',
        title: 'Submit one VQA text answer',
        desc: 'Forwards one frozen TEXT candidate with revision checks and the workspace task_scope_key; the backend sends answer text only.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/submit/vqa" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "team-a", "task_scope_key": "dres-task-v1:<copy-from-workspace>", "expected_workspace_revision": 4, "candidate_id": "candidate-2", "expected_revision": 1}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/vbs/submit/avs',
        title: 'Submit the eligible AVS frame set',
        desc: 'Sends the ordered eligible FRAME candidate revisions with the workspace task_scope_key in one DRES request. The server rejects a stale workspace snapshot before forwarding.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/submit/avs" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "team-a", "task_scope_key": "dres-task-v1:<copy-from-workspace>", "expected_workspace_revision": 8, "candidates": [{"candidate_id": "candidate-1", "expected_revision": 2}, {"candidate_id": "candidate-3", "expected_revision": 1}]}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/vbs/submission-attempts/{attempt_id}/resolve',
        title: 'Resolve an unknown submission outcome',
        desc: 'Applies an operator-confirmed accepted or not_accepted decision to the current UNKNOWN attempt without retrying DRES.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vbs/submission-attempts/attempt-1/resolve" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "team-a", "outcome": "not_accepted"}'`,
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

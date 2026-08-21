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
        path: '/api/v1/search',
        title: 'Standalone KIS / Frame Search',
        desc: 'Retrieves top-K candidate keyframes matching natural language or structured queries with multimodal fusion and reranking.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/search" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "a red car turning left", "query_type": "kis", "top_k": 20}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/vqa',
        title: 'Video Question Answering',
        desc: 'Performs multi-frame evidence localization and VLM grounded question answering for natural questions.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/vqa" \\
  -H "Content-Type: application/json" \\
  -d '{"event_description": "a person reads a city sign", "question": "Which city is shown?", "top_k": 20}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/trake',
        title: 'TRAKE Sequential Event Alignment',
        desc: 'Aligns an ordered sequence of semantic events across long video timelines.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/trake" \\
  -H "Content-Type: application/json" \\
  -d '{"events": ["person enters room", "person sits at desk", "person leaves room"], "top_k": 20}'`,
      },
    ],
  },
  {
    category: 'Frame Assets & Evidence',
    items: [
      {
        method: 'GET',
        path: '/api/v1/frames/{frame_id}/image',
        title: 'Full Frame Image',
        desc: 'Fetches high-resolution video keyframe image asset.',
        curl: `curl -X GET "${API_BASE_URL}/api/v1/frames/FRAME_ID_HERE/image"`,
      },
      {
        method: 'GET',
        path: '/api/v1/frames/{frame_id}/thumbnail',
        title: 'Frame Thumbnail',
        desc: 'Fetches lightweight optimized frame thumbnail for grid view.',
        curl: `curl -X GET "${API_BASE_URL}/api/v1/frames/FRAME_ID_HERE/thumbnail"`,
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
    category: 'Query & Submissions',
    items: [
      {
        method: 'POST',
        path: '/api/v1/suggest-query',
        title: 'Query Suggestions',
        desc: 'Generates 5 recommended search queries for multimodal video retrieval exploration.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/suggest-query" \\
  -H "Content-Type: application/json" \\
  -d '{"count": 5}'`,
      },
      {
        method: 'POST',
        path: '/api/v1/parse-query-files',
        title: 'Parse Query Files to CSV',
        desc: 'Parses uploaded .txt query files and maps them to competition .csv submission targets.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/parse-query-files" -F "files=@query_1.txt"`,
      },
      {
        method: 'POST',
        path: '/api/v1/submission',
        title: 'Submit CSV Files',
        desc: 'Submits finalized CSV result files to the evaluation backend.',
        curl: `curl -X POST "${API_BASE_URL}/api/v1/submission" \\
  -H "Content-Type: application/json" \\
  -d '{"files": [{"name": "query-1-kis.csv", "content": "L01_V001,100"}]}'`,
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
              {ENDPOINTS.map((category) => (
                <div key={category.category} className="api-docs-category-group">
                  <h3 className="api-docs-category-title">{category.category}</h3>
                  <div className="api-docs-cards-list">
                    {category.items.map((endpoint, itemIdx) => {
                      const uniqueKey = `${category.category}-${itemIdx}`;
                      const isCopied = copiedIndex === uniqueKey;
                      return (
                        <div key={endpoint.path} className="api-docs-endpoint-card">
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

// Shared HTTP transport for every published FastAPI endpoint.
const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

const resolveDefaultBaseUrl = () => {
  if (process.env.REACT_APP_API_BASE_URL) {
    return process.env.REACT_APP_API_BASE_URL;
  }
  if (typeof window !== 'undefined' && window.location?.hostname) {
    const host = window.location.hostname;
    if (host.includes('iamphuckhang.dev') || host.includes('iamphuckhnag.dev')) {
      return 'https://backend.iamphuckhang.dev';
    }
  }
  return DEFAULT_API_BASE_URL;
};

export const API_BASE_URL = resolveDefaultBaseUrl().replace(/\/+$/, '');
export const DRES_LOG_STATUS_EVENT = 'hcmai:dres-log-status';

const publishDresLogStatus = (response, requestHeaders) => {
  const status = response.headers?.get?.('X-DRES-Log-Status');
  const userId = requestHeaders?.['X-VBS-User-ID']?.trim();
  if (!userId || (status !== 'sent' && status !== 'failed') || typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(DRES_LOG_STATUS_EVENT, {
    detail: { userId, status },
  }));
};

/** Resolve an API-relative asset path without rewriting absolute URLs. */
export const resolveApiUrl = (value) => {
  if (!value || /^(?:https?:|data:)/i.test(value)) return value;
  return `${API_BASE_URL}/${value.replace(/^\/+/, '')}`;
};

const errorMessage = (payload, status) => {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item?.msg || String(item)).filter(Boolean).join('; ')
      || `Request failed with HTTP ${status}`;
  }
  return payload?.detail?.message || `Request failed with HTTP ${status}`;
};

// Parses JSON once and makes backend, malformed-response, and network errors distinct.
export const requestJson = async (path, {
  method = 'GET', body, signal, headers = {},
} = {}) => {
  const options = {
    method,
    headers: { 'Content-Type': 'application/json', ...headers },
  };
  if (body !== undefined) options.body = JSON.stringify(body);
  if (signal) options.signal = signal;

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    const error = new Error(`Could not reach the backend: ${cause?.message || 'network request failed'}`);
    error.cause = cause;
    throw error;
  }
  publishDresLogStatus(response, headers);

  if (response.status === 204 || response.status === 205) {
    if (!response.ok) {
      const error = new Error(`Request failed with HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return null;
  }

  let payload;
  try {
    payload = await response.json();
  } catch (cause) {
    const error = new Error(response.ok ? 'Backend returned invalid JSON' : `Backend returned invalid JSON for HTTP ${response.status}`);
    error.status = response.status;
    error.cause = cause;
    throw error;
  }

  if (!response.ok) {
    const error = new Error(errorMessage(payload, response.status));
    error.status = response.status;
    const code = payload?.detail?.code;
    if (typeof code === 'string' && code.trim()) error.code = code;
    throw error;
  }
  return payload;
};

// Sends multipart form data without setting Content-Type (browser will add boundary).
export const requestFormData = async (path, formData, {
  method = 'POST', signal, headers = {},
} = {}) => {
  const options = {
    method,
    headers: { ...headers },
    body: formData,
  };
  if (signal) options.signal = signal;

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    const error = new Error(`Could not reach the backend: ${cause?.message || 'network request failed'}`);
    error.cause = cause;
    throw error;
  }
  publishDresLogStatus(response, headers);

  if (response.status === 204 || response.status === 205) {
    if (!response.ok) {
      const error = new Error(`Request failed with HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return null;
  }

  let payload;
  try {
    payload = await response.json();
  } catch (cause) {
    const error = new Error(response.ok ? 'Backend returned invalid JSON' : `Backend returned invalid JSON for HTTP ${response.status}`);
    error.status = response.status;
    error.cause = cause;
    throw error;
  }

  if (!response.ok) {
    const error = new Error(errorMessage(payload, response.status));
    error.status = response.status;
    throw error;
  }
  return payload;
};

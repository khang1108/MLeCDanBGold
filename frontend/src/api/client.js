// Shared HTTP transport for every published FastAPI endpoint.
const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export const API_BASE_URL = (
  process.env.REACT_APP_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, '');

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

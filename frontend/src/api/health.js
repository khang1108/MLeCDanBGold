import { requestJson } from './client';

const mockBackendEnabled = () => /^(1|true|yes)$/i.test(
  process.env.REACT_APP_MOCK_BACKEND || '',
);

// Health check endpoint returns status: "ok" and metadata readiness
export const getHealth = (signal) => {
  if (mockBackendEnabled()) {
    return Promise.resolve({
      status: 'ok',
      ready: true,
      frame_store_loaded: true,
      retriever_loaded: true,
      total_frames: 15,
      mock: true,
    });
  }
  return requestJson('/health', { signal });
};

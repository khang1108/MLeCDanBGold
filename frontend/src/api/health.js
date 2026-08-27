import { requestJson } from './client';

// Health check endpoint returns status: "ok" and metadata readiness
export const getHealth = (signal) => requestJson('/health', { signal });

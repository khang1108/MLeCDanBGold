import { getHealth } from './health';
import * as client from './client';

jest.mock('./client');

describe('health API', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('calls /health endpoint successfully', async () => {
    const mockHealth = {
      status: 'ok',
      ready: true,
      frame_store_loaded: true,
      retriever_loaded: true,
      total_frames: 100,
    };
    client.requestJson.mockResolvedValueOnce(mockHealth);

    const result = await getHealth();
    expect(client.requestJson).toHaveBeenCalledWith('/health', { signal: undefined });
    expect(result).toEqual(mockHealth);
  });
});

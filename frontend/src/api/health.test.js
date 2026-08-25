import { getHealth } from './health';
import * as client from './client';

jest.mock('./client');
const originalMockBackend = process.env.REACT_APP_MOCK_BACKEND;

describe('health API', () => {
beforeEach(() => {
  delete process.env.REACT_APP_MOCK_BACKEND;
});

afterEach(() => {
  jest.clearAllMocks();
});

afterAll(() => {
  if (originalMockBackend === undefined) delete process.env.REACT_APP_MOCK_BACKEND;
  else process.env.REACT_APP_MOCK_BACKEND = originalMockBackend;
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

  test('returns a ready local health response when mock backend is enabled', async () => {
    const previous = process.env.REACT_APP_MOCK_BACKEND;
    process.env.REACT_APP_MOCK_BACKEND = 'true';
    try {
      await expect(getHealth()).resolves.toEqual(expect.objectContaining({
        status: 'ok',
        ready: true,
        mock: true,
      }));
      expect(client.requestJson).not.toHaveBeenCalled();
    } finally {
      if (previous === undefined) delete process.env.REACT_APP_MOCK_BACKEND;
      else process.env.REACT_APP_MOCK_BACKEND = previous;
    }
  });
});

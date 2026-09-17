import { requestJson } from './client';

describe('client structured errors', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('propagates non-DRES structured error codes such as TRAIL_REVISION_CONFLICT', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 409,
      headers: { get: () => null },
      json: async () => ({
        detail: {
          code: 'TRAIL_REVISION_CONFLICT',
          message: 'Expected trail revision 2 but session is at 3',
        },
      }),
    });

    await expect(requestJson('/api/v1/event-trail/s1/actions')).rejects.toMatchObject({
      status: 409,
      code: 'TRAIL_REVISION_CONFLICT',
      message: 'Expected trail revision 2 but session is at 3',
    });
  });
});

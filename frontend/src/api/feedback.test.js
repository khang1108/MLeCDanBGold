import { openFeedbackSession, sendFeedbackTurn, undoFeedback } from './feedback';
import * as client from './client';

jest.mock('./client');

describe('feedback API client', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('openFeedbackSession validates inputs and makes POST request', async () => {
    client.requestJson.mockResolvedValueOnce({
      session_id: 'fbs_123',
      feedback_revision: 1,
      state: { session_id: 'fbs_123', status: 'applied', feedback_revision: 1 },
    });

    const result = await openFeedbackSession({
      intent: { revision: 1, query_text: 'dog', events: [] },
      originalQuery: 'dog',
      evidenceSnapshotId: 'snap_123',
    });

    expect(client.requestJson).toHaveBeenCalledWith(
      '/api/v1/kis/feedback/open',
      expect.objectContaining({
        method: 'POST',
        body: expect.objectContaining({
          original_query: 'dog',
          evidence_snapshot_id: 'snap_123',
        }),
      }),
    );
    expect(result.session_id).toBe('fbs_123');
  });

  test('sendFeedbackTurn posts to dynamic URL with encoded session ID', async () => {
    client.requestJson.mockResolvedValueOnce({
      session_id: 'fbs/special',
      feedback_revision: 2,
      status: 'applied',
    });

    const result = await sendFeedbackTurn('fbs/special', {
      requestId: 'req_1',
      expectedFeedbackRevision: 1,
      expectedKisRevision: 1,
      message: 'more red',
      selectedResultId: 'r1',
    });

    expect(client.requestJson).toHaveBeenCalledWith(
      '/api/v1/kis/feedback/fbs%2Fspecial/turn',
      expect.objectContaining({
        method: 'POST',
        body: expect.objectContaining({
          request_id: 'req_1',
          message: 'more red',
          selected_result_id: 'r1',
        }),
      }),
    );
    expect(result.feedback_revision).toBe(2);
  });

  test('undoFeedback posts expected revision to undo endpoint', async () => {
    client.requestJson.mockResolvedValueOnce({
      session_id: 'fbs_123',
      feedback_revision: 3,
      status: 'applied',
    });

    const result = await undoFeedback('fbs_123', {
      requestId: 'undo_1',
      expectedFeedbackRevision: 2,
    });

    expect(client.requestJson).toHaveBeenCalledWith(
      '/api/v1/kis/feedback/fbs_123/undo',
      expect.objectContaining({
        method: 'POST',
        body: expect.objectContaining({
          request_id: 'undo_1',
          expected_feedback_revision: 2,
        }),
      }),
    );
    expect(result.feedback_revision).toBe(3);
  });
});

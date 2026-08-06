import { act, renderHook } from '@testing-library/react';
import { getHealth } from '../../../api/health';
import { useHealthCheck } from './useHealthCheck';

jest.mock('../../../api/health');

describe('useHealthCheck', () => {
  afterEach(() => {
    jest.clearAllMocks();
    jest.restoreAllMocks();
  });

  test('checks once on mount without starting a polling interval', async () => {
    const setIntervalSpy = jest.spyOn(global, 'setInterval');
    getHealth.mockResolvedValueOnce({ status: 'ok' });

    const { result } = renderHook(() => useHealthCheck());

    await act(async () => {});

    expect(result.current.isHealthy).toBe(true);
    expect(getHealth).toHaveBeenCalledTimes(1);
    expect(setIntervalSpy).not.toHaveBeenCalled();
  });
});

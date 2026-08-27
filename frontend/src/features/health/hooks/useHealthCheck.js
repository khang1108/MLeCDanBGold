import { useCallback, useEffect, useRef, useState } from 'react';
import { getHealth } from '../../../api/health';

export const useHealthCheck = () => {
  const [isHealthy, setIsHealthy] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const isMountedRef = useRef(true);

  const requestRef = useRef(null);

  const checkHealth = useCallback(async () => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const data = await getHealth(controller.signal);
      if (isMountedRef.current) {
        const healthy = data?.status === 'ok' && data?.ready === true;
        setIsHealthy(healthy);
        setHealthData(data);
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (isMountedRef.current) {
        setIsHealthy(false);
        setHealthData(null);
      }
    } finally {
      if (isMountedRef.current && requestRef.current === controller) {
        requestRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    checkHealth();

    return () => {
      isMountedRef.current = false;
      requestRef.current?.abort();
    };
  }, [checkHealth]);

  return { isHealthy, healthData };
};

import { useCallback, useEffect, useRef, useState } from 'react';
import { getHealth } from '../../../api/health';

export const useHealthCheck = () => {
  const [isHealthy, setIsHealthy] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState(null);
  const isMountedRef = useRef(true);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);
    try {
      const data = await getHealth();
      if (isMountedRef.current) {
        const healthy = data?.status === 'ok';
        setIsHealthy(healthy);
        setHealthData(data);
        setError(healthy ? null : 'System returned unhealthy status');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setIsHealthy(false);
        setHealthData(null);
        setError(err.message || 'System offline');
      }
    } finally {
      if (isMountedRef.current) {
        setIsChecking(false);
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    checkHealth();

    return () => {
      isMountedRef.current = false;
    };
  }, [checkHealth]);

  return { isHealthy, healthData, isChecking, error, checkHealth };
};

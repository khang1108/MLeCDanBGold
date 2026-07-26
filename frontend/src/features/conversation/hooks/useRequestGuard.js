import { useCallback, useRef, useState } from 'react';

// Prevents New, Send, and session selection from creating overlapping mutations.
export const useRequestGuard = () => {
  const pendingRef = useRef(false);
  const [isPending, setIsPending] = useState(false);

  const runRequest = useCallback(async (work) => {
    if (pendingRef.current) return undefined;
    pendingRef.current = true;
    setIsPending(true);
    try {
      return await work();
    } finally {
      pendingRef.current = false;
      setIsPending(false);
    }
  }, []);

  return { isPending, pendingRef, runRequest };
};
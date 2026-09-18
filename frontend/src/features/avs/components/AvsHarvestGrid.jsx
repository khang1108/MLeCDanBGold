import React, { useCallback, useEffect, useRef, useState } from 'react';
import AvsCandidateCard from './AvsCandidateCard';
import { getDirectionalIndex } from '../gridNavigation';

/**
 * Responsive keyframe harvest grid with roving tab index,
 * directional arrow key navigation, Space to toggle, and Enter to inspect.
 */
const AvsHarvestGrid = ({
  candidates = [],
  pending = new Map(),
  submitted = new Map(),
  selectionDisabled = false,
  onToggle,
  onInspect,
}) => {
  const [focusedIndex, setFocusedIndex] = useState(0);
  const cardRefs = useRef([]);

  // Reset or constrain focusedIndex when candidate list changes
  useEffect(() => {
    if (candidates.length === 0) {
      setFocusedIndex(0);
    } else if (focusedIndex >= candidates.length) {
      setFocusedIndex(candidates.length - 1);
    }
  }, [candidates.length, focusedIndex]);

  const handleKeyDown = useCallback((e, index) => {
    // Ignore keyboard shortcuts originating from form elements or buttons inside the card
    const targetTag = e.target?.tagName?.toUpperCase();
    if (targetTag === 'INPUT' || targetTag === 'BUTTON' || targetTag === 'SELECT' || targetTag === 'TEXTAREA') {
      return;
    }

    const candidate = candidates[index];
    if (!candidate) return;

    if (e.key === ' ') {
      e.preventDefault();
      onToggle?.(candidate);
      return;
    }

    if (e.key === 'Enter') {
      e.preventDefault();
      onInspect?.(candidate);
      return;
    }

    if (['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft'].includes(e.key)) {
      e.preventDefault();
      const rects = cardRefs.current.map((node) =>
        node ? node.getBoundingClientRect() : null
      );
      const nextIndex = getDirectionalIndex(rects, index, e.key);
      if (nextIndex !== index && nextIndex >= 0 && nextIndex < candidates.length) {
        setFocusedIndex(nextIndex);
        cardRefs.current[nextIndex]?.focus();
      }
    }
  }, [candidates, onToggle, onInspect]);

  if (!Array.isArray(candidates) || candidates.length === 0) {
    return null;
  }

  return (
    <section className="avs-harvest-grid" aria-label="AVS search candidates">
      {candidates.map((cand, idx) => {
        const id = cand.candidate_id || cand.frame_id;
        const isSelected = pending.has(id);
        const isSubmitted = submitted.has(id);
        const isFocused = idx === focusedIndex;

        return (
          <AvsCandidateCard
            key={id}
            candidate={cand}
            selected={isSelected}
            submitted={isSubmitted}
            selectionDisabled={selectionDisabled}
            onToggle={onToggle}
            onInspect={onInspect}
            cardRef={(el) => {
              cardRefs.current[idx] = el;
            }}
            tabIndex={isFocused ? 0 : -1}
            onKeyDown={(e) => handleKeyDown(e, idx)}
          />
        );
      })}
    </section>
  );
};

export default AvsHarvestGrid;

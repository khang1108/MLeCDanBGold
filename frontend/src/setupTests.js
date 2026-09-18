// jest-dom adds custom jest matchers for asserting on DOM nodes.
import '@testing-library/jest-dom';

// Global stubs for jsdom missing media element and canvas methods in tests
if (typeof window !== 'undefined') {
  if (window.HTMLMediaElement) {
    window.HTMLMediaElement.prototype.play = () => Promise.resolve();
    window.HTMLMediaElement.prototype.pause = () => {};
    window.HTMLMediaElement.prototype.load = () => {};
  }

  if (window.HTMLCanvasElement) {
    const originalGetContext = window.HTMLCanvasElement.prototype.getContext;
    window.HTMLCanvasElement.prototype.getContext = function (...args) {
      try {
        if (originalGetContext) {
          const ctx = originalGetContext.apply(this, args);
          if (ctx) return ctx;
        }
      } catch {}
      return {
        clearRect: () => {},
        beginPath: () => {},
        moveTo: () => {},
        bezierCurveTo: () => {},
        lineTo: () => {},
        stroke: () => {},
        arc: () => {},
        fill: () => {},
        save: () => {},
        restore: () => {},
      };
    };
  }
}

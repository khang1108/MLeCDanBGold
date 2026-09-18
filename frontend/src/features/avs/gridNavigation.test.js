import { getDirectionalIndex } from './gridNavigation';

const rect = (left, top, width = 100, height = 100) => ({
  left,
  top,
  right: left + width,
  bottom: top + height,
  width,
  height,
  x: left,
  y: top,
});

describe('getDirectionalIndex', () => {
  // Grid layout (2x2):
  // 0 (0, 0)     1 (120, 0)
  // 2 (0, 120)   3 (120, 120)
  const cards = [
    rect(0, 0),
    rect(120, 0),
    rect(0, 120),
    rect(120, 120),
  ];

  test('ArrowDown chooses the nearest card below the current card', () => {
    expect(getDirectionalIndex(cards, 0, 'ArrowDown')).toBe(2);
    expect(getDirectionalIndex(cards, 1, 'ArrowDown')).toBe(3);
  });

  test('ArrowUp chooses the nearest card above the current card', () => {
    expect(getDirectionalIndex(cards, 2, 'ArrowUp')).toBe(0);
    expect(getDirectionalIndex(cards, 3, 'ArrowUp')).toBe(1);
  });

  test('ArrowRight chooses the nearest card to the right', () => {
    expect(getDirectionalIndex(cards, 0, 'ArrowRight')).toBe(1);
    expect(getDirectionalIndex(cards, 2, 'ArrowRight')).toBe(3);
  });

  test('ArrowLeft chooses the nearest card to the left', () => {
    expect(getDirectionalIndex(cards, 1, 'ArrowLeft')).toBe(0);
    expect(getDirectionalIndex(cards, 3, 'ArrowLeft')).toBe(2);
  });

  test('edge no-op returns current index when moving out of bounds', () => {
    expect(getDirectionalIndex(cards, 0, 'ArrowUp')).toBe(0);
    expect(getDirectionalIndex(cards, 0, 'ArrowLeft')).toBe(0);
    expect(getDirectionalIndex(cards, 3, 'ArrowDown')).toBe(3);
    expect(getDirectionalIndex(cards, 3, 'ArrowRight')).toBe(3);
  });

  test('handles invalid inputs gracefully', () => {
    expect(getDirectionalIndex([], 0, 'ArrowDown')).toBe(0);
    expect(getDirectionalIndex(cards, -1, 'ArrowDown')).toBe(-1);
    expect(getDirectionalIndex(cards, 0, 'KeyA')).toBe(0);
  });
});

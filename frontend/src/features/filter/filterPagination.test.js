import {
  calculateFramesPerPage,
  getPaginationItems,
} from './filterPagination';

test('calculates more frames for a larger result viewport', () => {
  const laptopPageSize = calculateFramesPerPage({ width: 1076, height: 540 });
  const desktopPageSize = calculateFramesPerPage({ width: 1600, height: 900 });

  expect(laptopPageSize).toBe(12);
  expect(desktopPageSize).toBeGreaterThan(laptopPageSize);
  expect(desktopPageSize).toBeLessThanOrEqual(48);
});

test('uses a safe fallback while the hidden workspace has no dimensions', () => {
  expect(calculateFramesPerPage({ width: 0, height: 0 })).toBe(12);
  expect(calculateFramesPerPage()).toBe(12);
});

test('keeps pagination compact while preserving nearby pages', () => {
  expect(getPaginationItems(5, 3)).toEqual([1, 2, 3, 4, 5]);
  expect(getPaginationItems(12, 6)).toEqual([
    1, 'ellipsis-5', 5, 6, 7, 'ellipsis-12', 12,
  ]);
});

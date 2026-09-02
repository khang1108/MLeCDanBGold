import {
  calculateFramesPerPage,
  FILTER_PAGE_SIZE_OPTIONS,
  getPaginationItems,
  resolveFramesPerPage,
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

test('resolves only Auto, 12, 24, and 48 into bounded request sizes', () => {
  expect(FILTER_PAGE_SIZE_OPTIONS).toEqual(['auto', 12, 24, 48]);
  expect(resolveFramesPerPage(24, { width: 200, height: 200 })).toBe(24);
  expect(resolveFramesPerPage('auto', { width: 1600, height: 900 })).toBeGreaterThan(12);
  expect(resolveFramesPerPage('invalid', { width: 0, height: 0 })).toBe(12);
  expect(resolveFramesPerPage(36, { width: 1600, height: 900 })).toBeGreaterThan(12);
});

test('keeps pagination compact while preserving nearby pages', () => {
  expect(getPaginationItems(5, 3)).toEqual([1, 2, 3, 4, 5]);
  expect(getPaginationItems(12, 6)).toEqual([
    1, 'ellipsis-5', 5, 6, 7, 'ellipsis-12', 12,
  ]);
});

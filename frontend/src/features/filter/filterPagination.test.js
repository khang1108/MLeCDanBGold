import { FRAMES_PER_PAGE, getPaginationItems } from './filterPagination';

test('uses one fixed Filter page size without viewport calculation or fallback', () => {
  expect(FRAMES_PER_PAGE).toBe(20);
});

test('keeps pagination compact while preserving nearby pages', () => {
  expect(getPaginationItems(5, 3)).toEqual([1, 2, 3, 4, 5]);
  expect(getPaginationItems(12, 6)).toEqual([
    1, 'ellipsis-5', 5, 6, 7, 'ellipsis-12', 12,
  ]);
});

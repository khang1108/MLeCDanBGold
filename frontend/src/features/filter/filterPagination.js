export const DEFAULT_FRAMES_PER_PAGE = 12;

const MIN_CARD_WIDTH = 170;
const GRID_GAP = 6;
const CARD_IMAGE_RATIO = 2 / 3;
const CARD_CHROME_HEIGHT = 104;
const RESULT_SUMMARY_HEIGHT = 28;
const PAGINATION_HEIGHT = 42;
const RESULT_BOTTOM_SPACE = 8;
const MAX_FRAMES_PER_PAGE = 48;

/**
 * Estimate how many cards fit in the visible Filter result area.
 *
 * The grid uses a 170px minimum card width and a 3:2 image area. The extra
 * card height accounts for the header, caption, and optional timestamp row.
 * Keeping the estimate here makes the request contract deterministic while
 * avoiding a page that immediately overflows on smaller laptop displays.
 */
export const calculateFramesPerPage = ({ width, height } = {}) => {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return DEFAULT_FRAMES_PER_PAGE;
  }

  const columns = Math.max(
    1,
    Math.floor((width + GRID_GAP) / (MIN_CARD_WIDTH + GRID_GAP)),
  );
  const cardWidth = (width - ((columns - 1) * GRID_GAP)) / columns;
  const cardHeight = (cardWidth * CARD_IMAGE_RATIO) + CARD_CHROME_HEIGHT;
  const usableHeight = Math.max(
    cardHeight,
    height - RESULT_SUMMARY_HEIGHT - PAGINATION_HEIGHT - RESULT_BOTTOM_SPACE,
  );
  const rows = Math.max(
    1,
    Math.floor((usableHeight + GRID_GAP) / (cardHeight + GRID_GAP)),
  );

  return Math.min(columns * rows, MAX_FRAMES_PER_PAGE);
};

/**
 * Build a compact page control model with first/last and nearby pages.
 * Ellipsis entries are presentation-only and never become interactive pages.
 */
export const getPaginationItems = (totalPages, currentPage) => {
  if (!Number.isInteger(totalPages) || totalPages <= 0) return [];

  const page = Math.min(Math.max(currentPage, 1), totalPages);
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }

  const pages = new Set([1, totalPages, page - 1, page, page + 1]);
  const sortedPages = Array.from(pages)
    .filter((pageNumber) => pageNumber >= 1 && pageNumber <= totalPages)
    .sort((left, right) => left - right);

  return sortedPages.reduce((items, pageNumber, index) => {
    if (index > 0 && pageNumber - sortedPages[index - 1] > 1) {
      items.push(`ellipsis-${pageNumber}`);
    }
    items.push(pageNumber);
    return items;
  }, []);
};

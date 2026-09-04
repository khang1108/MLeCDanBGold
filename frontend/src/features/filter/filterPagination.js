/** The Filter API and workspace always exchange exactly twenty frames per page. */
export const FRAMES_PER_PAGE = 20;


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

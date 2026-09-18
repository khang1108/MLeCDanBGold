/**
 * Directional keyboard navigation helper for responsive candidate grids.
 * Calculates nearest visual neighbor based on DOM rectangle centers.
 */

export const getDirectionalIndex = (cards, currentIndex, key) => {
  if (!Array.isArray(cards) || cards.length === 0) {
    return currentIndex;
  }
  if (currentIndex < 0 || currentIndex >= cards.length) {
    return currentIndex;
  }
  const current = cards[currentIndex];
  if (!current) return currentIndex;

  const currentCenterX = current.left + current.width / 2;
  const currentCenterY = current.top + current.height / 2;

  let bestIndex = currentIndex;
  let minPrimary = Infinity;
  let minSecondary = Infinity;

  for (let i = 0; i < cards.length; i++) {
    if (i === currentIndex) continue;
    const card = cards[i];
    if (!card) continue;

    const centerX = card.left + card.width / 2;
    const centerY = card.top + card.height / 2;

    let inHalfPlane = false;
    let primaryDist = 0;
    let secondaryDist = 0;

    switch (key) {
      case 'ArrowDown':
        inHalfPlane = centerY > currentCenterY;
        primaryDist = centerY - currentCenterY;
        secondaryDist = Math.abs(centerX - currentCenterX);
        break;
      case 'ArrowUp':
        inHalfPlane = centerY < currentCenterY;
        primaryDist = currentCenterY - centerY;
        secondaryDist = Math.abs(centerX - currentCenterX);
        break;
      case 'ArrowRight':
        inHalfPlane = centerX > currentCenterX;
        primaryDist = centerX - currentCenterX;
        secondaryDist = Math.abs(centerY - currentCenterY);
        break;
      case 'ArrowLeft':
        inHalfPlane = centerX < currentCenterX;
        primaryDist = currentCenterX - centerX;
        secondaryDist = Math.abs(centerY - currentCenterY);
        break;
      default:
        return currentIndex;
    }

    if (!inHalfPlane) continue;

    if (
      primaryDist < minPrimary ||
      (primaryDist === minPrimary && secondaryDist < minSecondary) ||
      (primaryDist === minPrimary && secondaryDist === minSecondary && i < bestIndex)
    ) {
      minPrimary = primaryDist;
      minSecondary = secondaryDist;
      bestIndex = i;
    }
  }

  return bestIndex;
};

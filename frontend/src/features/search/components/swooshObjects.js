/**
 * Standalone comic action particles inspired by manga/comic speed lines & action cues:
 * Small stars, lightning bolts, sweat/action droplets, speed dashes, mini bursts, and comic hearts.
 */

export const COMIC_SHAPES = [
  // 1. Classic 4-pointed comic star
  {
    type: "star4",
    viewBox: "0 0 24 24",
    path: "M12 0 L14.5 9.5 L24 12 L14.5 14.5 L12 24 L9.5 14.5 L0 12 L9.5 9.5 Z",
  },
  // 2. Dynamic 5-pointed action star
  {
    type: "star5",
    viewBox: "0 0 24 24",
    path: "M12 1.5 L15.2 8.5 L23 9.4 L17.2 14.8 L18.8 22.5 L12 18.6 L5.2 22.5 L6.8 14.8 L1 9.4 L8.8 8.5 Z",
  },
  // 3. Sharp lightning zap / electric crackle
  {
    type: "lightning",
    viewBox: "0 0 24 24",
    path: "M13 1 L4 13 L11 13 L8 23 L20 10 L13 10 Z",
  },
  // 4. Zigzag tremor spark
  {
    type: "zigzag",
    viewBox: "0 0 24 24",
    path: "M3 12 L7 5 L11 14 L15 7 L19 16 L22 10",
    strokeOnly: true,
  },
  // 5. Action speed droplet / sweat bead
  {
    type: "droplet",
    viewBox: "0 0 24 24",
    path: "M12 2 C15 7 20 13 20 17 A8 8 0 0 1 4 17 C4 13 9 7 12 2 Z",
  },
  // 6. Motion dash / speed line
  {
    type: "dash",
    viewBox: "0 0 32 12",
    path: "M2 6 C10 3 22 3 30 6 C22 9 10 9 2 6 Z",
  },
  // 7. Triple speed streaks
  {
    type: "streaks",
    viewBox: "0 0 28 20",
    path: "M2 4 L26 4 M6 10 L26 10 M10 16 L26 16",
    strokeOnly: true,
  },
  // 8. Comic impact starburst (mini-crack)
  {
    type: "burst",
    viewBox: "0 0 24 24",
    path: "M12 0 L14 7 L21 4 L17 11 L24 14 L17 17 L21 24 L14 21 L12 24 L10 21 L3 24 L7 17 L0 14 L7 11 L3 4 L10 7 Z",
  },
  // 9. Comic action heart (as seen in manga speed cues)
  {
    type: "heart",
    viewBox: "0 0 24 24",
    path: "M12 21.35 C12 21.35 3 14.5 3 8.5 A5.5 5.5 0 0 1 12 5.09 A5.5 5.5 0 0 1 21 8.5 C21 14.5 12 21.35 12 21.35 Z",
  },
  // 10. Slash / curved whip trail cut
  {
    type: "slash",
    viewBox: "0 0 24 24",
    path: "M2 22 Q12 18 16 10 Q19 5 22 2 Q18 7 14 13 Q9 18 2 22 Z",
  },
];

export const getRandomComicShape = () => {
  const index = Math.floor(Math.random() * COMIC_SHAPES.length);
  return COMIC_SHAPES[index];
};

// Backward-compatible export for existing tests/references
export const getRandomSwoosh = getRandomComicShape;
export const SWOOSH_OBJECTS = COMIC_SHAPES;

/**
 * Audio helpers for BadClaudeLoader:
 * 1. Background lottery music (nhac-xo-so.mp3) played continuously during search until completion.
 * 2. Random sound effects from public/sound/ (akhhhh.mp3, pum-impacto.mp3) played on whip whoosh/click.
 * 3. Instantaneous Web Audio API whip snap for zero-latency tactile feedback.
 */

// Sound effects in public/sound excluding nhac-xo-so.mp3
export const CLICK_SFX_FILES = [
  `${process.env.PUBLIC_URL || ""}/sound/akhhhh.mp3`,
  `${process.env.PUBLIC_URL || ""}/sound/pum-impacto.mp3`,
];

export const LOTTERY_MUSIC_FILE = `${process.env.PUBLIC_URL || ""}/sound/nhac-xo-so.mp3`;

/**
 * Instantaneous Web Audio API whip snap for tactile response
 */
const playSynthWhipCrack = () => {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    const t = ctx.currentTime;
    const bufferSize = Math.floor(ctx.sampleRate * 0.1);
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.009));
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.setValueAtTime(850, t);

    const whipGain = ctx.createGain();
    whipGain.gain.setValueAtTime(0.4, t);
    whipGain.gain.exponentialRampToValueAtTime(0.01, t + 0.09);

    noise.connect(filter);
    filter.connect(whipGain);
    whipGain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.1);

    setTimeout(() => {
      try {
        if (ctx.state !== "closed") {
          ctx.close().catch(() => {});
        }
      } catch {
        // ignore
      }
    }, 150);
  } catch {
    // ignore
  }
};

/**
 * Play a random sound effect from public/sound (excluding lottery music) on whip whoosh/click
 */
export const playRandomClickSound = (isMuted = false) => {
  if (isMuted) return;

  // 1. Zero-latency sharp whip snap
  playSynthWhipCrack();

  // 2. Random sound effect from public/sound (akhhhh.mp3, pum-impacto.mp3)
  if (typeof Audio !== "undefined" && CLICK_SFX_FILES.length > 0) {
    try {
      const randomIdx = Math.floor(Math.random() * CLICK_SFX_FILES.length);
      const sfx = new Audio(CLICK_SFX_FILES[randomIdx]);
      sfx.volume = 0.8;
      const playPromise = sfx.play();
      if (playPromise !== undefined && typeof playPromise.catch === "function") {
        playPromise.catch(() => {
          // Silently handle autoplay restriction or decode error
        });
      }
    } catch {
      // Silently handle
    }
  }
};

// Backward-compatible alias
export const playWhipCrackAndMemeSound = playRandomClickSound;

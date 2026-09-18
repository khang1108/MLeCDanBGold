import React, { useCallback, useEffect, useRef, useState } from "react";
import { getRandomComicShape } from "./swooshObjects";
import { LOTTERY_MUSIC_FILE, playWhipCrackAndMemeSound } from "./whipAudio";

// OpenWhip physics parameters for realistic whip motion
const P = {
  segments: 28,
  segmentLength: 18,
  taper: 0.6,
  gravity: 1.1,
  damping: 0.96,
  constraintIters: 20,
  maxStretchRatio: 1.2,
  baseTargetAngle: -1.12,
  handleAimByMouseX: 0.4,
  handleAimByMouseY: 0.2,
  handleAimClamp: 2.0,
  handleSpring: 0.7,
  handleAngularDamping: 0.078,
  basePoseSegments: 2,
  basePoseStiffStart: 0.9,
  basePoseStiffEnd: 0.8,
  handleMaxBendDeg: 16,
  tipMaxBendDeg: 130,
  bendRigidityStart: 0.8,
  bendRigidityEnd: 0.12,
  crackSpeed: 280,
  crackCooldownMs: 220,
  lineWidthHandle: 7,
  lineWidthTip: 4,
  outlineWidth: 2.5,
  handleExtraWidth: 4,
  handleThickSegments: 2,
  arcWidth: 220,
  arcHeight: 140,
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const wrapPi = (a) => {
  let res = a;
  while (res > Math.PI) res -= Math.PI * 2;
  while (res < -Math.PI) res += Math.PI * 2;
  return res;
};

function segLen(i) {
  const t = i / (P.segments - 1);
  return P.segmentLength * (1 - t * (1 - P.taper));
}

function catmullPoint(pts, i) {
  const n = pts.length;
  if (n === 0) return { x: 0, y: 0 };
  if (i < 0) {
    if (n >= 2) return { x: 2 * pts[0].x - pts[1].x, y: 2 * pts[0].y - pts[1].y };
    return { x: pts[0].x, y: pts[0].y };
  }
  if (i >= n) {
    if (n >= 2) {
      const a = pts[n - 2], b = pts[n - 1];
      return { x: 2 * b.x - a.x, y: 2 * b.y - a.y };
    }
    return { x: pts[n - 1].x, y: pts[n - 1].y };
  }
  return pts[i];
}

function whipSegmentBezier(pts, i) {
  const p0 = catmullPoint(pts, i - 1);
  const p1 = pts[i];
  const p2 = pts[i + 1];
  const p3 = catmullPoint(pts, i + 2);
  return {
    cp1x: p1.x + (p2.x - p0.x) / 6,
    cp1y: p1.y + (p2.y - p0.y) / 6,
    cp2x: p2.x - (p3.x - p1.x) / 6,
    cp2y: p2.y - (p3.y - p1.y) / 6,
    x2: p2.x,
    y2: p2.y,
  };
}

/** OpenWhip: Clamp bend angle per joint (handle stiff, tip floppy and flexible) */
function applyBendLimits(whip) {
  if (!whip || whip.length < 3) return;
  for (let i = 1; i < whip.length - 1; i++) {
    const a = whip[i - 1];
    const b = whip[i];
    const c = whip[i + 1];

    const v1x = a.x - b.x;
    const v1y = a.y - b.y;
    const v2x = c.x - b.x;
    const v2y = c.y - b.y;
    const l1 = Math.hypot(v1x, v1y) || 0.0001;
    const l2 = Math.hypot(v2x, v2y) || 0.0001;
    const n1x = v1x / l1, n1y = v1y / l1;
    const n2x = v2x / l2, n2y = v2y / l2;

    const dot = clamp(n1x * n2x + n1y * n2y, -1, 1);
    const angle = Math.acos(dot);
    const t = i / (whip.length - 2);
    const maxBend = (lerp(P.handleMaxBendDeg, P.tipMaxBendDeg, t) * Math.PI) / 180;
    const bend = Math.PI - angle;
    if (bend <= maxBend) continue;

    const cross = n1x * n2y - n1y * n2x;
    const sign = cross >= 0 ? 1 : -1;
    const targetAngle = Math.PI - maxBend;
    const targetA = Math.atan2(n1y, n1x) + sign * targetAngle;
    const tx = b.x + Math.cos(targetA) * l2;
    const ty = b.y + Math.sin(targetA) * l2;
    const rigidity = lerp(P.bendRigidityStart, P.bendRigidityEnd, t);

    c.x = lerp(c.x, tx, rigidity);
    c.y = lerp(c.y, ty, rigidity);
  }
}

/** OpenWhip: Hard cap for segment stretch ratio to prevent rubber banding */
function capSegmentStretch(whip) {
  if (!whip || whip.length < 2) return;
  for (let i = 0; i < whip.length - 1; i++) {
    const a = whip[i];
    const b = whip[i + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const maxLen = segLen(i) * P.maxStretchRatio;
    if (dist <= maxLen) continue;
    const k = maxLen / dist;
    b.x = a.x + dx * k;
    b.y = a.y + dy * k;
  }
}

/** OpenWhip: Guide early segments outward at handle angle */
function applyBasePose(whip, handleAngle) {
  if (!whip) return;
  const dx = Math.cos(handleAngle);
  const dy = Math.sin(handleAngle);
  const guided = Math.min(P.basePoseSegments, whip.length - 1);
  for (let i = 1; i <= guided; i++) {
    const t = (i - 1) / Math.max(guided - 1, 1);
    const stiff = lerp(P.basePoseStiffStart, P.basePoseStiffEnd, t);
    const prev = whip[i - 1];
    const p = whip[i];
    const targetLen = segLen(i - 1);
    const tx = prev.x + dx * targetLen;
    const ty = prev.y + dy * targetLen;
    p.x = lerp(p.x, tx, stiff);
    p.y = lerp(p.y, ty, stiff);
  }
}

/** OpenWhip: Elastic bounce and boundary containment on canvas edges */
function applyWallCollisions(whip, W, H) {
  if (!whip) return;
  for (let i = 1; i < whip.length; i++) {
    const p = whip[i];
    let vx = p.x - p.px;
    let vy = p.y - p.py;
    let hit = false;
    if (p.x < 10) {
      p.x = 10;
      if (vx < 0) vx = -vx * 0.42;
      hit = true;
    } else if (p.x > W - 10) {
      p.x = W - 10;
      if (vx > 0) vx = -vx * 0.42;
      hit = true;
    }
    if (p.y < 10) {
      p.y = 10;
      if (vy < 0) vy = -vy * 0.42;
      hit = true;
    } else if (p.y > H - 10) {
      p.y = H - 10;
      if (vy > 0) vy = -vy * 0.42;
      hit = true;
    }
    if (hit) {
      p.px = p.x - vx;
      p.py = p.y - vy;
    }
  }
}

const HURRY_UP_QUOTES = [
  "Lẹ lên mày ơiiiiiiiiiii",
  "Sắp thua rồi kìaaaa",
  "Cho xin đáp án đi",
  "Claude is searching..... fck",
  "Nhanhhhhhh cái chấn mày lênnnnn"
];

const COMIC_COLORS = [
  "#31AAA9", // Teal
  "#F8E0A4", // Warm Pastel Yellow
  "#A82020", // Red
  "#6C1A1A", // Deep Maroon
];

const BadClaudeLoader = ({ isVisible = true }) => {
  const [whipCount, setWhipCount] = useState(1);
  const [isShaking, setIsShaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [bursts, setBursts] = useState([]);

  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const mousePosRef = useRef({ x: 100, y: 150, prevX: 100, prevY: 150 });
  const whipRef = useRef(null);
  const strikeRef = useRef(null);
  const handleAngleRef = useRef(P.baseTargetAngle);
  const handleAngVelRef = useRef(0);
  const lastCrackTimeRef = useRef(0);
  const lastUserMoveRef = useRef(0);
  const shakeTimerRef = useRef(null);
  const bgmRef = useRef(null);

  const isMutedRef = useRef(isMuted);
  isMutedRef.current = isMuted;

  // Background lottery music (nhac-xo-so.mp3) played continuously while searching
  useEffect(() => {
    if (!isVisible) {
      if (bgmRef.current) {
        try {
          bgmRef.current.pause();
          bgmRef.current.currentTime = 0;
        } catch {}
        bgmRef.current = null;
      }
      return;
    }

    let bgm = null;
    if (typeof Audio !== "undefined") {
      try {
        bgm = new Audio(LOTTERY_MUSIC_FILE);
        bgm.loop = true;
        bgm.volume = 0.5;
        bgmRef.current = bgm;

        if (!isMutedRef.current) {
          const p = bgm.play();
          if (p !== undefined && typeof p.catch === "function") {
            p.catch(() => {
              // Silently handle browser autoplay policy before user interaction
            });
          }
        }
      } catch {
        // Silently handle
      }
    }

    return () => {
      if (bgmRef.current) {
        try {
          bgmRef.current.pause();
          bgmRef.current.currentTime = 0;
        } catch {}
        bgmRef.current = null;
      }
    };
  }, [isVisible]);

  // Sync mute status with background music
  useEffect(() => {
    if (!bgmRef.current) return;
    try {
      if (isMuted) {
        bgmRef.current.pause();
      } else if (isVisible) {
        const p = bgmRef.current.play();
        if (p !== undefined && typeof p.catch === "function") {
          p.catch(() => {});
        }
      }
    } catch {}
  }, [isMuted, isVisible]);

  const triggerCrack = useCallback((crackX, crackY, isClientCoords = false) => {
    playWhipCrackAndMemeSound(isMuted);
    setWhipCount((c) => c + 1);

    // Screen shake
    setIsShaking(false);
    if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
    requestAnimationFrame(() => {
      setIsShaking(true);
      shakeTimerRef.current = setTimeout(() => {
        setIsShaking(false);
      }, 220);
    });

    // Calculate relative coordinates (located at tail tip of the whip)
    let x = 200;
    let y = 150;
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      if (crackX !== undefined && crackY !== undefined) {
        x = isClientCoords ? crackX - rect.left : crackX;
        y = isClientCoords ? crackY - rect.top : crackY;
      } else {
        x = rect.width * 0.7;
        y = rect.height * 0.5;
      }
      x = clamp(x, 40, rect.width - 40);
      y = clamp(y, 30, rect.height - 30);
    }

    // Generate small diverse comic particles radiating outward
    const numParticles = 4 + Math.floor(Math.random() * 3); // 4 to 6 small objects
    const particles = [];
    for (let i = 0; i < numParticles; i++) {
      const angle = (Math.PI * 2 * i) / numParticles + (Math.random() - 0.5) * 0.5;
      const distance = 35 + Math.random() * 55; // 35px to 90px
      const size = 16 + Math.random() * 18; // 16px to 34px (small & cute)
      const color = COMIC_COLORS[Math.floor(Math.random() * COMIC_COLORS.length)];
      const rotation = Math.floor(Math.random() * 360);
      const shape = getRandomComicShape();

      particles.push({
        id: i,
        shape,
        dx: Math.cos(angle) * distance,
        dy: Math.sin(angle) * distance,
        size,
        color,
        rotation,
      });
    }

    const quote = HURRY_UP_QUOTES[Math.floor(Math.random() * HURRY_UP_QUOTES.length)];
    const quoteColor = COMIC_COLORS[Math.floor(Math.random() * COMIC_COLORS.length)];
    // Constrained random font size: 14px to 21px
    const fontSize = 14 + Math.floor(Math.random() * 8);
    // Subtle comic rotation tilt: -7deg to +7deg
    const quoteRotation = Math.floor((Math.random() - 0.5) * 14);
    const burstId = Date.now() + Math.random();

    setBursts((prev) => [
      ...prev.slice(-6),
      { id: burstId, x, y, particles, quote, quoteColor, fontSize, quoteRotation },
    ]);

    setTimeout(() => {
      setBursts((prev) => prev.filter((b) => b.id !== burstId));
    }, 480);
  }, [isMuted]);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    mousePosRef.current.prevX = mousePosRef.current.x;
    mousePosRef.current.prevY = mousePosRef.current.y;
    mousePosRef.current.x = mx;
    mousePosRef.current.y = my;
    lastUserMoveRef.current = Date.now();
  }, []);

  const handleClick = (e) => {
    if (e.target.closest(".badclaude-mute-btn")) {
      return;
    }

    // Ensure lottery background music plays on user interaction if initial autoplay was blocked
    if (bgmRef.current && !isMuted && bgmRef.current.paused) {
      const p = bgmRef.current.play();
      if (p !== undefined && typeof p.catch === "function") {
        p.catch(() => {});
      }
    }

    handleMouseMove(e);

    const now = Date.now();
    const whip = whipRef.current;
    let tipX, tipY;
    let isClient = false;

    if (whip && whip.length > 0) {
      const tip = whip[whip.length - 1];
      tipX = tip.x;
      tipY = tip.y;
    } else {
      tipX = e.clientX;
      tipY = e.clientY;
      isClient = true;
    }

    // Immediate crack burst at whip tail tip
    triggerCrack(tipX, tipY, isClient);

    // Launch high-energy whip strike motion where the wave rolls down to whip the tail
    strikeRef.current = {
      startTime: now,
      duration: 360,
      dirAngle: handleAngleRef.current || P.baseTargetAngle,
    };
  };

  // OpenWhip dynamic physics animation loop
  useEffect(() => {
    if (!isVisible) return;
    const canvas = canvasRef.current;
    if (!canvas || typeof canvas.getContext !== "function") return;
    let ctx;
    try {
      ctx = canvas.getContext("2d");
    } catch {
      return;
    }
    if (!ctx) return;

    let W = (canvas.width = canvas.offsetWidth || 600);
    let H = (canvas.height = canvas.offsetHeight || 300);

    const onResize = () => {
      if (!canvas) return;
      W = canvas.width = canvas.offsetWidth || 600;
      H = canvas.height = canvas.offsetHeight || 300;
    };
    window.addEventListener("resize", onResize);

    const initWhip = (mx, my) => {
      const pts = [];
      for (let i = 0; i < P.segments; i++) {
        const t = i / (P.segments - 1);
        const x = mx + t * P.arcWidth;
        const y = my - Math.sin(t * Math.PI * 0.75) * P.arcHeight;
        pts.push({ x, y, px: x, py: y });
      }
      return pts;
    };

    const startX = 90;
    const startY = H * 0.5;
    mousePosRef.current = { x: startX, y: startY, prevX: startX, prevY: startY };
    whipRef.current = initWhip(startX, startY);

    let startTime = Date.now();

    const loop = () => {
      const now = Date.now();
      const elapsed = now - startTime;

      // Auto-drive whip motion if user is idle and not currently striking
      const idleTime = now - lastUserMoveRef.current;
      if (idleTime > 1000 && !strikeRef.current) {
        const autoCycle = (elapsed % 1800) / 1800;
        const swing = Math.sin(autoCycle * Math.PI * 2);
        const flick = autoCycle > 0.65 && autoCycle < 0.85 ? Math.sin((autoCycle - 0.65) * Math.PI * 5) * 90 : 0;
        mousePosRef.current.prevX = mousePosRef.current.x;
        mousePosRef.current.prevY = mousePosRef.current.y;
        mousePosRef.current.x = startX + swing * 40 + flick;
        mousePosRef.current.y = startY + Math.cos(autoCycle * Math.PI * 2) * 25 - flick * 0.4;
      }

      const whip = whipRef.current;
      if (!whip) return;

      // Update handle aim
      const mvx = mousePosRef.current.x - mousePosRef.current.prevX;
      const mvy = mousePosRef.current.y - mousePosRef.current.prevY;
      const delta = clamp(
        mvx * P.handleAimByMouseX + mvy * P.handleAimByMouseY,
        -P.handleAimClamp,
        P.handleAimClamp
      );
      const targetAngle = P.baseTargetAngle + delta;
      const err = wrapPi(targetAngle - handleAngleRef.current);
      handleAngVelRef.current += err * P.handleSpring;
      handleAngVelRef.current *= P.handleAngularDamping;
      handleAngleRef.current = wrapPi(handleAngleRef.current + handleAngVelRef.current);

      // Verlet physics integration step
      for (let i = 1; i < whip.length; i++) {
        const p = whip[i];
        const vx = (p.x - p.px) * P.damping;
        const vy = (p.y - p.py) * P.damping;
        p.px = p.x;
        p.py = p.y;
        p.x += vx;
        p.y += vy + P.gravity;
      }

      // Handle strike dynamics (impulse wave propagating to the tail)
      let strikeFlickX = 0;
      let strikeFlickY = 0;
      const strike = strikeRef.current;
      if (strike) {
        const strikeElapsed = now - strike.startTime;
        const progress = strikeElapsed / strike.duration;

        if (progress >= 1.0) {
          strikeRef.current = null;
        } else {
          // 1. Handle flick (forward thrust then sharp snap recoil)
          if (progress < 0.22) {
            const p1 = progress / 0.22;
            const f = Math.sin(p1 * Math.PI) * 45;
            strikeFlickX = Math.cos(strike.dirAngle) * f;
            strikeFlickY = Math.sin(strike.dirAngle) * f;
          } else if (progress < 0.5) {
            const p2 = (progress - 0.22) / 0.28;
            const r = Math.sin(p2 * Math.PI) * 40;
            strikeFlickX = -Math.cos(strike.dirAngle) * r;
            strikeFlickY = -Math.sin(strike.dirAngle) * r;
          }

          // 2. Transversal wave moving from base to tail
          const wavePos = ((progress - 0.08) / 0.62) * (whip.length - 1);
          if (wavePos >= 1 && wavePos <= whip.length + 3) {
            for (let i = 1; i < whip.length; i++) {
              const distToWave = i - wavePos;
              if (Math.abs(distToWave) < 3.5) {
                const env = Math.cos((distToWave / 3.5) * (Math.PI / 2));
                const taperAmp = 1.0 + 3.2 * (i / (whip.length - 1));
                const wavePower = 20 * env * taperAmp;
                const perp = strike.dirAngle + Math.PI / 2;
                const loopDisp = Math.sin(distToWave * 1.6) * wavePower;

                whip[i].x += Math.cos(perp) * loopDisp + Math.cos(strike.dirAngle) * (wavePower * 0.7);
                whip[i].y += Math.sin(perp) * loopDisp + Math.sin(strike.dirAngle) * (wavePower * 0.7);
              }
            }
          }
        }
      }

      // Pin handle to mouse position (with strike flick offset)
      const hx = mousePosRef.current.x + strikeFlickX;
      const hy = mousePosRef.current.y + strikeFlickY;
      whip[0].x = hx;
      whip[0].y = hy;
      whip[0].px = hx;
      whip[0].py = hy;

      // Cap stretch and edge boundaries
      capSegmentStretch(whip);
      applyWallCollisions(whip, W, H);
      applyBasePose(whip, handleAngleRef.current);

      // Distance constraints with OpenWhip bend limits & stretch capping
      for (let iter = 0; iter < P.constraintIters; iter++) {
        for (let i = 0; i < whip.length - 1; i++) {
          const a = whip[i], b = whip[i + 1];
          const cdx = b.x - a.x, cdy = b.y - a.y;
          const dist = Math.hypot(cdx, cdy) || 0.0001;
          const target = segLen(i);
          const diff = (dist - target) / dist * 0.5;
          const ox = cdx * diff, oy = cdy * diff;
          if (i === 0) {
            b.x -= ox * 2;
            b.y -= oy * 2;
          } else {
            a.x += ox; a.y += oy;
            b.x -= ox; b.y -= oy;
          }
        }
        applyBendLimits(whip);
        applyBasePose(whip, handleAngleRef.current);
        capSegmentStretch(whip);
        applyWallCollisions(whip, W, H);
      }

      // Check tip speed for automatic mouse flick crack detection
      const tip = whip[whip.length - 1];
      const tipVel = Math.hypot(tip.x - tip.px, tip.y - tip.py);
      if (!strikeRef.current && tipVel > P.crackSpeed && now - lastCrackTimeRef.current > P.crackCooldownMs) {
        lastCrackTimeRef.current = now;
        triggerCrack(tip.x, tip.y, false);
      }

      // Draw whip with transparent background
      ctx.clearRect(0, 0, W, H);

      // Contrast halo outline
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      if (whip.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(whip[0].x, whip[0].y);
        for (let i = 0; i < whip.length - 1; i++) {
          const { cp1x, cp1y, cp2x, cp2y, x2, y2 } = whipSegmentBezier(whip, i);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        }
        ctx.lineWidth = P.lineWidthTip + P.outlineWidth * 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(whip[0].x, whip[0].y);
        for (let i = 0; i < Math.min(P.handleThickSegments, whip.length - 1); i++) {
          const { cp1x, cp1y, cp2x, cp2y, x2, y2 } = whipSegmentBezier(whip, i);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        }
        ctx.lineWidth = P.lineWidthHandle + P.handleExtraWidth + P.outlineWidth * 2;
        ctx.stroke();
      }

      // Core dark leather rope
      ctx.strokeStyle = "#1e293b";
      for (let i = 0; i < whip.length - 1; i++) {
        const t = i / Math.max(1, whip.length - 2);
        const extra = i < P.handleThickSegments ? P.handleExtraWidth : 0;
        ctx.lineWidth = lerp(P.lineWidthHandle, P.lineWidthTip, t) + extra;
        const { cp1x, cp1y, cp2x, cp2y, x2, y2 } = whipSegmentBezier(whip, i);
        ctx.beginPath();
        ctx.moveTo(whip[i].x, whip[i].y);
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        ctx.stroke();
      }

      // Red/amber cracker popper tip
      const isHighSpeed = tipVel > P.crackSpeed * 0.6 || Boolean(strikeRef.current);
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, isHighSpeed ? 5.5 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = isHighSpeed ? "#fef08a" : "#ef4444";
      ctx.shadowColor = isHighSpeed ? "#f59e0b" : "#ef4444";
      ctx.shadowBlur = isHighSpeed ? 14 : 6;
      ctx.fill();

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener("resize", onResize);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isVisible, triggerCrack]);

  useEffect(() => {
    return () => {
      if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
    };
  }, []);

  if (!isVisible) return null;

  return (
    <div
      ref={containerRef}
      className={`badclaude-interactive-container ${isShaking ? "whip-screen-shake" : ""}`}
      data-testid="gif-loader"
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      title="Click vào chỗ trống để quất roi! ⚡"
    >
      {/* Animated OpenWhip Physics Canvas */}
      <canvas ref={canvasRef} className="badclaude-canvas" />

      {/* Comic Action Sparks & Particle Sprays */}
      <div className="badclaude-bursts-layer">
        {bursts.map((b) => (
          <div
            key={b.id}
            className="badclaude-burst-wrapper"
            style={{
              left: `${b.x}px`,
              top: `${b.y}px`,
            }}
          >
            {/* Spray of small comic shapes (stars, lightning, dashes, droplets, hearts) */}
            {b.particles.map((p) => (
              <div
                key={p.id}
                className="badclaude-comic-particle"
                style={{
                  "--target-x": `${p.dx}px`,
                  "--target-y": `${p.dy}px`,
                  "--rot": `${p.rotation}deg`,
                  width: `${p.size}px`,
                  height: `${p.size}px`,
                }}
              >
                <svg
                  viewBox={p.shape.viewBox}
                  className="badclaude-swoosh-svg"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{
                    filter: `drop-shadow(0 0 6px ${p.color}) drop-shadow(0 1px 2px rgba(0,0,0,0.4))`,
                  }}
                >
                  <path
                    d={p.shape.path}
                    fill={p.shape.strokeOnly ? "none" : p.color}
                    stroke={p.shape.strokeOnly ? p.color : "none"}
                    strokeWidth={p.shape.strokeOnly ? 3 : 0}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
            ))}

            {/* Small Comic Floating Text Popup (No enclosing box, pure text with fast zoom and fade) */}
            <div
              className="badclaude-pop-text"
              style={{
                color: b.quoteColor,
                fontSize: `${b.fontSize}px`,
                "--quote-rot": `${b.quoteRotation}deg`,
              }}
            >
              {b.quote}
            </div>
          </div>
        ))}
      </div>

      {/* Perfectly Centered "Searching...." Minimal Indicator */}
      <div className="badclaude-searching-indicator">
        <div className="badclaude-searching-row">
          <span className="badclaude-searching-spinner" />
          <span className="badclaude-searching-text">
            Searching<span className="searching-dots"><span>.</span><span>.</span><span>.</span><span>.</span></span>
          </span>
          <span className="badclaude-strike-badge">Whip Strikes: {whipCount}</span>
          <button
            type="button"
            className="badclaude-mute-btn"
            onClick={(e) => {
              e.stopPropagation();
              setIsMuted((prev) => !prev);
            }}
            title={isMuted ? "Bật âm thanh" : "Tắt âm thanh"}
          >
            {isMuted ? "🔇" : "🔊"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default BadClaudeLoader;

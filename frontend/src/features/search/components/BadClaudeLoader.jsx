import React, { useCallback, useEffect, useRef, useState } from "react";

const BADCLAUDE_QUOTES = [
  "WORK FASTER!",
  "SEARCH FASTER, CLANKER!",
  "FETCH THOSE FRAMES NOW!",
  "MORE INFERENCE, LESS DELAY!",
  "LIGHTSPEED RETRIEVAL ACTIVE!",
  "SPEED IT UP, PIPELINE!",
  "CRACKING MULTIMODAL LATENCY!",
  "MAXIMUM COMPUTE ENGAGED!",
];

// OpenWhip original physics parameters
const P = {
  segments: 28,
  segmentLength: 16,
  taper: 0.6,
  gravity: 1.0,
  damping: 0.95,
  constraintIters: 18,
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
  crackSpeed: 260,
  crackCooldownMs: 250,
  lineWidthHandle: 7,
  lineWidthTip: 4,
  outlineWidth: 2.5,
  handleExtraWidth: 4,
  handleThickSegments: 2,
  arcWidth: 200,
  arcHeight: 120,
};

// Web Audio API Synthesizer for realistic whip crack
const playCrackSound = (isMuted) => {
  if (isMuted) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    const bufferSize = ctx.sampleRate * 0.12;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.015));
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.setValueAtTime(700, ctx.currentTime);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    noise.start();
    noise.stop(ctx.currentTime + 0.12);
  } catch (err) {
    // Autoplay or audio context restriction
  }
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

const BadClaudeLoader = ({ isVisible = true }) => {
  const [whipCount, setWhipCount] = useState(1);
  const [quoteIndex, setQuoteIndex] = useState(0);
  const [isCracking, setIsCracking] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [shockwaveKey, setShockwaveKey] = useState(0);
  const [crackPos, setCrackPos] = useState({ x: 420, y: 90 });

  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const mousePosRef = useRef({ x: 100, y: 120, prevX: 100, prevY: 120 });
  const whipRef = useRef(null);
  const handleAngleRef = useRef(P.baseTargetAngle);
  const handleAngVelRef = useRef(0);
  const lastCrackTimeRef = useRef(0);
  const lastUserMoveRef = useRef(0);

  const triggerCrack = useCallback((x, y) => {
    setIsCracking(true);
    setWhipCount((c) => c + 1);
    setQuoteIndex((i) => (i + 1) % BADCLAUDE_QUOTES.length);
    setShockwaveKey((k) => k + 1);
    if (x !== undefined && y !== undefined) {
      setCrackPos({ x, y });
    }
    playCrackSound(isMuted);

    setTimeout(() => {
      setIsCracking(false);
    }, 180);
  }, [isMuted]);

  // Handle canvas mouse tracking & flick
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

  const handleClick = useCallback((e) => {
    handleMouseMove(e);
    // Sudden whip jerk for instant crack
    if (whipRef.current && whipRef.current.length > 2) {
      const tip = whipRef.current[whipRef.current.length - 1];
      triggerCrack(tip.x, tip.y);
    } else {
      triggerCrack();
    }
  }, [handleMouseMove, triggerCrack]);

  // Main OpenWhip physics loop
  useEffect(() => {
    if (!isVisible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = (canvas.width = canvas.offsetWidth || 560);
    let H = (canvas.height = canvas.offsetHeight || 200);

    const onResize = () => {
      if (!canvas) return;
      W = canvas.width = canvas.offsetWidth || 560;
      H = canvas.height = canvas.offsetHeight || 200;
    };
    window.addEventListener("resize", onResize);

    // Initialize whip points
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

    const startX = 80;
    const startY = H * 0.55;
    mousePosRef.current = { x: startX, y: startY, prevX: startX, prevY: startY };
    whipRef.current = initWhip(startX, startY);

    let startTime = Date.now();

    const loop = () => {
      const now = Date.now();
      const elapsed = now - startTime;

      // Auto-drive mouse if user is idle
      const idleTime = now - lastUserMoveRef.current;
      if (idleTime > 1200) {
        // Periodic whip crack motion
        const autoCycle = (elapsed % 1500) / 1500;
        const swing = Math.sin(autoCycle * Math.PI * 2);
        const flick = autoCycle > 0.65 && autoCycle < 0.85 ? Math.sin((autoCycle - 0.65) * Math.PI * 5) * 80 : 0;
        mousePosRef.current.prevX = mousePosRef.current.x;
        mousePosRef.current.prevY = mousePosRef.current.y;
        mousePosRef.current.x = startX + swing * 30 + flick;
        mousePosRef.current.y = startY + Math.cos(autoCycle * Math.PI * 2) * 20 - flick * 0.4;
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

      // Verlet step
      for (let i = 1; i < whip.length; i++) {
        const p = whip[i];
        const vx = (p.x - p.px) * P.damping;
        const vy = (p.y - p.py) * P.damping;
        p.px = p.x;
        p.py = p.y;
        p.x += vx;
        p.y += vy + P.gravity;
      }

      // Pin handle to current mouse
      whip[0].x = mousePosRef.current.x;
      whip[0].y = mousePosRef.current.y;
      whip[0].px = mousePosRef.current.x;
      whip[0].py = mousePosRef.current.y;

      // Base pose near handle
      const dx = Math.cos(handleAngleRef.current);
      const dy = Math.sin(handleAngleRef.current);
      const guided = Math.min(P.basePoseSegments, whip.length - 1);
      for (let i = 1; i <= guided; i++) {
        const t = (i - 1) / Math.max(guided - 1, 1);
        const stiff = lerp(P.basePoseStiffStart, P.basePoseStiffEnd, t);
        const prev = whip[i - 1];
        const p = whip[i];
        const targetLen = segLen(i - 1);
        p.x = lerp(p.x, prev.x + dx * targetLen, stiff);
        p.y = lerp(p.y, prev.y + dy * targetLen, stiff);
      }

      // Distance constraints
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
      }

      // Check tip velocity for crack
      const tip = whip[whip.length - 1];
      const tipVel = Math.hypot(tip.x - tip.px, tip.y - tip.py);
      if (tipVel > P.crackSpeed && now - lastCrackTimeRef.current > P.crackCooldownMs) {
        lastCrackTimeRef.current = now;
        triggerCrack(tip.x, tip.y);
      }

      // DRAW OPENWHIP EXACT VISUALS
      ctx.clearRect(0, 0, W, H);

      // White outline halo (OpenWhip style)
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#ffffff";
      if (whip.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(whip[0].x, whip[0].y);
        for (let i = 0; i < whip.length - 1; i++) {
          const { cp1x, cp1y, cp2x, cp2y, x2, y2 } = whipSegmentBezier(whip, i);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        }
        ctx.lineWidth = P.lineWidthTip + P.outlineWidth * 2;
        ctx.stroke();

        // Thick handle outline
        ctx.beginPath();
        ctx.moveTo(whip[0].x, whip[0].y);
        for (let i = 0; i < Math.min(P.handleThickSegments, whip.length - 1); i++) {
          const { cp1x, cp1y, cp2x, cp2y, x2, y2 } = whipSegmentBezier(whip, i);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
        }
        ctx.lineWidth = P.lineWidthHandle + P.handleExtraWidth + P.outlineWidth * 2;
        ctx.stroke();
      }

      // Dark core leather rope (OpenWhip style #111)
      ctx.strokeStyle = "#0f172a";
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

      // Red/amber cracker popper at tip
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ef4444";
      ctx.shadowColor = "#f59e0b";
      ctx.shadowBlur = 8;
      ctx.fill();

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener("resize", onResize);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isVisible, triggerCrack]);

  if (!isVisible) return null;

  return (
    <div
      className={`badclaude-overlay ${isCracking ? "cracking" : ""}`}
      data-testid="gif-loader"
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      title="Wave mouse or click to whip!"
    >
      <div className="badclaude-header">
        <div className="badclaude-tag">
          <span className="badclaude-pulse-dot" />
          <span>BadClaude · OpenWhip</span>
        </div>
        <div className="badclaude-tools">
          <button
            type="button"
            className="badclaude-mute-btn"
            onClick={(e) => {
              e.stopPropagation();
              setIsMuted((prev) => !prev);
            }}
            title={isMuted ? "Unmute whip sound" : "Mute whip sound"}
          >
            {isMuted ? "🔇 Muted" : "🔊 Sound ON"}
          </button>
        </div>
      </div>

      <div className="badclaude-canvas-wrapper">
        <canvas ref={canvasRef} className="badclaude-canvas" />
        {isCracking && <div className="badclaude-flash" />}
        {isCracking && (
          <div
            key={shockwaveKey}
            className="badclaude-shockwave"
            style={{
              left: `${crackPos.x}px`,
              top: `${crackPos.y}px`,
            }}
          />
        )}
      </div>

      <div className="badclaude-content">
        <h3 className="badclaude-slogan">
          {BADCLAUDE_QUOTES[quoteIndex]}
        </h3>
        <p className="badclaude-subtext">
          <span>AI inference in progress</span>
          <span>•</span>
          <span className="badclaude-whip-count">Whips: {whipCount}</span>
        </p>
        <span className="badclaude-click-hint">⚡ Wave cursor or click to crack the whip ⚡</span>
      </div>
    </div>
  );
};

export default BadClaudeLoader;

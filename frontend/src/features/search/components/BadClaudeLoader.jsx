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

// Simple Web Audio API Synthesizer for realistic whip crack sound
const playWhipSound = (isMuted) => {
  if (isMuted) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    const bufferSize = ctx.sampleRate * 0.15; // 150ms burst
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      // White noise with sharp decay
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.02));
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.setValueAtTime(800, ctx.currentTime);

    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0.35, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.12);

    noise.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(ctx.destination);

    noise.start();
    noise.stop(ctx.currentTime + 0.15);
  } catch (err) {
    // Audio playback error or autoplay policy restricted
  }
};

const BadClaudeLoader = ({ isVisible = true }) => {
  const [whipCount, setWhipCount] = useState(1);
  const [quoteIndex, setQuoteIndex] = useState(0);
  const [isCracking, setIsCracking] = useState(false);
  const [isMuted, setIsMuted] = useState(true); // Default muted so it doesn't startle users
  const [shockwaveKey, setShockwaveKey] = useState(0);

  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const startTimeRef = useRef(Date.now());
  const crackPosRef = useRef({ x: 0, y: 0 });

  const triggerCrack = useCallback(() => {
    setIsCracking(true);
    setWhipCount((prev) => prev + 1);
    setQuoteIndex((prev) => (prev + 1) % BADCLAUDE_QUOTES.length);
    setShockwaveKey((prev) => prev + 1);
    playWhipSound(isMuted);

    setTimeout(() => {
      setIsCracking(false);
    }, 200);
  }, [isMuted]);

  // Whip physics and drawing on Canvas
  useEffect(() => {
    if (!isVisible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = (canvas.width = canvas.offsetWidth || 560);
    let height = (canvas.height = canvas.offsetHeight || 180);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth || 560;
      height = canvas.height = canvas.offsetHeight || 180;
    };
    window.addEventListener("resize", handleResize);

    const cycleDuration = 1400; // ms per whip cycle
    let lastCycleIndex = 0;

    const render = () => {
      const now = Date.now();
      const elapsed = now - startTimeRef.current;
      const progress = (elapsed % cycleDuration) / cycleDuration;
      const currentCycleIndex = Math.floor(elapsed / cycleDuration);

      // Auto-trigger crack at whip apex (progress ~ 0.65)
      if (currentCycleIndex !== lastCycleIndex && progress > 0.6) {
        lastCycleIndex = currentCycleIndex;
        triggerCrack();
      }

      ctx.clearRect(0, 0, width, height);

      // Handle anchor point on left
      const startX = 40;
      const startY = height * 0.55;

      // Calculate whip segments with wave physics
      const points = [];
      const numSegments = 24;

      // Whip whip-wave dynamic calculation
      const wavePhase = progress * Math.PI * 2;
      const snapIntensity = Math.sin(wavePhase);

      const targetX = width - 80;
      const targetY = height * 0.5 + Math.sin(progress * Math.PI * 4) * 20;

      crackPosRef.current = { x: targetX, y: targetY };

      for (let i = 0; i <= numSegments; i++) {
        const t = i / numSegments;
        const x = startX + (targetX - startX) * t;
        // Traveling wave sine displacement
        const wave = Math.sin(wavePhase - t * 6) * (1 - t) * 45 * snapIntensity;
        const curve = Math.sin(t * Math.PI) * 25 * (progress > 0.5 ? -1 : 1);
        const y = startY + (targetY - startY) * t + wave + curve;
        points.push({ x, y });
      }

      // Draw Whip Trail / Glow
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
      ctx.lineWidth = 9;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();

      // Draw Main Whip Cord (gradient from handle to tip)
      const grad = ctx.createLinearGradient(startX, startY, targetX, targetY);
      grad.addColorStop(0, "#f59e0b"); // Leather handle amber
      grad.addColorStop(0.3, "#0284c7"); // Electric cyan blue
      grad.addColorStop(1, "#38bdf8"); // Glowing tip

      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.strokeStyle = grad;
      ctx.lineWidth = 3.5;
      ctx.stroke();

      // Draw Whip Handle
      ctx.beginPath();
      ctx.moveTo(startX - 15, startY + 6);
      ctx.lineTo(startX + 10, startY - 4);
      ctx.strokeStyle = "#d97706";
      ctx.lineWidth = 8;
      ctx.stroke();

      // Draw Whip Tip Spark
      const tip = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 12;
      ctx.fill();

      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isVisible, triggerCrack]);

  if (!isVisible) return null;

  return (
    <div
      className={`badclaude-overlay ${isCracking ? "cracking" : ""}`}
      data-testid="gif-loader"
      onClick={triggerCrack}
      title="Click to Crack the Whip!"
    >
      <div className="badclaude-header">
        <div className="badclaude-tag">
          <span className="badclaude-pulse-dot" />
          <span>BadClaude Accelerator</span>
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
              left: `${crackPosRef.current.x || 480}px`,
              top: `${crackPosRef.current.y || 90}px`,
            }}
          />
        )}
      </div>

      <div className="badclaude-content">
        <h3 className="badclaude-slogan">
          {BADCLAUDE_QUOTES[quoteIndex]}
        </h3>
        <p className="badclaude-subtext">
          <span>Retrieval in progress</span>
          <span>•</span>
          <span className="badclaude-whip-count">Whip Strikes: {whipCount}</span>
        </p>
        <span className="badclaude-click-hint">⚡ Click anywhere to whip faster ⚡</span>
      </div>
    </div>
  );
};

export default BadClaudeLoader;

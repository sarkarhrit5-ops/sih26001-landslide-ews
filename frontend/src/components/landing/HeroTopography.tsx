import React, { useEffect, useRef } from 'react';

export const HeroTopography: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 500);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    // Generate terrain contour lines
    const numLines = 14;
    let time = 0;

    // Rain particles
    const particles = Array.from({ length: 40 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      length: 12 + Math.random() * 15,
      speed: 1.2 + Math.random() * 2.0,
      opacity: 0.15 + Math.random() * 0.35
    }));

    const render = () => {
      time += 0.008;
      ctx.clearRect(0, 0, width, height);

      // Draw subtle topographic contours
      ctx.lineWidth = 1;
      for (let i = 0; i < numLines; i++) {
        const radiusBase = (i + 1) * (Math.min(width, height) / (numLines * 1.5));
        const cx = width * 0.65;
        const cy = height * 0.45;

        ctx.beginPath();
        const steps = 80;
        for (let j = 0; j <= steps; j++) {
          const angle = (j / steps) * Math.PI * 2;
          const noise =
            Math.sin(angle * 4 + time + i) * 18 +
            Math.cos(angle * 2 - time * 0.5) * 12 +
            Math.sin(angle * 7 + time * 1.2) * 6;

          const r = radiusBase + noise;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r * 0.6; // Slight perspective tilt

          if (j === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();

        const alpha = 0.08 + (1 - i / numLines) * 0.15;
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
        ctx.stroke();

        // Highlight high hazard elevation contour (e.g. index 4)
        if (i === 4) {
          ctx.strokeStyle = `rgba(245, 158, 11, 0.3)`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.lineWidth = 1;
        }
      }

      // Draw subtle rainfall particles
      ctx.lineWidth = 1;
      particles.forEach((p) => {
        p.y += p.speed;
        p.x -= p.speed * 0.25; // Wind angle
        if (p.y > height) {
          p.y = -20;
          p.x = Math.random() * width;
        }

        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.length * 0.25, p.y + p.length);
        ctx.strokeStyle = `rgba(148, 163, 184, ${p.opacity})`;
        ctx.stroke();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-2xl overflow-hidden border border-slate-800/80 bg-slate-950/60 shadow-2xl">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      
      {/* Overlay elevation grid marker badges */}
      <div className="absolute top-6 left-6 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-700/80 text-[11px] font-mono text-emerald-400 backdrop-blur flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
        EAST SIKKIM PILOT AOI (27.0°N - 28.1°N)
      </div>

      <div className="absolute bottom-6 right-6 px-3 py-1.5 rounded bg-slate-900/90 border border-slate-700/80 text-[11px] font-mono text-slate-300 backdrop-blur">
        ELEVATION: 30m COPERNICUS GLO-30 DEM
      </div>
    </div>
  );
};

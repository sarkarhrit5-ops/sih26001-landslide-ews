/**
 * BhūRaksha brand mark — a protective shield enclosing a contoured summit.
 * The three ridge lines echo the topographic contours that are the model's
 * primary real input (Copernicus GLO-30 terrain derivatives).
 */

interface BrandMarkProps {
  size?: number;
  className?: string;
  /** When true, renders the slow contour-draw animation on mount. */
  animated?: boolean;
}

export function BrandMark({ size = 40, className = '', animated = false }: BrandMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      className={className}
      role="img"
      aria-label="BhūRaksha mark"
    >
      <defs>
        <linearGradient id="brk-shield" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#0f2a24" />
          <stop offset="100%" stopColor="#0a0d12" />
        </linearGradient>
      </defs>
      {/* Shield */}
      <path
        d="M24 2.5 41.5 8.5 V22 C41.5 33.5 33.7 41.8 24 45.5 C14.3 41.8 6.5 33.5 6.5 22 V8.5 Z"
        fill="url(#brk-shield)"
        stroke="#10b981"
        strokeWidth="1.6"
      />
      {/* Summit + contour ridges */}
      <g
        stroke="#34d399"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
        className={animated ? 'brk-draw' : ''}
      >
        <path d="M13 30 L21 18 L27 25 L31 20 L36 30" />
        <path d="M15 34 L21.5 24 L27 30 L30.5 26 L34 34" opacity="0.55" />
      </g>
      {/* Scan dot at summit */}
      <circle cx="21" cy="18" r="1.7" fill="#22d3ee" />
    </svg>
  );
}

/** Full lock-up: mark + wordmark, used in headers and the hero. */
interface WordmarkProps {
  size?: number;
  showDevanagari?: boolean;
  className?: string;
}

export function BrandLockup({ size = 34, showDevanagari = true, className = '' }: WordmarkProps) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <BrandMark size={size} />
      <div className="leading-none">
        <div className="flex items-baseline gap-2">
          <span
            className="font-bold tracking-tight text-slate-50"
            style={{ fontSize: size * 0.62 }}
          >
            Bhū<span className="text-emerald-400">Raksha</span>
          </span>
          {showDevanagari && (
            <span className="text-slate-500 font-medium" style={{ fontSize: size * 0.42 }}>
              भू-रक्षा
            </span>
          )}
        </div>
        <div
          className="font-mono uppercase text-slate-500 tracking-[0.25em]"
          style={{ fontSize: size * 0.2 }}
        >
          NER Landslide Early-Warning
        </div>
      </div>
    </div>
  );
}

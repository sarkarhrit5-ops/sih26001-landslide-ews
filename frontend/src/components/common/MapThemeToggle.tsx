/**
 * Compact Dark / Light basemap toggle, mounted in the top-right corner of a map.
 *
 * Presentation only: it reports the chosen theme upward and the map swaps the
 * tile treatment. It never touches prediction cells, AOI rectangles, markers,
 * labels, zoom controls or any backend/model behaviour. Each map owns its own
 * theme state, so toggling one console's map leaves the others alone.
 */
import { Moon, Sun } from 'lucide-react';
import { MAP_THEMES, MAP_THEME_LABEL } from '../pilot/mapTheme';
import type { MapTheme } from '../pilot/mapTheme';
import { cn } from './ui';

interface MapThemeToggleProps {
  theme: MapTheme;
  onChange: (theme: MapTheme) => void;
  className?: string;
}

const ICONS = { dark: Moon, light: Sun } as const;

export function MapThemeToggle({ theme, onChange, className }: MapThemeToggleProps) {
  return (
    <div
      role="group"
      aria-label="Basemap theme"
      className={cn(
        'absolute right-3 top-3 z-[400] flex items-center gap-0.5 rounded-lg border border-slate-700/80 bg-slate-950/85 p-0.5 shadow-lg backdrop-blur',
        className,
      )}
    >
      {MAP_THEMES.map((option) => {
        const Icon = ICONS[option];
        const active = option === theme;
        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-pressed={active}
            title={`${MAP_THEME_LABEL[option]} basemap`}
            className={cn(
              'flex items-center gap-1 rounded-md px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider transition-colors duration-200',
              active
                ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/40'
                : 'text-slate-400 hover:bg-slate-800/70 hover:text-slate-200',
            )}
          >
            <Icon className="h-3 w-3" aria-hidden="true" />
            <span>{MAP_THEME_LABEL[option]}</span>
          </button>
        );
      })}
    </div>
  );
}

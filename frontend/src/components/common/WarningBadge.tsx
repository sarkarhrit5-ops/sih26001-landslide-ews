import React from 'react';
import type { WarningLevel } from '../../services/api';
import { ShieldCheck, AlertTriangle, AlertOctagon, Flame } from 'lucide-react';

interface WarningBadgeProps {
  level: WarningLevel;
  size?: 'sm' | 'md' | 'lg';
  showPattern?: boolean;
}

export const WarningBadge: React.FC<WarningBadgeProps> = ({ level, size = 'md', showPattern = true }) => {
  const getConfig = () => {
    switch (level) {
      case 'EXTREME':
        return {
          label: 'EXTREME RISK',
          colorBg: 'bg-red-950/80 border-red-500/80 text-red-200',
          badgeBg: 'bg-red-600',
          icon: <Flame className="w-4 h-4 text-red-300 animate-pulse" />,
          pattern: '/// EXTREME ///',
          barWidth: 'w-full'
        };
      case 'HIGH':
        return {
          label: 'HIGH RISK',
          colorBg: 'bg-orange-950/80 border-orange-500/80 text-orange-200',
          badgeBg: 'bg-orange-600',
          icon: <AlertOctagon className="w-4 h-4 text-orange-300" />,
          pattern: '// HIGH //',
          barWidth: 'w-3/4'
        };
      case 'MEDIUM':
        return {
          label: 'MEDIUM RISK',
          colorBg: 'bg-amber-950/80 border-amber-500/80 text-amber-200',
          badgeBg: 'bg-amber-600',
          icon: <AlertTriangle className="w-4 h-4 text-amber-300" />,
          pattern: '/ MODERATE /',
          barWidth: 'w-1/2'
        };
      case 'LOW':
      default:
        return {
          label: 'LOW RISK',
          colorBg: 'bg-emerald-950/80 border-emerald-500/80 text-emerald-200',
          badgeBg: 'bg-emerald-600',
          icon: <ShieldCheck className="w-4 h-4 text-emerald-300" />,
          pattern: '- STABLE -',
          barWidth: 'w-1/4'
        };
    }
  };

  const config = getConfig();

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1.5',
    md: 'text-xs px-3 py-1.5 gap-2 font-medium tracking-wide',
    lg: 'text-sm px-4 py-2 gap-2.5 font-semibold tracking-wider'
  }[size];

  return (
    <div className={`inline-flex flex-col gap-1`}>
      <div className={`inline-flex items-center rounded-md border ${config.colorBg} ${sizeClasses} shadow-md`}>
        <span className="flex items-center">{config.icon}</span>
        <span>{config.label}</span>
        {showPattern && (
          <span className="ml-1.5 font-mono text-[10px] opacity-70 border-l border-current/30 pl-1.5">
            {config.pattern}
          </span>
        )}
      </div>
      {showPattern && size !== 'sm' && (
        <div className="w-full bg-slate-800/80 h-1.5 rounded-full overflow-hidden border border-slate-700/50">
          <div className={`h-full ${config.badgeBg} ${config.barWidth} transition-all duration-500`} />
        </div>
      )}
    </div>
  );
};

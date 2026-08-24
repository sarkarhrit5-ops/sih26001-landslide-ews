import { useEffect, useState } from 'react';
import { WelcomePage } from './pages/WelcomePage';
import { NERDashboard } from './pages/NERDashboard';
import { SikkimDashboard } from './pages/SikkimDashboard';

export type ActivePage = 'welcome' | 'ner' | 'sikkim';

export function App() {
  const [page, setPage] = useState<ActivePage>('welcome');

  // Reset scroll position on navigation so each view opens at the top.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [page]);

  return (
    <div className="min-h-screen w-full bg-[#0a0d12] text-slate-100">
      {page === 'welcome' && (
        <WelcomePage
          onEnterConsole={() => setPage('ner')}
          onOpenSikkim={() => setPage('sikkim')}
        />
      )}
      {page === 'ner' && (
        <NERDashboard onBack={() => setPage('welcome')} onOpenSikkim={() => setPage('sikkim')} />
      )}
      {page === 'sikkim' && <SikkimDashboard onBack={() => setPage('ner')} />}
    </div>
  );
}

export default App;

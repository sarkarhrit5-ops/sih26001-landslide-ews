import { useState } from 'react';
import { LandingPage } from './components/landing/LandingPage';
import { DashboardPage } from './components/dashboard/DashboardPage';

export type ActivePage = 'landing' | 'dashboard';

export function App() {
  const [currentPage, setCurrentPage] = useState<ActivePage>('landing');

  return (
    <div className="w-full min-h-screen bg-[#0a0d12] text-slate-100">
      {currentPage === 'landing' ? (
        <LandingPage onNavigateToDashboard={() => setCurrentPage('dashboard')} />
      ) : (
        <DashboardPage onNavigateToLanding={() => setCurrentPage('landing')} />
      )}
    </div>
  );
}

export default App;

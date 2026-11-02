import React, { lazy, Suspense, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { CommunityDashboard } from './CommunityDashboard';
import './styles.css';
const LegacyWorkspace = lazy(() => import('./App').then(module => ({ default: module.App })));
function PublicEntry() {
  const [legacy, setLegacy] = useState(false);
  return legacy ? <Suspense fallback={<p role="status">Loading review workspace?</p>}><LegacyWorkspace initialResearch={false} /></Suspense> : <CommunityDashboard onLegacy={() => setLegacy(true)} />;
}
createRoot(document.getElementById('root') as HTMLElement).render(<React.StrictMode><PublicEntry /></React.StrictMode>);

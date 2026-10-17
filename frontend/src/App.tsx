import { Navigate, Route, Routes } from 'react-router-dom';
import { NavBar } from './components/layout/NavBar';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { ImplementedPage } from './pages/ImplementedPage';
import { PlanPage } from './pages/PlanPage';
import { ProgressPage } from './pages/ProgressPage';
import { ReferencesPage } from './pages/ReferencesPage';
import { StudiedPage } from './pages/StudiedPage';

export default function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<Navigate to="/plan" replace />} />
        <Route path="/plan" element={<PlanPage />} />
        <Route path="/architecture" element={<ArchitecturePage />} />
        <Route path="/implemented" element={<ImplementedPage />} />
        <Route path="/studied" element={<StudiedPage />} />
        <Route path="/references" element={<ReferencesPage />} />
        <Route path="/progress" element={<ProgressPage />} />
        <Route path="*" element={<Navigate to="/plan" replace />} />
      </Routes>
    </>
  );
}

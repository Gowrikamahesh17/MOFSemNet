// The whole web app is the single page public/mofworld.html (landing page, Live Demo, Research Explorer).
// Every other address, including the old console pages (/plan, /architecture, ...), forwards to it.
export default function App() {
  window.location.replace('/mofworld.html' + window.location.hash);
  return null;
}

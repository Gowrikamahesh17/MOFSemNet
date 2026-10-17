// Dev-only visual verification script — not part of the app build. Launches headless
// Chromium against the running dev server, visits each route, and screenshots it at a few
// sizes so layout/contrast/overlap issues can be caught before showing a human.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE_URL = process.env.SCREENSHOT_BASE_URL ?? 'http://localhost:5173';
const OUT_DIR = process.env.SCREENSHOT_OUT_DIR ?? '/tmp/mofsemnet-screenshots';

const VIEWPORTS = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'projector', width: 1920, height: 1080 },
  { name: 'mobile', width: 390, height: 844 },
];

const ROUTES = process.argv.slice(2).length ? process.argv.slice(2) : ['/architecture'];

mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch();
for (const viewport of VIEWPORTS) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height } });
  for (const route of ROUTES) {
    await page.goto(`${BASE_URL}${route}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    const safeName = route.replace(/\//g, '_') || 'root';
    const path = `${OUT_DIR}/${safeName}__${viewport.name}.png`;
    await page.screenshot({ path, fullPage: false });
    console.log('saved', path);
  }
  await page.close();
}
await browser.close();

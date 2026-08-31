/**
 * Unit tests for the pure basemap-theme logic (no Leaflet, no DOM).
 *
 * These pin the behaviour the toggle depends on: Dark is the default, the two
 * themes read the SAME key-free tile endpoint with the SAME attribution (so no
 * mode can lose attribution or silently need an API key), and a theme swap ends
 * with exactly one treatment class applied.
 */
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import {
  BASEMAP_ATTRIBUTION,
  BASEMAP_CLASS,
  BASEMAP_THEME_CLASS,
  BASEMAP_TILE_URL,
  DEFAULT_MAP_THEME,
  MAP_THEMES,
  applyBasemapThemeClass,
  basemapClassName,
  isMapTheme,
  otherMapTheme,
} from '../components/pilot/mapTheme';
import type { MapTheme } from '../components/pilot/mapTheme';

/** Minimal DOMTokenList stand-in that records the final class set. */
function fakeClassList(initial: string[] = []) {
  const tokens = new Set(initial);
  return {
    tokens,
    add(...values: string[]) {
      values.forEach((v) => tokens.add(v));
    },
    remove(...values: string[]) {
      values.forEach((v) => tokens.delete(v));
    },
  };
}

test('dark is the default so the existing console design is preserved', () => {
  assert.equal(DEFAULT_MAP_THEME, 'dark');
  assert.deepEqual([...MAP_THEMES], ['dark', 'light']);
});

test('both themes use the same key-free tile endpoint and attribution', () => {
  assert.equal(BASEMAP_TILE_URL, 'https://tile.openstreetmap.org/{z}/{x}/{y}.png');
  assert.ok(!BASEMAP_TILE_URL.includes('cartocdn'));
  assert.ok(!/api[_-]?key|apikey|access[_-]?token/i.test(BASEMAP_TILE_URL));
  assert.ok(BASEMAP_ATTRIBUTION.includes('OpenStreetMap'));
});

test('a new layer carries the marker class plus exactly one theme class', () => {
  for (const theme of MAP_THEMES) {
    const name = basemapClassName(theme);
    assert.ok(name.includes(BASEMAP_CLASS));
    assert.ok(name.includes(BASEMAP_THEME_CLASS[theme]));
    const other = BASEMAP_THEME_CLASS[otherMapTheme(theme)];
    assert.ok(!name.split(' ').includes(other));
  }
});

test('applying a theme leaves exactly one treatment class behind', () => {
  const list = fakeClassList([BASEMAP_CLASS, BASEMAP_THEME_CLASS.dark]);
  applyBasemapThemeClass(list, 'light');
  assert.ok(list.tokens.has(BASEMAP_THEME_CLASS.light));
  assert.ok(!list.tokens.has(BASEMAP_THEME_CLASS.dark));
  applyBasemapThemeClass(list, 'dark');
  assert.ok(list.tokens.has(BASEMAP_THEME_CLASS.dark));
  assert.ok(!list.tokens.has(BASEMAP_THEME_CLASS.light));
  assert.ok(list.tokens.has(BASEMAP_CLASS));
  assert.equal(list.tokens.size, 2);
});

test('repeated application is idempotent', () => {
  const list = fakeClassList();
  applyBasemapThemeClass(list, 'dark');
  applyBasemapThemeClass(list, 'dark');
  assert.equal(list.tokens.size, 2);
});

test('otherMapTheme round-trips and isMapTheme rejects junk', () => {
  assert.equal(otherMapTheme('dark'), 'light');
  assert.equal(otherMapTheme('light'), 'dark');
  for (const theme of MAP_THEMES) {
    assert.equal(otherMapTheme(otherMapTheme(theme)), theme);
    assert.ok(isMapTheme(theme as MapTheme));
  }
  assert.ok(!isMapTheme('sepia'));
  assert.ok(!isMapTheme(null));
  assert.ok(!isMapTheme(undefined));
});

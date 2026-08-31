/**
 * Minimal ESM resolve hook so node:test can load the app's TypeScript modules.
 *
 * The app source uses Vite/bundler-style extensionless imports
 * (`from '../../data/nerStates'`). Node's ESM resolver requires an explicit
 * extension, so this hook retries a failed relative resolution with `.ts`
 * appended. It changes nothing about the modules themselves — it exists only so
 * the same files the app ships can be exercised directly by `node --test`.
 *
 * Usage:
 *   node --experimental-strip-types --import ./src/__tests__/tsResolve.mjs \
 *        --test src/__tests__/pilotMap.test.ts
 */
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      return nextResolve(`${specifier}.ts`, context);
    }
    throw error;
  }
}

// When loaded via --import, register this same file as a module-customisation hook.
register(import.meta.url, pathToFileURL('./'));

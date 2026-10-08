import { chmodSync, copyFileSync } from 'node:fs';

for (const name of ['LICENSE', 'NOTICE']) {
  copyFileSync(new URL(`../../../${name}`, import.meta.url), new URL(`../${name}`, import.meta.url));
}
chmodSync(new URL('../dist/cli.mjs', import.meta.url), 0o755);

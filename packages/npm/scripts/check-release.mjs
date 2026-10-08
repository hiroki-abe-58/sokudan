import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const { version } = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
assert.equal(process.env.RELEASE_TAG, `npm-v${version}`, 'tag must match the npm version');
const response = await fetch(`https://pypi.org/pypi/sokudan/${version}/json`, {
  signal: AbortSignal.timeout(30_000),
});
assert.equal(response.status, 200, 'publish the matching Python package to PyPI first');
const release = await response.json();
assert.ok(release.urls.some((file) => !file.yanked), 'Python release must not be yanked');
console.log(`npm ${version}: matching PyPI release exists`);

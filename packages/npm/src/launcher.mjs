import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { constants } from 'node:os';

export const { version } = /** @type {{ version: string }} */ (
  JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))
);

/** @param {string[]} args @param {NodeJS.ProcessEnv} env */
export function launchCommand(args, env) {
  // Never invoke the `sokudan` on PATH: it might be this npm launcher again.
  if (env.SOKUDAN_PYTHON) {
    return { command: env.SOKUDAN_PYTHON, args: ['-m', 'sokudan.cli', ...args] };
  }
  // Research probes need the Python training/baseline extras too.
  const extras = args[0] === 'probe-position' ? 'serve,train,bench,torch' : 'serve';
  return {
    command: 'uv',
    args: ['tool', 'run', '--python', '3.11', '--from', `sokudan[${extras}]==${version}`,
      'sokudan', ...args],
  };
}

/** @param {string[]} args @returns {Promise<number>} */
export async function main(args) {
  if (args.length === 0 || (args.length === 1 && ['--help', '-h'].includes(args[0] ?? ''))) {
    console.log(`sokudan ${version} — Python CLI launcher\n
Usage: sokudan <command> [options]
  serve             Start the local /v1/systemone HTTP server
  probe-position    Run the Python position-bias probe (downloads research extras)
  --version         Print the npm launcher version

Requires uv on PATH; first use installs sokudan in uv's isolated cache.
Or set SOKUDAN_PYTHON to a Python executable with sokudan and its extras installed.
Use 'sokudan serve --help' for server options. Node.js SDK: import { Sokudan } from 'sokudan'.`);
    return 0;
  }
  if (args.length === 1 && args[0] === '--version') {
    console.log(version);
    return 0;
  }
  const launch = launchCommand(args, process.env);
  return new Promise((resolve) => {
    const child = spawn(launch.command, launch.args, { stdio: 'inherit', shell: false });
    const onInterrupt = () => { child.kill('SIGINT'); };
    const onTerminate = () => { child.kill('SIGTERM'); };
    process.on('SIGINT', onInterrupt);
    process.on('SIGTERM', onTerminate);
    const cleanup = () => {
      process.off('SIGINT', onInterrupt);
      process.off('SIGTERM', onTerminate);
    };
    child.once('error', (error) => {
      cleanup();
      console.error(`sokudan: could not start ${launch.command}: ${error.message}`);
      if (!process.env.SOKUDAN_PYTHON) {
        console.error('Install uv: https://docs.astral.sh/uv/getting-started/installation/');
        console.error('Or set SOKUDAN_PYTHON to your Python executable with sokudan[serve] installed.');
      }
      resolve(1);
    });
    child.once('close', (code, signal) => {
      cleanup();
      resolve(code ?? (signal ? 128 + constants.signals[signal] : 1));
    });
  });
}

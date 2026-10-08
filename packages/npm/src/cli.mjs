#!/usr/bin/env node
import { main } from './launcher.mjs';

process.exitCode = await main(process.argv.slice(2));

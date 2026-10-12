// Label maps of the Gothenburg viewport from TEE's own worker code (public/js/segmentation.js),
// for the map figure of the report: the methods of the "method" selector for two seeds.
//   node port_maps.mjs [seedA=42] [seedB=7] [k=5]
// Writes results/port_maps_gbg.i32 (Int32, one map per run) and results/port_maps_gbg.json.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [seedA, seedB, k] = [+(process.argv[2] || 42), +(process.argv[3] || 7), +(process.argv[4] || 5)];
const H = 385, W = 720, DIM = 128;
const emb = path.join(os.homedir(), 'tee_data/harness-cache/gbg.f32');
const methods = ['kmeans', 'stable', 'smooth', 'gauss'];
const runs = [], parts = [];
for (const method of methods) {
    const tmp = path.join(os.tmpdir(), `port_maps_${method}.i32`);
    execFileSync(process.execPath, [path.join(HERE, 'tests/port_blocks.mjs'), 'run', emb, W, H, DIM, k, method, `${seedA},${seedB}`, tmp].map(String));
    parts.push(fs.readFileSync(tmp));
    fs.unlinkSync(tmp);
    runs.push({ method, seed: seedA }, { method, seed: seedB });
}
fs.writeFileSync(path.join(HERE, 'results/port_maps_gbg.i32'), Buffer.concat(parts));
fs.writeFileSync(path.join(HERE, 'results/port_maps_gbg.json'), JSON.stringify({ H, W, k, seeds: [seedA, seedB], runs }, null, 1));
console.log('wrote', runs.length, 'maps');

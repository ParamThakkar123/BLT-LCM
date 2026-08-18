# auto_setup state -- a49e866a-3f90-4544-870f-958236fe6820-tlf56

Published 2026-08-18T03:58:51Z from `/marimo/BLT-LCM/runs/auto_setup` by `lcm_scripts/publish_state.py`.

This is the BLT-LCM experiment driver's own state directory: one log and one completion marker per pipeline step, the artifacts each step was expected to produce, and the manifest of everything it ran. It is a record of what this machine did -- the results themselves are published separately, under `results/runs/`.

## Steps

| step | status | elapsed | batch size | finished |
| --- | --- | --- | --- | --- |
| `decoder` | ok | 936s | 256 | 2026-08-17T18:52:58+0000 |
| `encode:0.25` | ok | 145s | 48 | 2026-08-17T18:55:24+0000 |
| `encode:0.50` | ok | 25s | 48 | 2026-08-17T18:55:49+0000 |
| `encode:0.80` | ok | 27s | 48 | 2026-08-17T18:56:16+0000 |
| `mt:f0.25_s42` | already_complete | 0s | 32 | 2026-08-17T18:56:16+0000 |
| `mt:f0.50_s42` | ok | 12116s | 32 | 2026-08-17T22:18:12+0000 |
| `mt:f0.80_s42` | interrupted | 2799s | 32 | 2026-08-18T03:58:50+0000 |

## Restored from published results

These steps are finished, but not necessarily here: their results were read back out of the results refs and are included above as that step's artifacts.

| step | published as | from | files |
| --- | --- | --- | --- |
| `decoder` | `blt_decoder` | `origin/main` | 5 |
| `encode_0.25` | `lcm_blt_base_fraction0.25` | `origin/main` | 3 |
| `encode_0.50` | `lcm_blt_base_fraction0.5` | `origin/main` | 3 |
| `encode_0.80` | `lcm_blt_base_fraction0.8` | `origin/main` | 3 |
| `mt_f0.25_s42` | `lcm_blt_mt_fraction0.25_s42` | `origin/main` | 8 |

## Truncated

Too large to commit whole; the tail is published instead.

- `logs/mt_f0.80_s42.log` (6318173 bytes on disk)

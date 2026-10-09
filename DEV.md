# DEV.md — MAGER direction

## Goal

MAGER exists to generate maps for **lanly-dev/rts** (RE-MECH): a browser-based 3D RTS
(Three.js, Metal Fatigue-inspired, 600x600 three-tier battlefield, procedural fBm
heightfield + canvas textures + instanced scenery, resource deposits: metal/energy/oil).

Final goal: MAGER outputs a **textured map playable in rts** — height, biomes, resources,
base slots, scenery — delivered through a shared map-spec contract.

## Feature pillars

1. **Reference extraction → similar-map generation.** Read another game's map, extract its
   structural info (region layout, resource clusters, base positions), generate a new map
   with similar character. Honest limit: height can't be recovered from a flat screenshot —
   synthesize it from biomes (mountains high, water low) or author it. Stronger long-term:
   parse open map formats (StarCraft `.scm`, OpenRA, AoE `.rms`) for ground truth instead
   of screenshots.
2. **Reskin on the same structure.** Keep heightfield + layout fixed; replace
   texture/elements/3D dressing (biome palettes, scenery prop sets). MAGER stays a 2D
   authoring tool — it outputs data, the game renders 3D. Needs a shaded-relief / 3D
   heightfield preview so maps can be judged without exporting blind.
3. **Agent-readable output.** The map-spec JSON *is* the agent interface: versioned JSON
   Schema + examples + validator. An LLM (or another tool) writes spec JSON from a prompt
   ("4-player bowl map, rich metal, scarce oil"); MAGER/rts renders it.

## Step 1 — the contract (do this first)

Everything above waits on one thing: **`map-spec.json` v1 + the rts `MapLoader`**.

- **Spec** (canonical: this repo): `meta` (name/version/seed), `heightfield`
  (`parameterized` fbm params OR `explicit` 16-bit heightmap PNG), `materials` grid,
  `resources` (`{kind, x, z, reserve}`), `bases` (`{x, z, radius}`), `scenery`
  (`{density, seed, propSet, clearRadius}`). Surface layer only in v1.
- **rts `src/world/MapLoader.ts`**: parse + validate, feed height sampler / deposits /
  bases / scenery into `WorldLayers`, `TerrainTexture`, `TerrainScenery`. `?map=` URL
  param in dev, falls back to current procgen when absent — never break the existing game.
- **MAGER Feature 4**: "Export RE-MECH spec" — writes spec JSON + referenced PNGs,
  schema-validated on write.

This is the forcing function: the moment a MAGER export loads in-game, every later
feature becomes testable in minutes instead of being a demo.

## Roadmap (after step 1)

1. RE-MECH-shaped generator: crater-bowl / lane-map archetypes (not generic Perlin blobs).
2. RTS map quality: symmetric resource fairness checks, flat base slots with min spacing,
   readable combat lanes. Balance > prettiness.
3. Reference extraction → spec (screenshot path first, open-format parsers later).
4. Reskin feature: same structure, new palettes/prop sets.

## Principles

- **File-based pipeline**: Python desktop tool ↔ browser game. The spec is the contract;
  the stacks never need to unify.
- **Deterministic seeds everywhere** (the game already is — keep it that way).
- **Additive, never breaking**: spec loading must not disturb the built-in procgen path.
- **The spec is the hinge**: extraction outputs it, the generator consumes it, agents
  write it, the game renders it. Every feature touches the spec, not the other features.

# DEV.md — MAGER direction

## Goal

MAGER is a **general-purpose map authoring tool for games**. Long-term, any game that
needs maps — RTS, RPG, turn-based, and beyond — should be able to use it.

The first consumer is **lanly-dev/rts**: a browser-based 3D RTS (Three.js,
Metal Fatigue-inspired, 600x600 three-tier battlefield, procedural fBm heightfield +
canvas textures + instanced scenery, resource deposits: metal/energy/oil). Building
against a real game first keeps the tool honest; generalizing comes after.

Near-term goal: MAGER outputs a **textured map playable in rts** — height, biomes,
resources, base slots, scenery — delivered through a shared map-spec contract.

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
   ("4-player bowl map, rich metal, scarce oil"); MAGER/the game renders it.

## Step 1 — the contract (do this first)

Everything above waits on one thing: **`map-spec.json` v1 + the rts `MapLoader`**.

- **Spec** (canonical: this repo): a generic core plus game profiles. Core: `meta`
  (name/version/seed), `heightfield` (`parameterized` fbm params OR `explicit` 16-bit
  heightmap PNG), `materials` grid, `resources` (`{kind, x, z, reserve}`), `bases`
  (`{x, z, radius}`), `scenery` (`{density, seed, propSet, clearRadius}`). Anything
  game-specific lives under `profiles: { <game>: {...} }` — v1 ships the `rts` profile
  (surface-layer-only fields, deposit kinds metal/energy/oil).
- **rts `src/world/MapLoader.ts`**: parse + validate, feed height sampler / deposits /
  bases / scenery into `WorldLayers`, `TerrainTexture`, `TerrainScenery`. `?map=` URL
  param in dev, falls back to current procgen when absent — never break the existing game.
- **MAGER Feature 4**: "Export map spec" — writes spec JSON + referenced PNGs,
  schema-validated on write.

This is the forcing function: the moment a MAGER export loads in-game, every later
feature becomes testable in minutes instead of being a demo.

## Roadmap (after step 1)

1. Spec v1 + rts `MapLoader` + MAGER export (first playable map in rts).
2. rts-shaped generator: crater-bowl / lane-map archetypes (not generic Perlin blobs).
3. RTS map quality: symmetric resource fairness checks, flat base slots with min spacing,
   readable combat lanes. Balance > prettiness.
4. Reference extraction → spec (screenshot path first, open-format parsers later).
5. Reskin feature: same structure, new palettes/prop sets.
6. **Generalize**: onboard a second consumer (e.g. turn-based grid maps or an RPG) —
   promote shared fields into the core, keep the rest in profiles. The spec grows by
   proven need, not speculation.

## Principles

- **File-based pipeline**: Python desktop tool ↔ games. The spec is the contract; the
  stacks never need to unify.
- **Generic core, game profiles**: the spec must not become "the rts format with a
  different name." Game-specific needs go in profiles.
- **Deterministic seeds everywhere** (rts already is — keep it that way).
- **Additive, never breaking**: spec loading must not disturb a game's built-in procgen.
- **The spec is the hinge**: extraction outputs it, the generator consumes it, agents
  write it, games render it. Every feature touches the spec, not the other features.
- **Prove, then generalize**: rts first, second consumer later. No speculative
  abstraction.

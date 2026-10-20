"""Our own poker world: engine, hand logger, bots, the SimRoom and (P8) game sessions.

Everything here produces hands in the competition schema (docs/data_semantics.md), so the same feature code and
models score simulated hands exactly as they score competition hands. All IDs use the `SIM-` format and can never
look like competition IDs.
"""

"""Auto-Forge: find the fastest llama-server setup this machine can run, and prove it.

Design: docs/superpowers/specs/2026-10-25-auto-forge-design.md (local only).
Pure stdlib. Every module takes its I/O as injected functions so the logic is
testable without a GPU.
"""

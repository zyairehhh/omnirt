"""Shared dispatch policy helpers."""

from __future__ import annotations


TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled"})

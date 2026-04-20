"""Lazy server exports."""

from __future__ import annotations


def create_app(**kwargs):
    from omnirt.server.app import create_app as _create_app

    return _create_app(**kwargs)

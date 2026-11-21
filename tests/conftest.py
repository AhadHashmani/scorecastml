"""Test guards.

The suite must not call the NBA stats API. It also pins season hashing so
the synthetic fixtures do not depend on CPython's per-process hash salt.
"""

import pytest

import data.loader as _loader

# generate_synthetic_season mixes hash(season) into its RNG seed. CPython
# randomizes str hashes per process, so the same fixture can miss the
# accuracy and home-win checks on one run and pass on the next. This pins
# the hash for the test process only; the library is unchanged.
_loader.hash = lambda _season: 0


@pytest.fixture(autouse=True)
def block_nba_api(monkeypatch):
    """Fail the test if a loader path tries to reach nba_api."""
    import data.loader as loader

    def _blocked(season: str):
        raise RuntimeError(
            f"nba_api must not be called during tests (season {season})"
        )

    monkeypatch.setattr(loader, "_load_from_nba_api", _blocked)

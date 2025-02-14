from __future__ import annotations

from qubecalib.instrument.quel.quel1.tool.skew import SkewDataBase


def test_create_database() -> None:
    """create_database should create a database."""
    db = SkewDataBase()
    assert isinstance(db, SkewDataBase)

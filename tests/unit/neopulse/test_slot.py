from qubecalib.neopulse import Item, Slot


def test_inheritance() -> None:
    """Slot should inherit from Item."""
    assert issubclass(Slot, Item)


def test_empty_init() -> None:
    """Slot should initialize with no arguments."""
    slot = Slot()
    assert slot.duration is None


def test_init() -> None:
    """Slot should initialize with arguments."""
    slot = Slot(duration=10.0)
    assert slot.duration == 10.0


def test_target() -> None:
    """target should set the target(s) of the slot."""
    slot1 = Slot()
    slot1.target("Q00")
    assert slot1.targets == ("Q00",)
    slot2 = Slot()
    slot2.target("Q00", "Q01")
    assert slot2.targets == ("Q00", "Q01")

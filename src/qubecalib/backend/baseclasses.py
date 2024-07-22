from .. import neopulse as pls


class Backend:
    def append_sequence(
        self,
        sequence: pls.Sequence,
        time_offset: dict[str, int],
        time_to_start: dict[str, int],
    ) -> None:
        pass


class Result:
    pass

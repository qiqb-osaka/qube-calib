from __future__ import annotations

from dataclasses import dataclass

from quel_ic_config import Quel1PortType


@dataclass(frozen=True)
class BandRef:
    box_key: str
    port: Quel1PortType
    band_key: int

    def __str__(self) -> str:
        p = (
            f"({self.port[0]},{self.port[1]})"
            if isinstance(self.port, tuple)
            else f"({self.port})"
        )
        return f"{self.box_key}:port{p}:band{self.band_key}"


# @dataclass
# class BandAssignment:
#     band_to_channels: dict[BandRef, list[str]] = field(default_factory=dict)

#     def add(self, band: BandRef, channel_key: str) -> None:
#         self.band_to_channels.setdefault(band, []).append(channel_key)

#     def channels_of(self, band: BandRef) -> list[str]:
#         return self.band_to_channels.get(band, [])

#     def bands(self) -> list[BandRef]:
#         return list(self.band_to_channels.keys())


# @dataclass
# class ChannelConfig:
#     mapping: dict[str, BandRef] = field(default_factory=dict)

#     def register(self, channel_key: str, band: BandRef) -> None:
#         self.mapping[channel_key] = band

#     def resolve(self, channel_key: str) -> BandRef:
#         return self.mapping[channel_key]

#     def descrive(self, chhannel_key: str) -> dict:
#         b = self.mapping[chhannel_key]
#         return {
#             "channel_key": chhannel_key,
#             "box": b.box_key,
#             "port": b.port,
#             "band_key": b.band_key,
#         }

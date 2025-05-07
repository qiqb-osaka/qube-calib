from __future__ import annotations

from typing import Callable

from .....sysconfdb import SystemConfigDatabase


def define_port_and_channel(
    sysdb: SystemConfigDatabase,
    box_name: str,
    port_number: int | tuple[int, int],
    total_channels: int,
    port_id: Callable,
    channel_id: Callable | None = None,
) -> str:
    port_name = port_id(box_name)
    sysdb.define_port(
        port_name=port_name,
        box_name=box_name,
        port_number=port_number,
    )
    for i in range(total_channels):
        if channel_id is not None:
            channel_name = channel_id(box_name, i)
        else:
            channel_name = f"{port_name}{i}"
        sysdb.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=i,
        )
    return port_name


def create_sysdb_items_quel1_riken8(
    sysdb: SystemConfigDatabase,
    *,
    box_name: str,
    ipaddr_wss: str,
    default_ndelay: int = 7,
) -> None:
    sysdb.define_box(
        box_name=box_name,
        ipaddr_wss=ipaddr_wss,
        boxtype="quel1se-riken8",
    )

    # define read channels
    port_name = define_port_and_channel(
        sysdb,
        box_name=box_name,
        port_number=0,
        total_channels=4,
        port_id=lambda box_name: f"{box_name}.READ.IN",
    )
    sysdb._port_settings[port_name].ndelay_or_nwait = tuple(4 * [default_ndelay])

    define_port_and_channel(
        sysdb,
        box_name=box_name,
        port_number=1,
        total_channels=1,
        port_id=lambda box_name: f"{box_name}.READ.OUT",
    )

    define_port_and_channel(
        sysdb,
        box_name=box_name,
        port_number=(1, 1),
        total_channels=1,
        port_id=lambda box_name: f"{box_name}.READ.FOGI.OUT",
    )

    i: int | str | object
    for p, i in [(4, 0), (10, 1)]:
        port_name = define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=1,
            port_id=lambda box_name: f"{box_name}.MNTR{i}.IN",
        )
        sysdb._port_settings[port_name].ndelay_or_nwait = (default_ndelay,)

    for p, i, c in [(3, "X", 3), (6, 0, 3), (7, 1, 3), (8, 2, 1), (9, 3, 1)]:
        define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=c,
            port_id=lambda box_name: f"{box_name}.CTRL{i}",
            channel_id=lambda box_name, j: f"{box_name}.CTRL{i}.CH{j}",
        )

    define_port_and_channel(
        sysdb,
        box_name="Q132SE8",
        port_number=2,
        total_channels=3,
        port_id=lambda box_name: f"{box_name}.PUMP",
        channel_id=lambda box_name, j: f"{box_name}.PUMP.CH{j}",
    )


def create_sysdb_items_qube_riken_a(
    sysdb: SystemConfigDatabase,
    *,
    box_name: str,
    ipaddr_wss: str,
    default_ndelay: int = 7,
) -> None:
    sysdb.define_box(
        box_name=box_name,
        ipaddr_wss=ipaddr_wss,
        boxtype="qube-riken-a",
    )

    # define read channels
    i: int | str | object
    for p, i in [(0, 0), (13, 1)]:
        define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=1,
            port_id=lambda box_name: f"{box_name}.READ{i}.OUT",
        )

    for p, i in [(1, 0), (12, 1)]:
        port_name = define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=4,
            port_id=lambda box_name: f"{box_name}.READ{i}.IN",
        )
        sysdb._port_settings[port_name].ndelay_or_nwait = tuple(4 * [default_ndelay])

    for p, i in [(2, 0), (11, 1)]:
        define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=1,
            port_id=lambda box_name: f"{box_name}.PUMP{i}.OUT",
        )

    for p, i in [(4, 0), (9, 1)]:
        port_name = define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=1,
            port_id=lambda box_name: f"{box_name}.MNTR{i}.IN",
        )
        sysdb._port_settings[port_name].ndelay_or_nwait = (default_ndelay,)

    for p, i in [(5, 0), (6, 1), (7, 2), (8, 3)]:
        define_port_and_channel(
            sysdb,
            box_name=box_name,
            port_number=p,
            total_channels=3,
            port_id=lambda box_name: f"{box_name}.CTRL{i}",
            channel_id=lambda box_name, j: f"{box_name}.CTRL{i}.CH{j}",
        )

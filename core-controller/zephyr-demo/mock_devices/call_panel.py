#!/usr/bin/env python3
"""
Emit CiA-417 hall call PDOs on a virtual CAN interface for controller testing.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Iterable

import can
from rich.console import Console

console = Console()


def build_payload(floor: int, direction: int) -> bytes:
    """
    CiA-417 0x181 PDO layout (simplified):
    byte0: floor number
    byte1: direction (0=none,1=up,2=down)
    byte2: priority
    rest: reserved
    """
    data = bytearray(8)
    data[0] = floor & 0xFF
    data[1] = direction & 0x03
    data[2] = 1  # priority
    return bytes(data)


def iter_floors(total: int, pattern: str) -> Iterable[int]:
    seq = list(range(1, total + 1))
    idx = 0
    while True:
        if pattern == "random":
            yield random.choice(seq)
        else:
            yield seq[idx % len(seq)]
            idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="CiA-417 hall call generator")
    parser.add_argument("--channel", default="vcan0", help="python-can channel")
    parser.add_argument("--interface", default="socketcan", help="python-can interface")
    parser.add_argument("--floors", type=int, default=10, help="Number of floors to cycle")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between calls")
    parser.add_argument("--pattern", choices=("roundrobin", "random"), default="roundrobin")
    parser.add_argument("--cob-id", type=lambda x: int(x, 0), default=0x181, help="PDO COB-ID")
    args = parser.parse_args()

    bus = can.interface.Bus(channel=args.channel, bustype=args.interface)
    console.print(f"[cyan]Call panel[/cyan] publishing on {args.interface}:{args.channel} (COB-ID 0x{args.cob_id:X})")
    for floor in iter_floors(args.floors, args.pattern):
        direction = random.choice((0, 1, 2))
        payload = build_payload(floor, direction)
        msg = can.Message(arbitration_id=args.cob_id, data=payload, is_extended_id=False)
        bus.send(msg)
        console.print(f"[green]→ Hall call[/green] floor={floor} dir={direction}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()


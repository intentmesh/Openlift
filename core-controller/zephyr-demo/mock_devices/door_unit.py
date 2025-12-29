#!/usr/bin/env python3
"""
Simulated CiA-417 door unit that reacts to command PDOs and publishes status frames.
"""

from __future__ import annotations

import argparse
import time

import can
from rich.console import Console

console = Console()


def describe_command(cmd: int) -> str:
    parts = []
    if cmd & 0x01:
        parts.append("OPEN")
    if cmd & 0x02:
        parts.append("CLOSE")
    if cmd & 0x04:
        parts.append("NUDGE")
    return "|".join(parts) if parts else "IDLE"


def build_status(cmd: int) -> bytes:
    """
    byte0: door_closed sensor (1/0)
    byte1: door_locked sensor
    byte2: fault byte
    """
    door_closed = 0 if cmd & 0x01 else 1
    door_locked = door_closed
    fault = 0
    return bytes([door_closed, door_locked, fault, 0, 0, 0, 0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Door operator mock for CiA-417")
    parser.add_argument("--channel", default="vcan0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--command-cobid", type=lambda x: int(x, 0), default=0x221)
    parser.add_argument("--status-cobid", type=lambda x: int(x, 0), default=0x1A1)
    args = parser.parse_args()

    bus = can.interface.Bus(channel=args.channel, bustype=args.interface)
    console.print(f"[cyan]Door unit[/cyan] listening on COB-ID 0x{args.command_cobid:X}")

    while True:
        msg = bus.recv()
        if msg is None:
            continue
        if msg.arbitration_id != args.command_cobid or len(msg.data) == 0:
            continue
        command = msg.data[0]
        console.print(f"[magenta]← Door command[/magenta] {describe_command(command)}")
        status = build_status(command)
        status_msg = can.Message(arbitration_id=args.status_cobid, data=status, is_extended_id=False)
        bus.send(status_msg)
        time.sleep(0.2)


if __name__ == "__main__":
    main()


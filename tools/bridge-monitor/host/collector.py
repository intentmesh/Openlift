#!/usr/bin/env python3
"""
Read JSON frames from the Bridge Monitor serial port and forward to MQTT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge Monitor host collector")
    parser.add_argument("--serial", required=True, help="Serial device (e.g., /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--mqtt", required=True, help="MQTT broker URL (mqtt://host[:port])")
    parser.add_argument("--car-id", required=True, help="Elevator identifier")
    parser.add_argument("--log", type=Path, default=Path("telemetry.log"))
    return parser.parse_args()


def parse_broker(url: str) -> tuple[str, int]:
    if not url.startswith("mqtt://"):
        raise ValueError("Broker must use mqtt:// scheme")
    host_port = url[len("mqtt://") :]
    if ":" in host_port:
        host, port = host_port.split(":", 1)
        return host, int(port)
    return host_port, 1883


def main() -> None:
    args = parse_args()
    host, port = parse_broker(args.mqtt)
    client = mqtt.Client()
    client.connect(host, port)
    client.loop_start()

    topic = f"openlift/bridge/{args.car_id}/telemetry"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(args.serial, args.baud, timeout=1) as ser, args.log.open("a", encoding="utf-8") as log_file:
        while True:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON frame: {line}", file=sys.stderr)
                continue
            payload["bridge_received_ts"] = time.time()
            body = json.dumps(payload)
            log_file.write(body + "\n")
            log_file.flush()
            client.publish(topic, body, qos=0, retain=False)


if __name__ == "__main__":
    main()


# Bridge Monitor – ESP32-S3 CAN/IO Telemetry

This module delivers a reference firmware + host collector that taps CANopen Lift traffic (or discrete door relays) and streams telemetry to MQTT/JSON dashboards.

## Architecture

```
[CANopen nodes] --(isolated CAN transceiver)--> [ESP32-S3 Bridge]
     ^                                                |
     |                                   UART/WebUSB JSON frames
     v                                                |
[Door interlocks, relays] --(GPIO opto board)--> [ESP32-S3 Bridge]
```

The bridge firmware:

* Initializes CAN controller (speed selectable 125/250/500 kbit/s)
* Samples discrete inputs (door closed, car top stop, limits)
* Packages telemetry into JSON frames:
  ```json
  {
    "ts": 1700000123.12,
    "can": {"id": 0x181, "dlc": 8, "data": "021F001122334455"},
    "door_closed": true,
    "faults": []
  }
  ```
* Sends frames over UART (default 921600) and publishes via MQTT (Zephyr native network stack if Wi-Fi credentials provided).

The host collector (`host/collector.py`) can run on an SBC / service laptop, reading the serial stream and relaying frames to:

* Local MQTT broker (topic `openlift/bridge/<car_id>/telemetry`)
* WebSocket dashboard (planned)

## Firmware Quickstart

Requirements:

* Zephyr SDK 0.16+ and `west` set up (`export ZEPHYR_BASE=...`)
* ESP32-S3 DevKitC with isolated CAN transceiver (e.g., TI SN65HVD230 + ISO1042)

```bash
cd tools/bridge-monitor/firmware
west init -l .
west update
west build -b esp32s3_devkitm -- -DCONF_FILE=prj.conf
west flash
```

Update `firmware/prj.conf` for Wi-Fi SSID/PSK (optional) and CAN bitrate.

## Host Collector

```bash
cd tools/bridge-monitor/host
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python collector.py --serial /dev/ttyACM0 --mqtt mqtt://localhost --car-id CAR_12
```

Collector responsibilities:

1. Parse line-delimited JSON frames
2. Enrich with host timestamp + location metadata
3. Publish to MQTT + append to local log file (`telemetry.log`)

## Dashboard Integration (planned)

* Create a lightweight React dashboard (reuse `tools/openvibe` components) that subscribes to MQTT and displays door status, CAN faults, and overlayed OpenVibe ride-quality reports.
* Add anomaly detection hooks (RMS of door torque, stop counts, brake temps).

## Next Steps

* Support OTA updates from Firebase or AWS IoT
* Add TLS for MQTT
* Provide CNC-ready PCB design for the isolated CAN/GPIO board


# Zephyr Split-Kernel Demo (Controller Workbench)

This demo shows how the OpenLift controller can split **safety-critical logic** from **application/dispatch logic** while speaking the CiA‑417 object dictionary.

## Layout

```
core-controller/
 └─ zephyr-demo/
    ├─ app/            # Zephyr firmware (safety + motion threads)
    │  ├─ prj.conf
    │  ├─ CMakeLists.txt
    │  └─ src/
    │     ├─ main.c
    │     └─ cia417_mock.[ch]
    └─ mock_devices/
       ├─ call_panel.py
       └─ door_unit.py
```

### Threads

| Thread          | Priority | Responsibility |
|-----------------|----------|----------------|
| `safety_core`   | 3        | Validates door locks, estop loop, grants drive-ready |
| `motion_core`   | 4        | Implements dispatcher logic over the CiA‑417 OD |

The two cores communicate over `k_mutex`/shared structs only through the strictly typed API defined in `cia417_mock.h`. The safety thread alone can transition into `DRIVE_READY`. If inputs go invalid, it drops back to `IDLE/FAULT`, forcing the dispatcher to re-request motion.

## Building the Zephyr Firmware

Prereqs:

* Zephyr SDK 0.16+ (or West + toolchain) with `ZEPHYR_BASE` exported
* Board support for `esp32s3_devkitm`, `nrf52840dk_nrf52840`, or `native_sim`

```bash
cd core-controller/zephyr-demo/app
west build -b native_sim -- -DCONF_FILE=prj.conf
west build -t run    # launches QEMU and exposes the Zephyr shell
```

Useful shell commands once the demo is running:

```
openlift> status
openlift> sensor door closed
openlift> call 8
openlift> sensor estop trip
```

Each `call` updates the CiA‑417 object dictionary and the motion thread will move the car toward the nearest pending floor while logging the PDO-equivalent state transitions.

## Mock CiA‑417 Devices

The `mock_devices/` scripts mimic CANopen Lift nodes via `python-can`. They are optional utilities for feeding traffic from a laptop while the Zephyr app runs on actual hardware.

```bash
cd core-controller/zephyr-demo/mock_devices
pip install -r requirements.txt
python call_panel.py --channel vcan0 --floors 12 --interval 5
python door_unit.py --channel vcan0
```

* `call_panel.py` publishes virtual hall/car call PDOs (`0x181` / `0x191`) with the same payload schema defined in `cia417_mock.h`.
* `door_unit.py` listens for door control commands (`0x221`) and prints interpreted actions while also echoing a status PDO (`0x1A1`).

These mocks allow the dispatcher logic to be tested against realistic CAN frames even before custom hardware exists.

## Extending the Demo

1. Replace `cia417_mock.c` with actual CANopen stack bindings.
2. Swap the host shell commands for a UART/WebUSB protocol so phone apps can manipulate the OD.
3. Co-locate the Rust `safety` crate via `rust_bindings.c` to run certified logic inside Zephyr’s `userspace`.


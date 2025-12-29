> **Disclaimer**: This repository is a field-driven reference implementation and discussion starter. It is not a complete or production-ready system. Development may be intermittent; please view this as an open blueprint for the community to learn from and iterate on rather than a supported product.
# 🚀 OpenLift: The Open Vertical Transportation Platform

> Vendor-neutral safety core + intelligent app layer = the open standard for the modernization era.

[Vision](#vision) · [Why Now](#why-now) · [Stack](#stack-overview) · [Hardware](#hardware-reference-blueprint) · [Intelligence](#intelligence-layer) · [Roadmap](#roadmap) · [Contribute](#how-to-get-involved)

---

## 📊 Status at a Glance
| Track | Maturity | Next Up |
| --- | --- | --- |
| Vision & Governance | Concept | Formalize OpenLift Forum charter |
| Tooling | Prototype | OpenVibe + SpecScanner PoC |
| Simulator | Prototype | Real-time CAN mock + HIL hooks |
| Core Controller | Research | Zephyr safety subset + CANopenNode port |
| Safety Path | Research | IEC 61508 SIL 3 plan + ISO 22201 checklist |

*(Open source release will be phased; see [Licensing Plan](#licensing-plan).)*

---

## 🌍 Vision
OpenLift breaks the OEM walled gardens by separating a safety-critical kernel from an extensible application/runtime layer. Independent service providers, component manufacturers, and building owners get the freedom to mix and match hardware, run any compliant software, and keep long-term maintenance in their control.

---

## ⚡ Why Now
- **Modernization demand**: $20B market by 2030; building owners want "Right to Repair" and retrofit-friendly controllers.
- **Standards convergence**: CANopen Lift (CiA 417) virtual devices + Zephyr's safety push remove the hardest interoperability barriers.
- **AI-assisted development**: Modern LLMs can parse specs, generate protocol code, run simulations, and produce verification artifacts—compressing timelines for lean teams.
- **Commodity hardware**: Dual-MCU boards (STM32H7 + ESP32-S3) with isolated CAN + FPGA co-processors offer a path to SIL readiness without OEM markup.

---

## 🏗️ Stack Overview

| Layer | Component | Stack | Responsibilities |
| --- | --- | --- | --- |
| L4 | Cloud & Analytics | Firebase · Python | Fleet management, predictive maintenance, dashboards |
| L3 | Application/UI | Flutter (Dart) · Linux/ESP32 | HMIs, technician tooling, theming, configuration, multimedia |
| L2 | Motion Control | Zephyr RTOS (C/C++) | Dispatch, door profiles, floor leveling, CANopen Master |
| L1 | Safety Core | Zephyr SIL subset · FPGA | Safety chain, overspeed, interlocks, watchdog arbitration |

### 🛡️ Safety/Application Split
- **Architecture Goal**: Isolate safety-critical logic in a sealed "Safety Kernel" designed to target IEC 61508 SIL 3 and ISO 22201 (PESSRAL).
- Open APIs (`request_floor(5)`, `door_profile.set(...)`) enforce guardrails; watchdogs + redundant sensors are designed to fail safe.
- Apps, drivers, and OTA updates live in the innovation layer; the architecture ensures faults trigger a controlled stop.

---

## 🛠️ Hardware Reference Blueprint
1. **Dual-Brain MCU**
   - **Safety Processor (STM32H7/F4)**: Runs Zephyr safety subset, CAN FD master, deterministic door/motion loops.
   - **Connectivity Processor (ESP32-S3 or Linux SBC)**: Hosts Flutter HMI, Wi-Fi/BLE/OTA, acts as air-gapped gateway to the cloud/Web Serial.
2. **Industrial CAN Backbone**
   - Galvanically isolated transceivers (ISO1050 family) + digital isolators to survive car/machine room ground differentials.
   - TVS + gas discharge protection absorb lightning/motor transients; 24 V tolerant and maintenance-friendly.
3. **CANopen Lift compliance**
   - Targets virtual devices (Call Controller, Door Unit, Drive Unit) so third-party fixtures "just work."
   - CANopenNode + generated bindings provide ANSI C core with idiomatic C++/Python wrappers.

---

## 🧠 Intelligence Layer
- **OR-Tools dispatch**: Model hall calls as PDPTW; minimize waiting + journey time, with tunable penalty weights for up-peak, lunch, down-peak profiles.
- **Reinforcement learning frontier**: SimPy gym + DQN agents learn adaptive allocation strategies and benchmark against OR-Tools baselines.
- **Edge diagnostics**:  
  - `OpenVibe` FFT pipeline flags 12 Hz guide-roller wear via phone accelerometers (ISO 18738).  
  - `SpecScanner` ingests legacy PDFs so mechanics can chat with manuals ("What's the Dover DMC-1 fault blink?").  
  - Autoencoders on CAN traces detect door/drive anomalies before they strand riders.

---

## 🛣️ Roadmap
1. **Phase 1 – Digital Utility Belt (0–6 mo)**  
   Ship AI-native mechanic tools (OpenVibe, SpecScanner, Pollsheet integrations) to win trust and gather pain points.
2. **Phase 2 – IoT Overlay (6–12 mo)**  
   Launch `OpenLift Bridge` (ESP32-S3) for read-only CAN taps, discrete I/O monitoring, and Firebase dashboards with anomaly detection.
3. **Phase 3 – Open Controller (12–24 mo)**  
   Deliver `OpenLift Core` kits: dual-MCU boards, Zephyr hybrid kernel, CANopenNode port, OR-Tools dispatch, compliance artifacts, plus governance via the Open Source Elevator Forum.

*(Timeline is directional; each phase graduates when verification artifacts meet IEC/ISO requirements.)*

---

## 📐 Development Playbook
- **Engineering rules**: MISRA-C adherence, no heap allocation in RT threads, determinism-first logging, hardware watchdogs everywhere.
- **Browser flashing**: esptool-js + Web Serial enables OTA/USB updates from Chrome (even phones with USB-OTG).
- **Testing**: Every feature gets a plan artifact, QEMU/HIL run, and dashboard visualization before hardware pilots.

---

## 📂 Repository Structure
- `core-controller/` – split-kernel controller logic (C++/Rust) plus the `zephyr-demo` reference safety/application split.
- `simulator/` – Python digital twin + traffic/kinematics models.
- `dashboard/` – React/TS cloud dashboards and technician tooling.
- `tools/openvibe/` – ride-quality analyzer that turns accelerometer CSV logs into diagnostics.
- `tools/bridge-monitor/` – ESP32-S3 CAN/IO telemetry bridge firmware + host collector.
- `tools/specscanner/` – document ingestion & Q/A service for PDF/TXT manuals.
- `docs/` – Architecture, compliance strategy, and detailed roadmap.

## 🧰 Featured Tool: OpenVibe
OpenVibe is the Phase-1 flagship utility. It ingests phone accelerometer logs, applies ISO 18738 metrics (RMS acceleration, jerk, comfort class), classifies dominant vibration peaks via configurable frequency bands, and emits Markdown/JSON reports. Run it locally via `make run`, spin up the REST API via `make serve`, or experiment with the Flutter scaffold in `tools/openvibe/mobile`. Extend the issue knowledge base by editing `tools/openvibe/issues.json`.

---

## 🤝 How to Get Involved
- **Independent Service Providers**: Share modernization blockers, test diagnostics, define "must have" workflows.
- **Hardware partners**: Offer CANopen Lift fixtures, door operators, or drive units for interoperability testing.
- **Developers**: Contribute Flutter HMIs, Python analytics, Zephyr board support, or OR-Tools/RL enhancements.
- **Compliance & safety experts**: Help draft certification plans, safety case documents, and audit-ready artifacts.

Ping the maintainers via GitHub Discussions (coming) or drop an issue describing your elevator pain point. Early collaborators shape the spec, tooling, and licensing priorities.

---

## 📜 Licensing Plan
- **Application layer & tooling**: Apache 2.0 (encourages commercial add-ons and themes).
- **Drivers & hardware adapters**: GPLv3 (keeps interoperability work open).
- **Safety kernel**: distributed as a signed commercial binary to fund certification while exposing stable APIs to the open community.

Source drops will occur progressively; nothing here obligates releasing sensitive or employer-owned material. This README is a public vision so the community can align before code lands.

---

OpenLift exists to give technicians, ISPs, and component vendors the same freedom Android gave handset makers—an open, safety-first platform where innovation is decoupled from proprietary hardware. 🚧 Let's build the elevator ecosystem we always wished existed.

---
**Disclaimer:** OpenLift is currently a research and reference architecture project. The code provided is for experimental and educational purposes only. It is not certified for use in safety-critical lift applications. Always use certified, redundant safety chains and follow local regulations (ASME A17.1, EN 81) when working with vertical transportation equipment.

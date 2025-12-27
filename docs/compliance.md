# Regulatory Strategy & Compliance

## Approach
OpenLift is designed to support compliance with major international safety codes, but the open-source code itself is **not certified**. Certification applies to the final assembled system.

## Traceability Targets

| Standard | Applicability | Strategy |
| --- | --- | --- |
| **ASME A17.1 / CSA B44** | North America | Compliance via certified safety components (SIL 3 kernel + hardware watchdogs). |
| **EN 81-20/50** | Europe | Meeting PESSRAL (Programmable Electronic Systems) requirements. |
| **IEC 61508** | Functional Safety | Targeting SIL 3 for the L1 Safety Kernel. |
| **ISO 22201** | Lifts (PESSRAL) | Architecture designed for diverse redundancy (Dual MCU). |

## Safety Case
We are building a "Safety Case" repository of artifacts:
- **FMEA:** Failure Mode and Effects Analysis (Planned).
- **HAZOP:** Hazard and Operability Study (Planned).
- **V-Model:** All requirements map to specific test vectors in the `simulator/`.


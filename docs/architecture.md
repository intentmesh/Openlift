# Architecture Overview

## The Hybrid Kernel Strategy
OpenLift uses a "Split Kernel" architecture to satisfy two conflicting requirements:
1. **Safety:** Hard real-time constraints, formal verification (IEC 61508), and zero-crash reliability.
2. **Flexibility:** Modern UI frameworks, cloud connectivity, and rapid iteration.

### L1: The Safety Core (Zephyr SIL Subset)
- **Role:** The "Lizard Brain." It ensures the elevator never moves with doors open and never exceeds speed limits.
- **Tech:** Rust / Embedded C (MISRA-C compliant).
- **Hard Constraints:** No dynamic memory allocation, fixed-cycle execution.

### L2: Motion Control & Dispatch (Zephyr / C++)
- **Role:** The "Driver." Calculates curves, manages floor stops, and speaks CANopen Lift.
- **Tech:** Zephyr RTOS, CANopenNode.

### L3: Application Layer (Flutter / Linux)
- **Role:** The "Experience." Draws the screens, talks to the cloud, plays music.
- **Isolation:** Can crash or restart without stopping the car (Safety Core takes over and holds at floor).

## Interfaces
- **Internal Bus:** SPI/UART link between Safety and App processors.
- **External Bus:** CANopen Lift (CiA 417) for all peripheral devices.


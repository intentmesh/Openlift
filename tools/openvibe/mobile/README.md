# OpenVibe Mobile (Flutter Skeleton)

This directory holds a Flutter scaffold that will evolve into the live ride-quality companion app. The app streams accelerometer data to the analyzer service (`service.py`) and previews ISO metrics in real time.

## Quick Start

```bash
cd tools/openvibe/mobile/openvibe_mobile
flutter pub get
flutter run
```

The current UI is a minimal placeholder that:

- Requests accelerometer permissions
- Streams sensor data and displays instantaneous RMS
- Sends data batches to the REST endpoint (`/analyze`) for full reports

## Next Steps

- Replace mock networking with a WebSocket/gRPC client
- Store ride segments locally for offline mode
- Integrate Firebase Auth + Cloud Firestore for fleet sync


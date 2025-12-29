import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:sensors_plus/sensors_plus.dart';

void main() {
  runApp(const OpenVibeApp());
}

class OpenVibeApp extends StatefulWidget {
  const OpenVibeApp({super.key});

  @override
  State<OpenVibeApp> createState() => _OpenVibeAppState();
}

class _OpenVibeAppState extends State<OpenVibeApp> {
  StreamSubscription<AccelerometerEvent>? _sub;
  final List<String> _log = [];
  double _rms = 0;

  @override
  void initState() {
    super.initState();
    _sub = accelerometerEvents.listen((event) {
      final rms = math.sqrt(event.x * event.x + event.y * event.y + event.z * event.z);
      setState(() {
        _rms = rms;
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  Future<void> _sendMockRecording() async {
    final uri = Uri.parse('http://localhost:8000/analyze');
    final csv = 'timestamp,ax,ay,az\n0,0.1,0.0,9.8\n0.02,0.2,0.1,9.8';
    final req = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromString('file', csv, filename: 'mobile.csv'));
    final resp = await req.send();
    final body = await resp.stream.bytesToString();
    setState(() {
      _log.add(body);
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('OpenVibe Mobile'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Instantaneous RMS: ${_rms.toStringAsFixed(3)} m/s²'),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _sendMockRecording,
                child: const Text('Send Sample Recording'),
              ),
              const SizedBox(height: 16),
              const Text('Analysis log:'),
              Expanded(
                child: ListView.builder(
                  itemCount: _log.length,
                  itemBuilder: (context, index) => Text(_log[index]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}


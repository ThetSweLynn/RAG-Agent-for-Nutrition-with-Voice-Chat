import 'dart:io';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

class VoiceChatHelper {
  final _recorder = FlutterSoundRecorder();
  bool _isRecorderInitialized = false;

  // Initialize the recorder and request microphone permission
  Future<void> initRecorder() async {
    final status = await Permission.microphone.request();
    if (status.isGranted) {
      await _recorder.openRecorder();
      _isRecorderInitialized = true;
    } else {
      throw 'Microphone permission not granted';
    }
  }

  // Start recording and save the audio to a file
  Future<void> startRecording() async {
    if (!_isRecorderInitialized) return;
    await _recorder.startRecorder(toFile: 'temp_audio.wav');
  }

  // Stop recording, send the audio to Flask, and get the AI response
  Future<String?> stopRecordingAndSend() async {
    if (!_isRecorderInitialized) return null;

    // Stop recording and get the file path
    final path = await _recorder.stopRecorder();
    if (path == null) return null;

    final file = File(path);

    // Replace with your Flask backend URL
    final uri = Uri.parse('http://<your-flask-server-ip>:5000/audio_query');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('audio', file.path));

    // Send the request to Flask and get the response
    final response = await request.send();
    if (response.statusCode == 200) {
      final responseData = await response.stream.bytesToString();
      return responseData; // AI's response
    } else {
      throw 'Failed to process audio';
    }
  }

  // Dispose the recorder when done
  Future<void> dispose() async {
    if (_isRecorderInitialized) {
      await _recorder.closeRecorder();
      _isRecorderInitialized = false;
    }
  }
}

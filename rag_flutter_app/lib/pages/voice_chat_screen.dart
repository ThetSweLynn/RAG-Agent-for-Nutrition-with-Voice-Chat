import 'package:flutter/material.dart';
import 'package:rag_flutter_app/services/voice_chat_helper.dart';

class VoiceChatScreen extends StatefulWidget {
  const VoiceChatScreen({super.key});

  @override
  _VoiceChatScreenState createState() => _VoiceChatScreenState();
}

class _VoiceChatScreenState extends State<VoiceChatScreen> {
  final VoiceChatHelper _voiceChatHelper = VoiceChatHelper();
  String? _response = "Press and hold the button to speak.";
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();
    _initializeRecorder();
  }

  // Initialize the recorder
  Future<void> _initializeRecorder() async {
    try {
      await _voiceChatHelper.initRecorder();
    } catch (e) {
      setState(() {
        _response = "Error initializing recorder: $e";
      });
    }
  }

  // Handle recording start
  void _startRecording() async {
    try {
      await _voiceChatHelper.startRecording();
      setState(() {
        _isRecording = true;
        _response = "Listening...";
      });
    } catch (e) {
      setState(() {
        _response = "Error starting recording: $e";
      });
    }
  }

  // Handle recording stop and send the audio
  void _stopRecording() async {
    try {
      final response = await _voiceChatHelper.stopRecordingAndSend();
      setState(() {
        _isRecording = false;
        _response = response ?? "No response received.";
      });
    } catch (e) {
      setState(() {
        _response = "Error stopping recording: $e";
      });
    }
  }

  @override
  void dispose() {
    _voiceChatHelper.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Voice Chat")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _response ?? "",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            GestureDetector(
              onLongPress: _startRecording,
              onLongPressUp: _stopRecording,
              child: CircleAvatar(
                radius: 40,
                backgroundColor: _isRecording ? Colors.red : Colors.blue,
                child: Icon(
                  _isRecording ? Icons.mic : Icons.mic_none,
                  color: Colors.white,
                  size: 40,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

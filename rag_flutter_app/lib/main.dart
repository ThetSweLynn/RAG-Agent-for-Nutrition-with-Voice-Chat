import 'package:flutter/material.dart';
import 'package:rag_flutter_app/pages/voice_chat_screen.dart';
import 'pages/introduction_page.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RAG Lifestyle Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.teal,
      ),
      home: IntroductionPage(),
    );
  }
}

import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;

  ApiService({required this.baseUrl});

  Future<String> getResponseFromApi(String query) async {
    final response = await http.post(
      Uri.parse('$baseUrl/query'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'query': query}),
    );

    if (response.statusCode == 200) {
      // Extract the response data
      final Map<String, dynamic> data = json.decode(response.body);
      return data['response'];  // Assuming 'result' is the field in the response
    } else {
      throw Exception('Failed to get data from API');
    }
  }
}

import 'dart:async';
import 'dart:typed_data';
import 'dart:io';

class SocketService {
  // Change this to your computer's IP address
  static const String serverHost = '127.0.0.1'; // Replace with YOUR computer's IP
  static const int serverPort = 5555;
  
  /// Send image bytes to Python backend and receive result
  static Future<String> sendImageToServer(Uint8List imageBytes) async {
    Socket? socket;
    
    try {
      print('Connecting to $serverHost:$serverPort...');
      
      // Connect to server with timeout
      socket = await Socket.connect(
        serverHost, 
        serverPort,
        timeout: Duration(seconds: 10),
      );
      
      print('Connected to server');
      
      // Send image bytes
      print('Sending image (${imageBytes.length} bytes)...');
      socket.add(imageBytes);
      await socket.flush();
      
      // Close the write side to signal we're done sending
      await socket.close();
      
      print('Waiting for response...');
      
      // Receive response with timeout
      final responseBytes = await socket.timeout(
        Duration(minutes: 5, seconds: 30),
        onTimeout: (sink) {
          sink.close();
          throw TimeoutException('Server took too long to respond (>5 minutes)');
        },
      ).fold<List<int>>(
        <int>[],
        (previous, element) => previous..addAll(element),
      );
      
      if (responseBytes.isEmpty) {
        throw Exception('No response received from server');
      }
      
      // Convert response bytes to string
      final result = String.fromCharCodes(responseBytes);
      
      print('Successfully received result (${result.length} chars)');
      
      // Check for error messages
      if (result.startsWith('ERROR:')) {
        throw Exception(result);
      }
      
      return result;
      
    } on SocketException catch (e) {
      print('Socket error: $e');
      throw Exception('Cannot connect to server at $serverHost:$serverPort. '
          'Make sure the Python server is running and the IP address is correct.');
    } on TimeoutException catch (e) {
      print('Timeout error: $e');
      throw Exception('Connection or response timeout: ${e.message}');
    } catch (e) {
      print('Error: $e');
      throw Exception('Error communicating with server: $e');
    } finally {
      // Cleanup
      try {
        socket?.destroy();
      } catch (e) {
        print('Error destroying socket: $e');
      }
    }
  }
  
  /// Test connection to server
  static Future<bool> testConnection() async {
    Socket? socket;
    try {
      socket = await Socket.connect(
        serverHost,
        serverPort,
        timeout: Duration(seconds: 5),
      );
      print('Connection test successful');
      return true;
    } catch (e) {
      print('Connection test failed: $e');
      return false;
    } finally {
      socket?.destroy();
    }
  }
  
  /// Check if server is reachable (ping test)
  static Future<bool> isServerReachable() async {
    try {
      final result = await InternetAddress.lookup(serverHost);
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } catch (e) {
      print('Server not reachable: $e');
      return false;
    }
  }
}
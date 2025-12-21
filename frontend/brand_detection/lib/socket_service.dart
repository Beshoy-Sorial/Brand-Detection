import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'dart:convert';

class SocketService {
  static const String serverHost = '127.0.0.1';
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
      
      // Create a completer to handle the async response
      final completer = Completer<String>();
      final responseBuffer = <int>[];
      
      // Set up listener BEFORE sending data
      socket.listen(
        (data) {
          responseBuffer.addAll(data);
          print('Received ${data.length} bytes from server (total: ${responseBuffer.length})');
        },
        onDone: () {
          print('Server closed connection');
          if (!completer.isCompleted) {
            if (responseBuffer.isEmpty) {
              completer.completeError(Exception('No response received from server'));
            } else {
              final result = String.fromCharCodes(responseBuffer);
              completer.complete(result);
            }
          }
        },
        onError: (error) {
          print('Socket error in listener: $error');
          if (!completer.isCompleted) {
            completer.completeError(Exception('Socket error: $error'));
          }
        },
        cancelOnError: true,
      );
      
      // Send length prefix (4 bytes, big endian)
      final lengthBytes = ByteData(4);
      lengthBytes.setUint32(0, imageBytes.length, Endian.big);
      socket.add(lengthBytes.buffer.asUint8List());
      
      // Send image bytes
      print('Sending image (${imageBytes.length} bytes)...');
      socket.add(imageBytes);
      await socket.flush();
      print('Data sent, waiting for response...');
      
      print('Waiting for response...');
      
      // Wait for response with longer timeout (5 minutes)
      final result = await completer.future.timeout(
        Duration(minutes: 5),
        onTimeout: () {
          throw TimeoutException('Server took too long to respond (>5 minutes)');
        },
      );
      
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
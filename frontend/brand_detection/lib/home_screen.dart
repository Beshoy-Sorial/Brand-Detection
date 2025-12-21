import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'socket_service.dart';

class HomeScreen extends StatefulWidget {
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isProcessing = false;
  String? _result;
  String? _imagePath;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            // App Bar
            SliverAppBar(
              expandedHeight: 120,
              floating: false,
              pinned: true,
              backgroundColor: Colors.white,
              elevation: 0,
              flexibleSpace: FlexibleSpaceBar(
                title: Text(
                  'Barnoda',
                  style: TextStyle(
                    color: Colors.black87,
                    fontWeight: FontWeight.bold,
                    fontSize: 24,
                  ),
                ),
                centerTitle: true,
              ),
            ),

            // Content
            SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Header Text
                    Text(
                      'Brand Logo Verification',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Upload or capture an image to verify brand authenticity',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 32),

                    // Image Preview
                    if (_imagePath != null)
                      Container(
                        height: 250,
                        margin: EdgeInsets.only(bottom: 24),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.1),
                              blurRadius: 10,
                              offset: Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.file(
                            File(_imagePath!),
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),

                    // Processing Indicator
                    if (_isProcessing)
                      Container(
                        padding: EdgeInsets.all(32),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.05),
                              blurRadius: 10,
                              offset: Offset(0, 2),
                            ),
                          ],
                        ),
                        child: Column(
                          children: [
                            SizedBox(
                              width: 60,
                              height: 60,
                              child: CircularProgressIndicator(
                                strokeWidth: 5,
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  Colors.blue[600]!,
                                ),
                              ),
                            ),
                            SizedBox(height: 24),
                            Text(
                              "Analyzing Image...",
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                                color: Colors.black87,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              "This may take a few moments",
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ),

                    // Result Display
                    if (_result != null && !_isProcessing)
                      Container(
                        padding: EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: _result!.contains('ERROR')
                                ? [Colors.red[50]!, Colors.red[100]!]
                                : [Colors.blue[50]!, Colors.blue[100]!],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.05),
                              blurRadius: 10,
                              offset: Offset(0, 2),
                            ),
                          ],
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Icon(
                                  _result!.contains('ERROR')
                                      ? Icons.error_outline
                                      : Icons.check_circle_outline,
                                  color: _result!.contains('ERROR')
                                      ? Colors.red[700]
                                      : Colors.blue[700],
                                  size: 28,
                                ),
                                SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    'Verification Result',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.black87,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            SizedBox(height: 16),
                            Text(
                              _result!,
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.black87,
                                height: 1.5,
                              ),
                            ),
                          ],
                        ),
                      ),

                    // Action Buttons
                    if (!_isProcessing) ...[
                      SizedBox(height: 32),
                      
                      // Show different buttons based on whether result exists
                      if (_result == null) ...[
                        _buildActionButton(
                          icon: Icons.photo_library_rounded,
                          label: 'Choose from Gallery',
                          color: Colors.blue[600]!,
                          onPressed: () async {
                            XFile? image = await pickImageFromGallery();
                            if (image != null) {
                              await processAndSendImage(image);
                            }
                          },
                        ),
                        SizedBox(height: 16),
                        _buildActionButton(
                          icon: Icons.camera_alt_rounded,
                          label: 'Take a Photo',
                          color: Colors.green[600]!,
                          onPressed: () async {
                            XFile? image = await pickImageFromCamera();
                            if (image != null) {
                              await processAndSendImage(image);
                            }
                          },
                        ),
                      ] else ...[
                        // Show "Check Another Image" button when result is displayed
                        _buildActionButton(
                          icon: Icons.refresh_rounded,
                          label: 'Check Another Image',
                          color: Colors.blue[600]!,
                          onPressed: () {
                            setState(() {
                              _result = null;
                              _imagePath = null;
                            });
                          },
                        ),
                        SizedBox(height: 16),
                        // Optional: Add a secondary button to take new photo
                        OutlinedButton.icon(
                          onPressed: () async {
                            setState(() {
                              _result = null;
                              _imagePath = null;
                            });
                            XFile? image = await pickImageFromCamera();
                            if (image != null) {
                              await processAndSendImage(image);
                            }
                          },
                          icon: Icon(Icons.camera_alt_rounded),
                          label: Text('Take New Photo'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.green[600],
                            side: BorderSide(color: Colors.green[600]!, width: 2),
                            padding: EdgeInsets.symmetric(vertical: 16, horizontal: 24),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                        ),
                      ],
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        padding: EdgeInsets.symmetric(vertical: 16, horizontal: 24),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        elevation: 2,
        shadowColor: color.withOpacity(0.4),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 24),
          SizedBox(width: 12),
          Text(
            label,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> processAndSendImage(XFile image) async {
    print('=== Starting processAndSendImage ===');
    setState(() {
      _isProcessing = true;
      _result = null;
      _imagePath = image.path;
    });
    print('State updated: _isProcessing = true, _imagePath = ${image.path}');

    try {
      final imageBytes = await image.readAsBytes();
      print('Image size: ${imageBytes.length} bytes');

      print('Calling SocketService.sendImageToServer...');
      final result = await SocketService.sendImageToServer(imageBytes);
      print('Received result from server (length: ${result.length})');
      print('Result content: "$result"');

      setState(() {
        _result = result;
        _isProcessing = false;
      });
      print('State updated: _result = "$_result", _isProcessing = false');

    } catch (e, stackTrace) {
      print('!!! Error caught: $e');
      print('Stack trace: $stackTrace');
      
      String errorMessage = e.toString();
      if (errorMessage.contains('TimeoutException')) {
        errorMessage = "Processing timeout. Please try again with a smaller image.";
      } else if (errorMessage.contains('Cannot connect')) {
        errorMessage = "Cannot connect to server. Please check if the server is running.";
      }
      
      setState(() {
        _result = "ERROR: $errorMessage";
        _isProcessing = false;
      });
      print('Error state updated: $_result');
    }
    print('=== Finished processAndSendImage ===');
  }

  Future<XFile?> pickImageFromGallery() async {
    final ImagePicker _picker = ImagePicker();
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (pickedFile == null) return null;

      if (await pickedFile.length() >= 10485760) {
        Fluttertoast.showToast(
          msg: "File size should be less than 10MB.",
          backgroundColor: Colors.orange,
          toastLength: Toast.LENGTH_LONG,
        );
        return null;
      }
      
      return pickedFile;
    } catch (e, stacktrace) {
      print(e);
      print(stacktrace);
      Fluttertoast.showToast(
        msg: "Failed to pick image",
        backgroundColor: Colors.red,
      );
      return null;
    }
  }

  Future<XFile?> pickImageFromCamera() async {
    final ImagePicker _picker = ImagePicker();
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (pickedFile == null) return null;

      if (await pickedFile.length() >= 10485760) {
        Fluttertoast.showToast(
          msg: "File size should be less than 10MB.",
          backgroundColor: Colors.orange,
          toastLength: Toast.LENGTH_LONG,
        );
        return null;
      }
      
      return pickedFile;
    } catch (e, stacktrace) {
      print(e);
      print(stacktrace);
      Fluttertoast.showToast(
        msg: "Failed to capture image",
        backgroundColor: Colors.red,
      );
      return null;
    }
  }
}
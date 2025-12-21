import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'socket_service.dart'; // Import the socket service

class HomeScreen extends StatefulWidget {
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isProcessing = false;
  String? _result;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Barnoda"),
        centerTitle: true,
        backgroundColor: Colors.white,
      ),
      backgroundColor: Colors.white,
      body: Column(
        children: [
          if (_result != null)
            Container(
              margin: EdgeInsets.all(16),
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                children: [
                  Text(
                    'Server Response:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text(_result!),
                ],
              ),
            ),
          Expanded(
            child: Center(
              child: _isProcessing
                  ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text("Processing image..."),
                      ],
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        ElevatedButton(
                          onPressed: () async {
                            XFile? image = await pickImageFromGallery();
                            if (image != null) {
                              await processAndSendImage(image);
                            }
                          },
                          child: Text("Pick an image from gallery"),
                        ),
                        SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: () async {
                            XFile? image = await pickImageFromCamera();
                            if (image != null) {
                              await processAndSendImage(image);
                            }
                          },
                          child: Text("Take a photo"),
                        ),
                      ],
                    ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> processAndSendImage(XFile image) async {
    setState(() {
      _isProcessing = true;
      _result = null;
    });

    try {
      // Read image bytes
      final imageBytes = await image.readAsBytes();
      print('Image size: ${imageBytes.length} bytes');

      // Send to server
      final result = await SocketService.sendImageToServer(imageBytes);

      setState(() {
        _result = result;
      });

      Fluttertoast.showToast(
        msg: "Image processed successfully!",
        backgroundColor: Colors.green,
      );
    } catch (e) {
      print('Error: $e');
      Fluttertoast.showToast(
        msg: "Error: ${e.toString()}",
        backgroundColor: Colors.red,
      );
      setState(() {
        _result = "Error: ${e.toString()}";
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
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
          msg: "Your File's size should be less than 10MB.",
        );
        return null;
      } else {
        return pickedFile;
      }
    } catch (e, stacktrace) {
      print(e);
      print(stacktrace);
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
          msg: "Your File's size should be less than 10MB.",
        );
        return null;
      } else {
        return pickedFile;
      }
    } catch (e, stacktrace) {
      print(e);
      print(stacktrace);
      return null;
    }
  }
}
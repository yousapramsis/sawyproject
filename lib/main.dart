import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For rootBundle
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib; // Prefix to avoid conflicts

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SkinSense AI',
      theme: ThemeData(
        primarySwatch: Colors.teal, // A nice primary color
        hintColor: Colors.orangeAccent, // Accent color
        brightness: Brightness.light,
        fontFamily: 'Roboto', // A clean font
        textTheme: const TextTheme(
          headlineSmall: TextStyle(
              fontSize: 24.0, fontWeight: FontWeight.bold, color: Colors.teal),
          titleLarge: TextStyle(
              fontSize: 20.0,
              fontStyle: FontStyle.italic,
              color: Colors.black87),
          bodyMedium: TextStyle(fontSize: 16.0, color: Colors.black54),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.teal, // Button background
            foregroundColor: Colors.white, // Button text color
            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
            textStyle: const TextStyle(fontSize: 18),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 4.0,
          margin: const EdgeInsets.all(10.0),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15.0),
          ),
        ),
      ),
      home: const SkinDiseasePredictorPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SkinDiseasePredictorPage extends StatefulWidget {
  const SkinDiseasePredictorPage({super.key});

  @override
  State<SkinDiseasePredictorPage> createState() =>
      _SkinDiseasePredictorPageState();
}

class _SkinDiseasePredictorPageState extends State<SkinDiseasePredictorPage> {
  File? _imageFile;
  Interpreter? _interpreter;
  List<String>? _labels;
  Map<String, dynamic>? _predictionResult;
  bool _isLoading = false;

  // --- IMPORTANT: Adjust these based on your model ---
  final int _inputSize = 224; // e.g., 224 for MobileNet
  final double _mean = 0; // For normalization if needed, e.g. 127.5
  final double _std =
      255.0; // For normalization if needed, e.g. 127.5 or 1.0 if already [0,1]
  // ----------------------------------------------------

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      // Load the TFLite model
      _interpreter =
          await Interpreter.fromAsset('assets/SkinSenseModel.tflite');
      _interpreter!.allocateTensors(); // Allocate memory for tensors

      // Load labels
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData
          .split('\n')
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      if (_labels == null || _labels!.isEmpty) {
        print("Error: Labels file is empty or could not be loaded.");
      }
      if (_interpreter == null) {
        print("Error: Model could not be loaded.");
      }
    } catch (e) {
      print("Error loading model or labels: $e");
      setState(() {
        _predictionResult = {'error': 'Failed to load model/labels: $e'};
      });
    }
  }

  Future<void> _pickImage() async {
    if (_isLoading) return;
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _predictionResult = null; // Clear previous results
        _isLoading = true;
      });
      await _runInference();
    }
  }

  // Preprocess the image and run inference
  Future<void> _runInference() async {
    if (_imageFile == null || _interpreter == null || _labels == null) {
      setState(() {
        _isLoading = false;
        _predictionResult = {
          'error': 'Model/Labels not loaded or no image selected'
        };
      });
      return;
    }

    try {
      // 1. Decode Image
      img_lib.Image? image =
          img_lib.decodeImage(await _imageFile!.readAsBytes());
      if (image == null) {
        throw Exception("Could not decode image");
      }

      // 2. Resize image to model's expected input size
      img_lib.Image resizedImage =
          img_lib.copyResize(image, width: _inputSize, height: _inputSize);

      // 3. Convert to Float32List and normalize
      //    Input shape for many models is [1, height, width, 3]
      //    Values are typically normalized (e.g., to [0,1] or [-1,1])
      var inputBytes = Float32List(1 * _inputSize * _inputSize * 3);
      var bufferIndex = 0;
      for (var y = 0; y < _inputSize; y++) {
        for (var x = 0; x < _inputSize; x++) {
          var pixel = resizedImage.getPixel(x, y);
          // Normalize pixel values. Adjust this based on your model's training!
          // Example: (pixel_value - mean) / std
          inputBytes[bufferIndex++] = (pixel.r - _mean) / _std;
          inputBytes[bufferIndex++] = (pixel.g - _mean) / _std;
          inputBytes[bufferIndex++] = (pixel.b - _mean) / _std;
        }
      }
      // Reshape to [1, _inputSize, _inputSize, 3]
      final input = inputBytes.reshape([1, _inputSize, _inputSize, 3]);

      // Output shape: Usually [1, num_classes]
      // Assuming your model has a single output tensor for classification probabilities
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final numClasses =
          outputShape[1]; // e.g., if shape is [1, 7], then numClasses is 7
      final output = List.filled(1 * numClasses, 0.0).reshape([1, numClasses]);

      // Run inference
      _interpreter!.run(input, output);

      // Process output
      final List<double> probabilities = output[0] as List<double>;
      double maxConfidence = 0.0;
      int bestIndex = -1;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxConfidence) {
          maxConfidence = probabilities[i];
          bestIndex = i;
        }
      }

      if (bestIndex != -1 && bestIndex < _labels!.length) {
        setState(() {
          _predictionResult = {
            'predicted_label': _labels![bestIndex],
            'confidence': double.parse(
                maxConfidence.toStringAsFixed(2)), // Format to 2 decimal places
          };
        });
      } else {
        throw Exception(
            "Error processing output or label index out of bounds. Probabilities: $probabilities, Labels count: ${_labels!.length}");
      }
    } catch (e) {
      print("Error during inference: $e");
      setState(() {
        _predictionResult = {'error': 'Inference error: $e'};
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SkinSense AI Analyzer'),
        backgroundColor: Theme.of(context).primaryColor,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Text(
                'Upload a Photo of Skin Condition',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Center(
                child: _imageFile == null
                    ? Container(
                        width: 250,
                        height: 250,
                        decoration: BoxDecoration(
                          color: Colors.grey[300],
                          borderRadius: BorderRadius.circular(15),
                          border: Border.all(
                              color: Theme.of(context).primaryColor, width: 2),
                        ),
                        child: Icon(Icons.image_search,
                            size: 100,
                            color: Theme.of(context).primaryColorDark),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(15.0),
                        child: Image.file(
                          _imageFile!,
                          width: 250,
                          height: 250,
                          fit: BoxFit.cover,
                        ),
                      ),
              ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                icon: const Icon(Icons.photo_library),
                label: const Text('Upload Photo'),
                onPressed: _pickImage,
              ),
              const SizedBox(height: 30),
              if (_isLoading) const Center(child: CircularProgressIndicator()),
              if (_predictionResult != null && !_isLoading) _buildResultCard(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    if (_predictionResult == null) return const SizedBox.shrink();

    if (_predictionResult!['error'] != null) {
      return Card(
        color: Colors.red[100],
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            'Error: ${_predictionResult!['error']}',
            style: TextStyle(color: Colors.red[800], fontSize: 16),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return Card(
      elevation: 5,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: Container(
        padding: const EdgeInsets.all(20.0),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(15),
          gradient: LinearGradient(
            colors: [
              Theme.of(context).primaryColor.withOpacity(0.7),
              Theme.of(context).hintColor.withOpacity(0.7)
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Prediction Result:',
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 15),
            _buildResultRow('Condition:',
                '${_predictionResult!['predicted_label']}', context),
            const SizedBox(height: 10),
            _buildResultRow(
                'Confidence:',
                '${(_predictionResult!['confidence'] * 100).toStringAsFixed(1)}%',
                context),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildResultRow(String title, String value, BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title,
            style: Theme.of(context)
                .textTheme
                .bodyMedium
                ?.copyWith(color: Colors.white70, fontWeight: FontWeight.w600)),
        Text(value,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 17)),
      ],
    );
  }
}

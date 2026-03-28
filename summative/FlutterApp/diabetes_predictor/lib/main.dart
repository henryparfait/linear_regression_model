import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() {
  runApp(const DiabetesPredictionApp());
}

class DiabetesPredictionApp extends StatelessWidget {
  const DiabetesPredictionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Diabetes Progression Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF0D6E56),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _resultMessage;
  bool _isError = false;

  // Controllers for all 9 input features
  final _ageController = TextEditingController();
  final _sexController = TextEditingController();
  final _bmiController = TextEditingController();
  final _bpController = TextEditingController();
  final _s2Controller = TextEditingController();
  final _s3Controller = TextEditingController();
  final _s4Controller = TextEditingController();
  final _s5Controller = TextEditingController();
  final _s6Controller = TextEditingController();

  // Feature metadata: name, controller, min, max, hint
  late final List<Map<String, dynamic>> _features;

  @override
  void initState() {
    super.initState();
    _features = [
      {
        'name': 'Age',
        'key': 'age',
        'controller': _ageController,
        'min': -0.15,
        'max': 0.15,
        'hint': 'e.g. 0.0453',
        'description': 'Normalized age',
      },
      {
        'name': 'Sex',
        'key': 'sex',
        'controller': _sexController,
        'min': -0.07,
        'max': 0.07,
        'hint': 'e.g. -0.0447',
        'description': 'Normalized sex indicator',
      },
      {
        'name': 'BMI',
        'key': 'bmi',
        'controller': _bmiController,
        'min': -0.10,
        'max': 0.20,
        'hint': 'e.g. -0.0058',
        'description': 'Normalized body mass index',
      },
      {
        'name': 'BP',
        'key': 'bp',
        'controller': _bpController,
        'min': -0.15,
        'max': 0.15,
        'hint': 'e.g. -0.0159',
        'description': 'Normalized blood pressure',
      },
      {
        'name': 'S2 (LDL)',
        'key': 's2',
        'controller': _s2Controller,
        'min': -0.20,
        'max': 0.25,
        'hint': 'e.g. -0.0037',
        'description': 'Low-density lipoproteins',
      },
      {
        'name': 'S3 (HDL)',
        'key': 's3',
        'controller': _s3Controller,
        'min': -0.15,
        'max': 0.20,
        'hint': 'e.g. 0.0081',
        'description': 'High-density lipoproteins',
      },
      {
        'name': 'S4 (TCH)',
        'key': 's4',
        'controller': _s4Controller,
        'min': -0.15,
        'max': 0.20,
        'hint': 'e.g. -0.0396',
        'description': 'Cholesterol / HDL ratio',
      },
      {
        'name': 'S5 (LTG)',
        'key': 's5',
        'controller': _s5Controller,
        'min': -0.20,
        'max': 0.20,
        'hint': 'e.g. -0.0031',
        'description': 'Log of serum triglycerides',
      },
      {
        'name': 'S6 (GLU)',
        'key': 's6',
        'controller': _s6Controller,
        'min': -0.15,
        'max': 0.20,
        'hint': 'e.g. 0.0112',
        'description': 'Blood sugar level',
      },
    ];
  }

  @override
  void dispose() {
    _ageController.dispose();
    _sexController.dispose();
    _bmiController.dispose();
    _bpController.dispose();
    _s2Controller.dispose();
    _s3Controller.dispose();
    _s4Controller.dispose();
    _s5Controller.dispose();
    _s6Controller.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _resultMessage = null;
      _isError = false;
    });

    try {
      // Build the request body
      final Map<String, dynamic> body = {};
      for (var feature in _features) {
        body[feature['key']] =
            double.parse((feature['controller'] as TextEditingController).text);
      }

      final response = await http.post(
        Uri.parse(
            'https://diabetes-prediction-api-p7cy.onrender.com/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _resultMessage =
              'Predicted Disease Progression: ${data['prediction']}\n'
              'Model: ${data['model_used']}';
          _isError = false;
        });
      } else if (response.statusCode == 422) {
        final data = jsonDecode(response.body);
        final details = data['detail'] as List;
        final messages =
            details.map((d) => '${d['loc'].last}: ${d['msg']}').join('\n');
        setState(() {
          _resultMessage = 'Validation Error:\n$messages';
          _isError = true;
        });
      } else {
        setState(() {
          _resultMessage = 'Server Error (${response.statusCode})';
          _isError = true;
        });
      }
    } catch (e) {
      setState(() {
        _resultMessage = 'Connection Error:\n${e.toString()}';
        _isError = true;
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  String? _validateField(String? value, Map<String, dynamic> feature) {
    if (value == null || value.trim().isEmpty) {
      return '${feature['name']} is required';
    }
    final number = double.tryParse(value);
    if (number == null) {
      return 'Enter a valid number';
    }
    if (number < feature['min'] || number > feature['max']) {
      return 'Range: ${feature['min']} to ${feature['max']}';
    }
    return null;
  }

  void _fillExample() {
    final examples = [0.0453, -0.0447, -0.0058, -0.0159, -0.0037, 0.0081, -0.0396, -0.0031, 0.0112];
    for (int i = 0; i < _features.length; i++) {
      (_features[i]['controller'] as TextEditingController).text =
          examples[i].toString();
    }
  }

  void _clearAll() {
    for (var feature in _features) {
      (feature['controller'] as TextEditingController).clear();
    }
    setState(() {
      _resultMessage = null;
      _isError = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diabetes Progression Predictor'),
        centerTitle: true,
        backgroundColor: colorScheme.primaryContainer,
        foregroundColor: colorScheme.onPrimaryContainer,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header card
                Card(
                  color: colorScheme.primaryContainer.withOpacity(0.3),
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        Icon(Icons.health_and_safety,
                            size: 36, color: colorScheme.primary),
                        const SizedBox(height: 8),
                        Text(
                          'Enter Patient Clinical Data',
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'All 9 normalized features are required',
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: colorScheme.onSurfaceVariant),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 12),

                // Quick action buttons
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _fillExample,
                        icon: const Icon(Icons.auto_fix_high, size: 18),
                        label: const Text('Fill Example'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _clearAll,
                        icon: const Icon(Icons.clear_all, size: 18),
                        label: const Text('Clear All'),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 16),

                // Input fields
                ..._features.map((feature) => Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: TextFormField(
                        controller:
                            feature['controller'] as TextEditingController,
                        keyboardType: const TextInputType.numberWithOptions(
                            decimal: true, signed: true),
                        decoration: InputDecoration(
                          labelText: feature['name'],
                          hintText: feature['hint'],
                          helperText: '${feature['description']}  •  Range: ${feature['min']} to ${feature['max']}',
                          helperMaxLines: 2,
                          border: const OutlineInputBorder(),
                          prefixIcon:
                              const Icon(Icons.science_outlined, size: 20),
                          filled: true,
                          fillColor:
                              colorScheme.surfaceContainerHighest.withOpacity(0.3),
                        ),
                        validator: (value) =>
                            _validateField(value, feature),
                      ),
                    )),

                const SizedBox(height: 8),

                // Predict button
                SizedBox(
                  height: 52,
                  child: FilledButton.icon(
                    onPressed: _isLoading ? null : _predict,
                    icon: _isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.analytics),
                    label: Text(
                      _isLoading ? 'Predicting...' : 'Predict',
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Result display area
                if (_resultMessage != null)
                  Card(
                    color: _isError
                        ? colorScheme.errorContainer
                        : colorScheme.secondaryContainer,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: BorderSide(
                        color: _isError
                            ? colorScheme.error.withOpacity(0.3)
                            : colorScheme.secondary.withOpacity(0.3),
                      ),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                _isError
                                    ? Icons.error_outline
                                    : Icons.check_circle_outline,
                                color: _isError
                                    ? colorScheme.error
                                    : colorScheme.secondary,
                                size: 22,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                _isError ? 'Error' : 'Prediction Result',
                                style: TextStyle(
                                  fontWeight: FontWeight.w600,
                                  fontSize: 15,
                                  color: _isError
                                      ? colorScheme.onErrorContainer
                                      : colorScheme.onSecondaryContainer,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Text(
                            _resultMessage!,
                            style: TextStyle(
                              fontSize: 14,
                              height: 1.5,
                              color: _isError
                                  ? colorScheme.onErrorContainer
                                  : colorScheme.onSecondaryContainer,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
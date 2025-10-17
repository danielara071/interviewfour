# Geometric Shape Classifier

A Python program that classifies geometric shapes using computational geometry and topology (no machine learning or deep learning). The classifier identifies shapes as Rectangle, Ellipse, Triangle, or Other.

## Features

- **Pure Computational Geometry**: Uses geometric properties like vertices, angles, convexity, and eccentricity
- **Performance Analysis**: Reports processing time as a function of image size (m) and number of images (n)
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix
- **Complexity Analysis**: Analyzes asymptotic complexity with different image sizes
- **Visual Results**: Generates plots showing performance metrics and complexity analysis

## Algorithm Overview

The classifier uses the following geometric features:

1. **Contour Detection**: Finds external contours in binary images
2. **Geometric Properties**:
   - Number of vertices (from polygon approximation)
   - Angles between consecutive vertices
   - Solidity (ratio of contour area to convex hull area)
   - Circularity (4π×area/perimeter²)
   - Eccentricity (from second-order central moments)
   - Aspect ratio and extent
   - Convexity defects

3. **Classification Rules**:
   - **Triangle**: 3 vertices with high solidity (>0.85)
   - **Rectangle**: 4 vertices with high solidity and right angles (80-100°)
   - **Ellipse**: High circularity (>0.7) or smooth elongated shape
   - **Other**: Shapes that don't match the above criteria

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset is downloaded in the `data/` directory

## Usage

Run the main script:
```bash
python main.py
```

This will:
- Load the dataset
- Evaluate the classifier on test images
- Analyze computational complexity
- Generate performance plots
- Save results to `classification_results.txt`

## Dataset Structure

The program expects the following directory structure:
```
data/
└── user.if2/
    └── images/
        ├── rectangle/
        ├── triangle/
        ├── ellipse/
        └── other/
```

## Performance Metrics

The classifier reports:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results
- **Processing Time**: Time complexity analysis
- **Complexity Analysis**: Performance vs image size

## Computational Complexity

The algorithm has the following complexity characteristics:
- **Image Preprocessing**: O(m²) where m is image size
- **Contour Detection**: O(m²) in worst case
- **Feature Extraction**: O(k) where k is number of contour points
- **Classification**: O(1) - constant time decision rules

Overall complexity: **O(m²)** per image

## Output Files

- `classification_results.txt`: Detailed performance metrics
- `classification_results.png`: Visualization plots
- Console output: Real-time progress and summary

## Example Results

```
Overall Accuracy: 0.8750
Average Processing Time: 0.0234 seconds
Total Images Processed: 400

Class-wise Performance:
Rectangle:
  Precision: 0.9000
  Recall: 0.8500
  F1-Score: 0.8744

Triangle:
  Precision: 0.8200
  Recall: 0.9000
  F1-Score: 0.8583

Ellipse:
  Precision: 0.8800
  Recall: 0.8800
  F1-Score: 0.8800

Other:
  Precision: 0.9200
  Recall: 0.9200
  F1-Score: 0.9200
```

## Technical Details

The classifier uses OpenCV for image processing and contour analysis, NumPy for numerical computations, and scikit-learn for evaluation metrics. The geometric approach makes it robust to variations in drawing style and image quality while maintaining fast processing times.

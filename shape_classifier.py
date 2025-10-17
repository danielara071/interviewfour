"""
Geometric Shape Classifier using Computational Geometry and Topology
Classifies shapes into: Rectangle, Ellipse, Triangle, Other
"""

import cv2
import numpy as np
import os
import time
import random
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class GeometricShapeClassifier:
    """
    A classifier that uses computational geometry and topology to classify shapes.
    No machine learning or deep learning is used - only geometric properties.
    """
    
    def __init__(self):
        self.classes = ['rectangle', 'triangle', 'ellipse', 'other']
        self.performance_metrics = {}
        
    def preprocess_image(self, image_path: str, target_size: int = 100) -> np.ndarray:
        """
        Preprocess image: load, convert to grayscale, resize, and binarize.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing (m x m pixels)
            
        Returns:
            Preprocessed binary image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        resized = cv2.resize(gray, (target_size, target_size))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Try multiple thresholding methods and choose the best one
        # Method 1: Otsu's thresholding
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Method 3: Simple thresholding
        _, binary_simple = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Choose the method that gives the most reasonable contour
        # (not too fragmented, not too simple)
        contours_otsu, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_adaptive, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_simple, _ = cv2.findContours(binary_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Choose the method with the most reasonable number of contours
        # (prefer 1-3 contours, not too many fragments)
        if 1 <= len(contours_otsu) <= 3:
            binary = binary_otsu
        elif 1 <= len(contours_adaptive) <= 3:
            binary = binary_adaptive
        elif 1 <= len(contours_simple) <= 3:
            binary = binary_simple
        else:
            # If all methods give too many contours, use the one with the fewest
            min_contours = min(len(contours_otsu), len(contours_adaptive), len(contours_simple))
            if len(contours_otsu) == min_contours:
                binary = binary_otsu
            elif len(contours_adaptive) == min_contours:
                binary = binary_adaptive
            else:
                binary = binary_simple
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the binary image.
        
        Args:
            binary_image: Binary image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out very small contours
        min_area = 100
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return filtered_contours
    
    def extract_geometric_features(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Extract geometric features from a contour using computational geometry.
        
        Args:
            contour: Contour points
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        # Basic geometric properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Extent (ratio of contour area to bounding rectangle area)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        num_defects = len(defects) if defects is not None else 0
        
        # Approximate contour to polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Calculate angles between consecutive vertices
        angles = self._calculate_angles(approx)
        
        # Circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Eccentricity using moments
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Calculate second order central moments
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            
            # Calculate eigenvalues for eccentricity
            a = mu20 + mu02
            b = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
            eccentricity = np.sqrt(1 - (a - b) / (a + b)) if (a + b) > 0 else 0
        else:
            eccentricity = 0
        
        # Store all features
        features.update({
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'num_defects': num_defects,
            'num_vertices': num_vertices,
            'angles': angles,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'approx_contour': approx
        })
        
        return features
    
    def _calculate_angles(self, vertices: np.ndarray) -> List[float]:
        """
        Calculate angles between consecutive vertices.
        
        Args:
            vertices: Array of vertex points
            
        Returns:
            List of angles in degrees
        """
        if len(vertices) < 3:
            return []
        
        angles = []
        for i in range(len(vertices)):
            p1 = vertices[i - 1][0]
            p2 = vertices[i][0]
            p3 = vertices[(i + 1) % len(vertices)][0]
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        
        return angles
    
    def classify_shape(self, features: Dict[str, Any]) -> str:
        """
        Classify shape based on geometric features using computational geometry rules.
        Optimized for hand-drawn outline shapes.
        
        Args:
            features: Dictionary of geometric features
            
        Returns:
            Predicted shape class
        """
        num_vertices = features['num_vertices']
        solidity = features['solidity']
        circularity = features['circularity']
        eccentricity = features['eccentricity']
        extent = features['extent']
        angles = features['angles']
        aspect_ratio = features['aspect_ratio']
        area = features['area']
        perimeter = features['perimeter']
        
        # Classification rules optimized for hand-drawn outline shapes
        
        # 1. Check for Ellipse first (these work well with current features)
        if solidity > 0.8 and circularity > 0.5:
            # High solidity and circularity indicates ellipse
            if eccentricity < 0.5:
                return 'ellipse'
            elif eccentricity > 0.6 and solidity > 0.9:
                return 'ellipse'
        
        # Additional ellipse detection for smooth shapes
        if solidity > 0.9 and num_vertices >= 6:
            # Very smooth shape with many vertices - likely ellipse
            if circularity > 0.3:
                return 'ellipse'
        
        # 2. Check for Triangle (adapted for outline shapes)
        if num_vertices <= 8:  # Allow more vertices for rough triangles
            if len(angles) >= 3:
                # Look for three main angles that sum to approximately 180 degrees
                main_angles = sorted(angles, reverse=True)[:3]
                angle_sum = sum(main_angles)
                if 140 <= angle_sum <= 220:  # Allow tolerance
                    # Check for triangular characteristics
                    # Should have at least one acute angle
                    acute_angles = sum(1 for angle in main_angles if angle < 90)
                    if acute_angles >= 1:
                        # Additional check: should not be too rectangular
                        right_angles = sum(1 for angle in main_angles if 80 <= angle <= 100)
                        if right_angles <= 1:  # Not too many right angles
                            return 'triangle'
        
        # 3. Check for Rectangle (adapted for outline shapes)
        if num_vertices <= 10:  # Allow more vertices for rough rectangles
            if len(angles) >= 4:
                # Check if angles are close to 90 degrees
                right_angle_count = sum(1 for angle in angles if 70 <= angle <= 110)
                if right_angle_count >= 2:  # At least 2 right angles
                    # Check for rectangular characteristics
                    # Should have reasonable aspect ratio
                    if 0.3 <= aspect_ratio <= 3.0:
                        # Check if it's not too triangular
                        acute_angles = sum(1 for angle in angles if angle < 60)
                        if acute_angles <= 2:  # Not too many acute angles
                            return 'rectangle'
        
        # 4. Rectangle detection using extent and aspect ratio
        if extent > 0.4:  # Lower threshold for outline shapes
            if 0.3 <= aspect_ratio <= 3.0:
                if num_vertices <= 8:
                    # Check for rectangular-like properties
                    right_angle_count = sum(1 for angle in angles if 60 <= angle <= 120)
                    if right_angle_count >= 2:
                        return 'rectangle'
        
        # 5. Triangle detection using angle analysis
        if num_vertices <= 7:
            if len(angles) >= 3:
                # Look for triangular angle patterns
                main_angles = sorted(angles, reverse=True)[:3]
                angle_sum = sum(main_angles)
                if 150 <= angle_sum <= 210:
                    # Check for triangular characteristics
                    acute_angles = sum(1 for angle in main_angles if angle < 90)
                    if acute_angles >= 1:
                        # Check if it's not too rectangular
                        right_angles = sum(1 for angle in main_angles if 80 <= angle <= 100)
                        if right_angles <= 1:
                            return 'triangle'
        
        # 6. Ellipse detection for smooth shapes with moderate solidity
        if solidity > 0.6 and num_vertices >= 6:
            if circularity > 0.4:
                # Check if it's not too angular
                sharp_angles = sum(1 for angle in angles if angle < 60 or angle > 120)
                if sharp_angles <= 3:  # Not too many sharp angles
                    return 'ellipse'
        
        # 7. Fallback classifications based on dominant characteristics
        
        # If it has many vertices and is smooth, likely ellipse
        if num_vertices >= 8 and solidity > 0.5:
            if circularity > 0.3:
                return 'ellipse'
        
        # If it has few vertices and right angles, likely rectangle
        if num_vertices <= 6:
            right_angles = sum(1 for angle in angles if 70 <= angle <= 110)
            if right_angles >= 2:
                return 'rectangle'
        
        # If it has few vertices and acute angles, likely triangle
        if num_vertices <= 6:
            acute_angles = sum(1 for angle in angles if angle < 90)
            if acute_angles >= 2:
                return 'triangle'
        
        # Default to 'other' if no specific shape is detected
        return 'other'
    
    def classify_image(self, image_path: str, target_size: int = 100) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a single image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
            
        Returns:
            Tuple of (predicted_class, processing_time, features)
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            binary_img = self.preprocess_image(image_path, target_size)
            
            # Find contours
            contours = self.find_contours(binary_img)
            
            if not contours:
                return 'other', time.time() - start_time, {}
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract features
            features = self.extract_geometric_features(largest_contour)
            
            # Classify
            predicted_class = self.classify_shape(features)
            
            processing_time = time.time() - start_time
            return predicted_class, processing_time, features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 'other', time.time() - start_time, {}
    
    def load_dataset(self, data_dir: str, max_samples_per_class: int = 100) -> List[Tuple[str, str]]:
        """
        Load dataset from directory structure.
        
        Args:
            data_dir: Path to data directory
            max_samples_per_class: Maximum samples to load per class
            
        Returns:
            List of (image_path, true_label) tuples
        """
        dataset = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_dir, 'user.if2', 'images', class_name)
            if not os.path.exists(class_path):
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            # Sample randomly if too many files
            if len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
            
            # Add to dataset
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                dataset.append((img_path, class_name))
        
        return dataset
    
    def evaluate_classifier(self, dataset: List[Tuple[str, str]], target_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            dataset: List of (image_path, true_label) tuples
            target_size: Target size for resizing
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        true_labels = []
        processing_times = []
        
        print(f"Evaluating classifier on {len(dataset)} images...")
        
        for i, (img_path, true_label) in enumerate(dataset):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(dataset)}")
            
            pred_class, proc_time, _ = self.classify_image(img_path, target_size)
            
            predictions.append(pred_class)
            true_labels.append(true_label)
            processing_times.append(proc_time)
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
        avg_processing_time = np.mean(processing_times)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=self.classes)
        
        # Classification report
        report = classification_report(true_labels, predictions, labels=self.classes, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'avg_processing_time': avg_processing_time,
            'processing_times': processing_times,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return results
    
    def analyze_complexity(self, data_dir: str, sizes: List[int] = [50, 100, 150, 200]) -> Dict[str, Any]:
        """
        Analyze computational complexity as a function of image size.
        
        Args:
            data_dir: Path to data directory
            sizes: List of image sizes to test
            
        Returns:
            Dictionary with complexity analysis
        """
        # Load a small sample for testing
        dataset = self.load_dataset(data_dir, max_samples_per_class=20)
        
        complexity_results = {}
        
        for size in sizes:
            print(f"Testing with image size {size}x{size}")
            
            times = []
            for img_path, _ in dataset[:50]:  # Test on first 50 images
                _, proc_time, _ = self.classify_image(img_path, size)
                times.append(proc_time)
            
            complexity_results[size] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'times': times
            }
        
        return complexity_results
    
    def plot_results(self, results: Dict[str, Any], complexity_results: Dict[str, Any]):
        """
        Plot evaluation results and complexity analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes, ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        
        # Processing Time Distribution
        axes[0,1].hist(results['processing_times'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Processing Time Distribution')
        axes[0,1].set_xlabel('Time (seconds)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(results['avg_processing_time'], color='red', linestyle='--', 
                         label=f'Mean: {results["avg_processing_time"]:.4f}s')
        axes[0,1].legend()
        
        # Complexity Analysis
        sizes = list(complexity_results.keys())
        avg_times = [complexity_results[s]['avg_time'] for s in sizes]
        std_times = [complexity_results[s]['std_time'] for s in sizes]
        
        axes[1,0].errorbar(sizes, avg_times, yerr=std_times, marker='o', capsize=5)
        axes[1,0].set_title('Processing Time vs Image Size')
        axes[1,0].set_xlabel('Image Size (pixels)')
        axes[1,0].set_ylabel('Average Processing Time (seconds)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Class-wise Accuracy
        report = results['classification_report']
        classes = [cls for cls in self.classes if cls in report]
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[1,1].bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        axes[1,1].bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        axes[1,1].set_title('Precision and Recall by Class')
        axes[1,1].set_xlabel('Class')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(classes)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("="*50)
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Average Processing Time: {results['avg_processing_time']:.4f} seconds")
        print(f"Total Images Processed: {len(results['predictions'])}")
        
        print("\nClass-wise Performance:")
        for cls in classes:
            print(f"{cls.capitalize()}:")
            print(f"  Precision: {report[cls]['precision']:.4f}")
            print(f"  Recall: {report[cls]['recall']:.4f}")
            print(f"  F1-Score: {report[cls]['f1-score']:.4f}")
        
        print("\nComplexity Analysis:")
        for size in sizes:
            avg_time = complexity_results[size]['avg_time']
            print(f"Size {size}x{size}: {avg_time:.4f}s average processing time")


def main():
    """
    Main function to run the geometric shape classifier.
    """
    # Initialize classifier
    classifier = GeometricShapeClassifier()
    
    # Set data directory
    data_dir = "data"
    
    print("Loading dataset...")
    dataset = classifier.load_dataset(data_dir, max_samples_per_class=100)
    print(f"Loaded {len(dataset)} images")
    
    # Evaluate classifier
    print("\nEvaluating classifier...")
    results = classifier.evaluate_classifier(dataset, target_size=100)
    
    # Analyze complexity
    print("\nAnalyzing computational complexity...")
    complexity_results = classifier.analyze_complexity(data_dir, sizes=[50, 100, 150, 200])
    
    # Plot results
    print("\nGenerating plots...")
    classifier.plot_results(results, complexity_results)
    
    # Save results to file
    with open('classification_results.txt', 'w') as f:
        f.write("Geometric Shape Classification Results\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Average Processing Time: {results['avg_processing_time']:.4f} seconds\n")
        f.write(f"Total Images Processed: {len(results['predictions'])}\n\n")
        
        f.write("Class-wise Performance:\n")
        for cls in classifier.classes:
            if cls in results['classification_report']:
                report = results['classification_report'][cls]
                f.write(f"{cls.capitalize()}:\n")
                f.write(f"  Precision: {report['precision']:.4f}\n")
                f.write(f"  Recall: {report['recall']:.4f}\n")
                f.write(f"  F1-Score: {report['f1-score']:.4f}\n\n")
        
        f.write("Complexity Analysis:\n")
        for size in [50, 100, 150, 200]:
            if size in complexity_results:
                avg_time = complexity_results[size]['avg_time']
                f.write(f"Size {size}x{size}: {avg_time:.4f}s average processing time\n")
    
    print("\nResults saved to 'classification_results.txt'")
    print("Plots saved to 'classification_results.png'")


if __name__ == "__main__":
    main()

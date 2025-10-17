"""
Main script to run the Geometric Shape Classifier
"""

from shape_classifier import GeometricShapeClassifier
import os

def main():
    """
    Main function to run the geometric shape classifier.
    """
    print("Geometric Shape Classifier using Computational Geometry")
    print("="*60)
    
    # Initialize classifier
    classifier = GeometricShapeClassifier()
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Data directory not found. Please ensure the dataset is downloaded.")
        return
    
    print("Loading dataset...")
    dataset = classifier.load_dataset(data_dir, max_samples_per_class=100)
    print(f"Loaded {len(dataset)} images")
    
    if len(dataset) == 0:
        print("No images found in dataset. Please check the data directory structure.")
        return
    
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
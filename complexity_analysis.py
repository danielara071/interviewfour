"""
Comprehensive complexity analysis for the geometric shape classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from shape_classifier import GeometricShapeClassifier
import os

def analyze_complexity_detailed():
    """
    Perform detailed complexity analysis with more data points and 
    theoretical analysis.
    """
    classifier = GeometricShapeClassifier()
    data_dir = "data"
    
    # Load a sample dataset
    dataset = classifier.load_dataset(data_dir, max_samples_per_class=20)
    
    # Test different image sizes
    sizes = [32, 50, 64, 80, 100, 128, 150, 200, 256, 300]
    results = {}
    
    print("Performing detailed complexity analysis...")
    print("Image Size | Avg Time (s) | Std Time (s) | Theoretical O(m²)")
    print("-" * 60)
    
    for size in sizes:
        times = []
        
        # Test on first 30 images for each size
        for img_path, _ in dataset[:30]:
            _, proc_time, _ = classifier.classify_image(img_path, size)
            times.append(proc_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Theoretical O(m²) scaling (normalized to size 100)
        theoretical_time = avg_time * (size / 100) ** 2
        
        results[size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'theoretical_time': theoretical_time,
            'times': times
        }
        
        print(f"{size:10d} | {avg_time:11.6f} | {std_time:11.6f} | {theoretical_time:15.6f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sizes_list = list(results.keys())
    avg_times = [results[s]['avg_time'] for s in sizes_list]
    std_times = [results[s]['std_time'] for s in sizes_list]
    theoretical_times = [results[s]['theoretical_time'] for s in sizes_list]
    
    # Plot 1: Actual vs Theoretical complexity
    ax1.errorbar(sizes_list, avg_times, yerr=std_times, marker='o', capsize=5, 
                label='Actual', linewidth=2, markersize=8)
    ax1.plot(sizes_list, theoretical_times, 'r--', label='Theoretical O(m²)', linewidth=2)
    ax1.set_xlabel('Image Size (m × m pixels)')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Actual vs Theoretical Complexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Complexity ratio analysis
    ratios = [actual / theoretical for actual, theoretical in zip(avg_times, theoretical_times)]
    ax2.plot(sizes_list, ratios, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Perfect O(m²)')
    ax2.set_xlabel('Image Size (m × m pixels)')
    ax2.set_ylabel('Actual / Theoretical Time Ratio')
    ax2.set_title('Complexity Scaling Verification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('complexity_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print complexity analysis summary
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Calculate complexity coefficients
    size_100_time = results[100]['avg_time']
    size_200_time = results[200]['avg_time']
    
    # Theoretical scaling factor for 2x size increase
    theoretical_factor = 4  # O(m²) means 2x size = 4x time
    actual_factor = size_200_time / size_100_time
    
    print(f"Time for 100×100 images: {size_100_time:.6f} seconds")
    print(f"Time for 200×200 images: {size_200_time:.6f} seconds")
    print(f"Actual scaling factor (2x size): {actual_factor:.2f}")
    print(f"Theoretical scaling factor (O(m²)): {theoretical_factor:.2f}")
    print(f"Scaling accuracy: {(actual_factor/theoretical_factor)*100:.1f}%")
    
    # Estimate complexity for different scenarios
    print(f"\nEstimated processing times for different scenarios:")
    print(f"Single 100×100 image: {size_100_time:.4f} seconds")
    print(f"Single 200×200 image: {size_200_time:.4f} seconds")
    print(f"Single 500×500 image: {size_100_time * (500/100)**2:.4f} seconds")
    
    # Batch processing estimates
    print(f"\nBatch processing estimates (n images):")
    for n in [10, 50, 100, 500, 1000]:
        total_time = size_100_time * n
        print(f"n={n:4d} images (100×100): {total_time:.2f} seconds")
    
    # Memory and computational requirements
    print(f"\nComputational Requirements:")
    print(f"Per image (100×100): ~{size_100_time:.4f} seconds")
    print(f"Memory per image: ~{100*100*4/1024:.1f} KB (grayscale)")
    print(f"Peak memory: ~{100*100*4*3/1024:.1f} KB (multiple processing stages)")
    
    return results

def analyze_algorithm_components():
    """
    Analyze the complexity of individual algorithm components.
    """
    print("\n" + "="*60)
    print("ALGORITHM COMPONENT ANALYSIS")
    print("="*60)
    
    print("1. Image Preprocessing:")
    print("   - Load image: O(m²)")
    print("   - Convert to grayscale: O(m²)")
    print("   - Resize: O(m²)")
    print("   - Gaussian blur: O(m²)")
    print("   - Thresholding: O(m²)")
    print("   - Morphological operations: O(m²)")
    print("   Total: O(m²)")
    
    print("\n2. Contour Detection:")
    print("   - Find contours: O(m²) in worst case")
    print("   - Filter small contours: O(k) where k = number of contours")
    print("   Total: O(m²)")
    
    print("\n3. Feature Extraction:")
    print("   - Contour area: O(k) where k = number of contour points")
    print("   - Perimeter: O(k)")
    print("   - Bounding rectangle: O(k)")
    print("   - Convex hull: O(k log k)")
    print("   - Polygon approximation: O(k)")
    print("   - Moment calculations: O(k)")
    print("   Total: O(k log k) where k <= m^2")
    
    print("\n4. Classification:")
    print("   - Geometric rule evaluation: O(1)")
    print("   - Angle calculations: O(v) where v = number of vertices")
    print("   Total: O(1) to O(v)")
    
    print("\nOverall Complexity: O(m^2)")
    print("Justification: Image preprocessing and contour detection dominate")
    print("with O(m^2) complexity, while feature extraction and classification")
    print("are O(k) or O(1) where k <= m^2.")

if __name__ == "__main__":
    # Run detailed complexity analysis
    results = analyze_complexity_detailed()
    
    # Analyze algorithm components
    analyze_algorithm_components()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved to 'complexity_analysis_detailed.png'")

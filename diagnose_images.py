"""
Diagnostic script to examine sample images and understand the dataset better.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from shape_classifier import GeometricShapeClassifier

def examine_sample_images():
    """Examine sample images from each class to understand the data."""
    
    classifier = GeometricShapeClassifier()
    data_dir = "data"
    
    # Get sample images from each class
    classes = ['rectangle', 'triangle', 'ellipse', 'other']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, 'user.if2', 'images', class_name)
        if not os.path.exists(class_path):
            continue
        
        # Get first 4 images from this class
        image_files = [f for f in os.listdir(class_path) if f.endswith('.png')][:4]
        
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            
            # Load original image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Preprocess
            binary_img = classifier.preprocess_image(img_path, 100)
            
            # Find contours
            contours = classifier.find_contours(binary_img)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                features = classifier.extract_geometric_features(largest_contour)
                predicted = classifier.classify_shape(features)
                
                # Show original image
                axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[i, j].set_title(f'{class_name}\nPred: {predicted}\n'
                                   f'Vertices: {features["num_vertices"]}\n'
                                   f'Solidity: {features["solidity"]:.2f}\n'
                                   f'Circularity: {features["circularity"]:.2f}')
                axes[i, j].axis('off')
                
                print(f"{class_name} - {img_file}:")
                print(f"  True: {class_name}, Predicted: {predicted}")
                print(f"  Vertices: {features['num_vertices']}")
                print(f"  Solidity: {features['solidity']:.3f}")
                print(f"  Circularity: {features['circularity']:.3f}")
                print(f"  Eccentricity: {features['eccentricity']:.3f}")
                print(f"  Extent: {features['extent']:.3f}")
                print(f"  Aspect Ratio: {features['aspect_ratio']:.3f}")
                if features['angles']:
                    print(f"  Angles: {[f'{a:.1f}°' for a in features['angles'][:5]]}")
                print()
    
    plt.tight_layout()
    plt.savefig('sample_images_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    examine_sample_images()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLine 
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2025/7/8 
'''
import numpy as np
import cv2
import os
import zipfile
import struct

def masks_to_fiji_rois(out_mask, output_dir="roi_sets"):
    """
    Convert segmentation masks to FIJI/ImageJ ROI set files.
    
    Args:
        out_mask: Dict, keys are image paths, values are mask arrays (n_instances, channels, height, width)
        output_dir: Directory to save ROI files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_path, masks in out_mask.items():
        print(f"Processing {image_path}")
        print(f"Masks shape: {masks.shape} (instances, channels, height, width)")
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        all_contours = []
        n_instances = masks.shape[0]
        
        for i in range(n_instances):
            instance_mask = masks[i, 0, :, :]  # Take first channel
            
            print(f"  Processing instance {i+1}/{n_instances}, mask shape: {instance_mask.shape}")
            
            if instance_mask.dtype == bool:
                mask_uint8 = instance_mask.astype(np.uint8) * 255
            elif instance_mask.max() <= 1.0:
                mask_uint8 = (instance_mask * 255).astype(np.uint8)
            else:
                mask_uint8 = instance_mask.astype(np.uint8)
            
            if mask_uint8.max() == 0:
                print(f"    Warning: Empty mask for instance {i+1}, skipping")
                continue
            
            try:
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"    Found {len(contours)} contours for instance {i+1}")
                
                for j, contour in enumerate(contours):
                    if len(contour) >= 3 and cv2.contourArea(contour) > 10:
                        epsilon = 0.002 * cv2.arcLength(contour, True)
                        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(simplified_contour) >= 3:
                            contour_squeezed = simplified_contour.squeeze()
                            if len(contour_squeezed.shape) == 1:
                                contour_squeezed = contour_squeezed.reshape(1, -1)
                            if contour_squeezed.shape[0] >= 3:
                                x_coords = contour_squeezed[:, 0]
                                y_coords = contour_squeezed[:, 1]
                                
                                roi_info = {
                                    'name': f'Instance_{i+1:03d}_{j+1}',
                                    'coordinates': np.column_stack([x_coords, y_coords]),
                                    'area': cv2.contourArea(contour)
                                }
                                all_contours.append(roi_info)
                                print(f"    Added ROI: Instance_{i+1:03d}_{j+1} with {len(x_coords)} points, area: {roi_info['area']:.1f}")
            
            except Exception as e:
                print(f"    Error processing instance {i+1}: {str(e)}")
                continue
        
        if all_contours:
            save_imageJ_rois(all_contours, image_name, output_dir)
            print(f"Saved {len(all_contours)} ROIs for {image_name}")
        else:
            print(f"No valid contours found for {image_path}")

def save_imageJ_rois(contours_info, image_name, output_dir):
    """Save ROIs to ImageJ-format zip file."""
    zip_path = os.path.join(output_dir, f"{image_name}_RoiSet.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for i, roi_info in enumerate(contours_info):
            roi_name = f"{i+1:04d}-{roi_info['name']}.roi"
            coordinates = roi_info['coordinates']
            roi_data = create_imageJ_roi(coordinates)
            if roi_data:
                zip_file.writestr(roi_name, roi_data)
    
    print(f"ROI set saved: {zip_path}")

def create_imageJ_roi(coordinates):
    """Create ImageJ-compatible polygon ROI binary data."""
    n_coords = len(coordinates)
    if n_coords < 3:
        return None
    
    x_coords = coordinates[:, 0].astype(np.int16)
    y_coords = coordinates[:, 1].astype(np.int16)
    left = int(np.min(x_coords))
    top = int(np.min(y_coords))
    right = int(np.max(x_coords))
    bottom = int(np.max(y_coords))
    
    roi_data = b'Iout'  # Magic number
    roi_data += struct.pack('>h', 227)        # Version
    roi_data += struct.pack('>b', 2)          # Type: polygon
    roi_data += struct.pack('>b', 0)          # Flags
    roi_data += struct.pack('>h', top)        # Top
    roi_data += struct.pack('>h', left)       # Left
    roi_data += struct.pack('>h', bottom)     # Bottom
    roi_data += struct.pack('>h', right)      # Right
    roi_data += struct.pack('>h', n_coords)   # N coordinates
    roi_data += b'\x00' * (64 - len(roi_data)) # Pad header to 64 bytes
    
    for i in range(n_coords):
        x_rel = max(0, min(65535, int(x_coords[i]) - left))
        y_rel = max(0, min(65535, int(y_coords[i]) - top))
        roi_data += struct.pack('>h', x_rel)
        roi_data += struct.pack('>h', y_rel)
    
    return roi_data

def create_summary_report(out_mask, output_dir="roi_sets"):
    """Create analysis summary report."""
    report_path = os.path.join(output_dir, "segmentation_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("Segmentation Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_instances = 0
        for image_path, masks in out_mask.items():
            image_name = os.path.basename(image_path)
            n_instances = masks.shape[0]
            total_instances += n_instances
            
            f.write(f"Image: {image_name}\n")
            f.write(f"  Number of instances: {n_instances}\n")
            f.write(f"  Mask dimensions: {masks.shape}\n\n")
        
        f.write(f"Total images processed: {len(out_mask)}\n")
        f.write(f"Total instances detected: {total_instances}\n")
        f.write(f"Average instances per image: {total_instances/len(out_mask):.1f}\n")
    
    print(f"Summary report saved: {report_path}")
    
    """
11. Morphological Analysis of Segmentation Masks

    - For each instance in each image:
        - Calculate basic shape, geometry, and texture statistics for every mask instance
        - Includes area, perimeter, bounding box, circularity, compactness, convexity, solidity, ellipse fit, skeleton length, branch and end points, intensity stats, etc.
    - Aggregate all results into a DataFrame and save:
        - Detailed instance-level data (CSV)
        - Per-image summary statistics (CSV)
        - Overall statistics (CSV)
        - Human-readable analysis report (TXT)
        - Visualization plots for major morphological features (PNG)
    - Main functions:
        - `calculate_morphological_statistics`: Extract and aggregate all stats for out_mask
        - `calculate_basic_statistics`, `calculate_shape_statistics`, `calculate_geometry_statistics`, `calculate_texture_statistics`: Feature calculation helpers
        - `save_morphology_results`, `generate_analysis_report`: Save and report results
        - `create_morphology_visualizations`: Visualization of key metrics

    **Application**:  
    This pipeline enables comprehensive post-segmentation morphological quantification, QC, and summary reporting for biological or industrial image analyses.
"""

import numpy as np
import cv2
import pandas as pd
import os
from sklearn.metrics import pairwise_distances
from scipy import ndimage
import math

def calculate_morphological_statistics(out_mask, output_dir="morphology_analysis"):
    """
    Calculate detailed morphological statistics for each instance in the segmentation masks.

    Args:
        out_mask (dict): Dictionary with image path as key and mask array (n_instances, channels, height, width) as value.
        output_dir (str): Directory to save the results.

    Returns:
        pd.DataFrame: DataFrame containing all morphological parameters for all instances.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_statistics = []
    
    for image_path, masks in out_mask.items():
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Analyzing {image_name}...")
        
        n_instances = masks.shape[0]
        
        for i in range(n_instances):
            # Extract single instance mask
            instance_mask = masks[i, 0, :, :].astype(np.uint8)
            
            if not instance_mask.any():
                continue
            
            # Convert to binary image
            binary_mask = (instance_mask > 0).astype(np.uint8) * 255
            
            try:
                # Basic statistics
                stats = calculate_basic_statistics(binary_mask, image_name, i+1)
                
                # Shape statistics
                shape_stats = calculate_shape_statistics(binary_mask)
                stats.update(shape_stats)
                
                # Geometry statistics
                geometry_stats = calculate_geometry_statistics(binary_mask)
                stats.update(geometry_stats)
                
                # Texture and intensity statistics
                texture_stats = calculate_texture_statistics(instance_mask)
                stats.update(texture_stats)
                
                all_statistics.append(stats)
                
            except Exception as e:
                print(f"  Error analyzing instance {i+1}: {str(e)}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(all_statistics)
    
    # Save results
    save_morphology_results(df, output_dir)
    
    return df

def calculate_basic_statistics(binary_mask, image_name, instance_id):
    """
    Compute basic statistics for a single instance.

    Args:
        binary_mask (np.ndarray): Binary mask of the instance.
        image_name (str): Name of the image.
        instance_id (int): Instance index.

    Returns:
        dict: Basic statistics of the instance.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    x, y, w, h = cv2.boundingRect(main_contour)
    bounding_area = w * h
    (center_x, center_y), radius = cv2.minEnclosingCircle(main_contour)
    enclosing_circle_area = np.pi * radius * radius
    rect = cv2.minAreaRect(main_contour)
    box_area = rect[1][0] * rect[1][1]
    
    stats = {
        'Image': image_name,
        'Instance_ID': instance_id,
        'Area_pixels': area,
        'Perimeter_pixels': perimeter,
        'Width_pixels': w,
        'Height_pixels': h,
        'Bounding_Box_Area': bounding_area,
        'Min_Enclosing_Circle_Radius': radius,
        'Min_Enclosing_Circle_Area': enclosing_circle_area,
        'Min_Area_Rect_Area': box_area,
        'Centroid_X': center_x,
        'Centroid_Y': center_y
    }
    
    return stats

def calculate_shape_statistics(binary_mask):
    """
    Compute shape-related statistics for a single instance.

    Args:
        binary_mask (np.ndarray): Binary mask of the instance.

    Returns:
        dict: Shape statistics.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    main_contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    # Circularity = 4π × Area / Perimeter²
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    # Compactness = Perimeter² / (4π × Area)
    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = w / h if h > 0 else 0
    rectangularity = area / (w * h) if (w * h) > 0 else 0
    (_, _), radius = cv2.minEnclosingCircle(main_contour)
    circle_area = np.pi * radius * radius
    circle_ratio = area / circle_area if circle_area > 0 else 0
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = area / hull_area if hull_area > 0 else 0
    solidity = convexity
    shape_complexity = perimeter / hull_perimeter if hull_perimeter > 0 else 1
    if len(main_contour) >= 5:
        ellipse = cv2.fitEllipse(main_contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
        ellipse_area = np.pi * major_axis/2 * minor_axis/2
        ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
    else:
        major_axis = minor_axis = eccentricity = ellipse_ratio = 0
    
    return {
        'Circularity': circularity,
        'Compactness': compactness,
        'Aspect_Ratio': aspect_ratio,
        'Rectangularity': rectangularity,
        'Circle_Ratio': circle_ratio,
        'Convexity': convexity,
        'Solidity': solidity,
        'Shape_Complexity': shape_complexity,
        'Hull_Area': hull_area,
        'Hull_Perimeter': hull_perimeter,
        'Major_Axis': major_axis,
        'Minor_Axis': minor_axis,
        'Eccentricity': eccentricity,
        'Ellipse_Ratio': ellipse_ratio
    }

def calculate_geometry_statistics(binary_mask):
    """
    Compute geometric statistics for a single instance.

    Args:
        binary_mask (np.ndarray): Binary mask of the instance.

    Returns:
        dict: Geometry statistics.
    """
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_distance = np.max(dist_transform)
    mean_distance = np.mean(dist_transform[dist_transform > 0])
    std_distance = np.std(dist_transform[dist_transform > 0])
    skeleton = skeletonize(binary_mask)
    skeleton_length = np.sum(skeleton)
    branch_points = count_branch_points(skeleton)
    end_points = count_end_points(skeleton)
    
    return {
        'Max_Inscribed_Circle_Radius': max_distance,
        'Mean_Distance_to_Boundary': mean_distance,
        'Std_Distance_to_Boundary': std_distance,
        'Skeleton_Length': skeleton_length,
        'Branch_Points': branch_points,
        'End_Points': end_points
    }

def calculate_texture_statistics(intensity_mask):
    """
    Compute texture and intensity statistics for a single instance.

    Args:
        intensity_mask (np.ndarray): Original (non-binary) mask.

    Returns:
        dict: Texture and intensity statistics.
    """
    if intensity_mask.max() == 0:
        return {
            'Mean_Intensity': 0,
            'Std_Intensity': 0,
            'Min_Intensity': 0,
            'Max_Intensity': 0,
            'Intensity_Range': 0
        }
    
    valid_pixels = intensity_mask[intensity_mask > 0]
    mean_intensity = np.mean(valid_pixels)
    std_intensity = np.std(valid_pixels)
    min_intensity = np.min(valid_pixels)
    max_intensity = np.max(valid_pixels)
    intensity_range = max_intensity - min_intensity
    
    return {
        'Mean_Intensity': mean_intensity,
        'Std_Intensity': std_intensity,
        'Min_Intensity': min_intensity,
        'Max_Intensity': max_intensity,
        'Intensity_Range': intensity_range
    }

def skeletonize(binary_mask):
    """
    Compute the skeleton of a binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Skeletonized mask (uint8).
    """
    from skimage.morphology import skeletonize as sk_skeletonize
    binary_image = binary_mask > 0
    skeleton = sk_skeletonize(binary_image)
    return skeleton.astype(np.uint8)

def count_branch_points(skeleton):
    """
    Count the number of branch points in a skeleton.

    Args:
        skeleton (np.ndarray): Skeletonized mask.

    Returns:
        int: Number of branch points.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    branch_points = np.sum((skeleton == 1) & (filtered >= 13))
    return branch_points

def count_end_points(skeleton):
    """
    Count the number of end points in a skeleton.

    Args:
        skeleton (np.ndarray): Skeletonized mask.

    Returns:
        int: Number of end points.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    end_points = np.sum((skeleton == 1) & (filtered == 11))
    return end_points

def save_morphology_results(df, output_dir):
    """
    Save detailed and summary morphology results to CSV and generate an analysis report.

    Args:
        df (pd.DataFrame): Morphology results DataFrame.
        output_dir (str): Output directory.
    """
    df.to_csv(os.path.join(output_dir, 'detailed_morphology.csv'), index=False)
    summary_stats = df.groupby('Image').agg({
        'Instance_ID': 'count',
        'Area_pixels': ['mean', 'std', 'min', 'max'],
        'Perimeter_pixels': ['mean', 'std'],
        'Circularity': ['mean', 'std'],
        'Solidity': ['mean', 'std'],
        'Aspect_Ratio': ['mean', 'std'],
        'Compactness': ['mean', 'std']
    }).round(3)
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats.to_csv(os.path.join(output_dir, 'summary_by_image.csv'))
    overall_stats = df.describe().round(3)
    overall_stats.to_csv(os.path.join(output_dir, 'overall_statistics.csv'))
    generate_analysis_report(df, output_dir)
    print(f"Morphology analysis completed!")
    print(f"Results saved in: {output_dir}")
    print(f"- detailed_morphology.csv: Detailed per-instance data")
    print(f"- summary_by_image.csv: Summary by image")
    print(f"- overall_statistics.csv: Overall descriptive statistics")
    print(f"- morphology_report.txt: Analysis report")

def generate_analysis_report(df, output_dir):
    """
    Generate a plain-text report summarizing the main morphology results.

    Args:
        df (pd.DataFrame): Morphology results DataFrame.
        output_dir (str): Output directory.
    """
    report_path = os.path.join(output_dir, 'morphology_report.txt')
    with open(report_path, 'w') as f:
        f.write("Morphology Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of images analyzed: {df['Image'].nunique()}\n")
        f.write(f"Total number of instances detected: {len(df)}\n")
        f.write(f"Average instances per image: {len(df)/df['Image'].nunique():.1f}\n\n")
        f.write("Size Statistics:\n")
        f.write(f"  Mean area: {df['Area_pixels'].mean():.1f} ± {df['Area_pixels'].std():.1f} pixels\n")
        f.write(f"  Area range: {df['Area_pixels'].min():.1f} - {df['Area_pixels'].max():.1f} pixels\n")
        f.write(f"  Mean perimeter: {df['Perimeter_pixels'].mean():.1f} ± {df['Perimeter_pixels'].std():.1f} pixels\n\n")
        f.write("Shape Features:\n")
        f.write(f"  Mean circularity: {df['Circularity'].mean():.3f} ± {df['Circularity'].std():.3f}\n")
        f.write(f"  Mean solidity: {df['Solidity'].mean():.3f} ± {df['Solidity'].std():.3f}\n")
        f.write(f"  Mean aspect ratio: {df['Aspect_Ratio'].mean():.3f} ± {df['Aspect_Ratio'].std():.3f}\n")
        f.write(f"  Mean compactness: {df['Compactness'].mean():.3f} ± {df['Compactness'].std():.3f}\n\n")
        f.write("Shape Classification:\n")
        circular = len(df[df['Circularity'] > 0.8])
        elongated = len(df[df['Aspect_Ratio'] > 2.0])
        irregular = len(df[df['Solidity'] < 0.8])
        f.write(f"  Circular objects (circularity > 0.8): {circular} ({circular/len(df)*100:.1f}%)\n")
        f.write(f"  Elongated objects (aspect ratio > 2.0): {elongated} ({elongated/len(df)*100:.1f}%)\n")
        f.write(f"  Irregular objects (solidity < 0.8): {irregular} ({irregular/len(df)*100:.1f}%)\n")

def create_morphology_visualizations(df, output_dir):
    """
    Create visualization charts for key morphological parameters.

    Args:
        df (pd.DataFrame): Morphology results DataFrame.
        output_dir (str): Output directory.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Morphological Parameter Distributions', fontsize=16)
    
    axes[0,0].hist(df['Area_pixels'], bins=30, alpha=0.7, color='skyblue')
    axes[0,0].set_xlabel('Area (pixels)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Area Distribution')
    
    axes[0,1].hist(df['Circularity'], bins=30, alpha=0.7, color='lightgreen')
    axes[0,1].set_xlabel('Circularity')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Circularity Distribution')
    
    axes[0,2].hist(df['Solidity'], bins=30, alpha=0.7, color='orange')
    axes[0,2].set_xlabel('Solidity')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Solidity Distribution')
    
    axes[1,0].hist(df['Aspect_Ratio'], bins=30, alpha=0.7, color='pink')
    axes[1,0].set_xlabel('Aspect Ratio')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Aspect Ratio Distribution')
    
    axes[1,1].scatter(df['Area_pixels'], df['Circularity'], alpha=0.6, color='purple')
    axes[1,1].set_xlabel('Area (pixels)')
    axes[1,1].set_ylabel('Circularity')
    axes[1,1].set_title('Area vs. Circularity')
    
    axes[1,2].scatter(df['Solidity'], df['Circularity'], alpha=0.6, color='red')
    axes[1,2].set_xlabel('Solidity')
    axes[1,2].set_ylabel('Circularity')
    axes[1,2].set_title('Solidity vs. Circularity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {os.path.join(output_dir, 'morphology_distributions.png')}")

# # Usage example
# print("Starting morphological analysis...")
# df_morphology = calculate_morphological_statistics(out_mask)

# print("\nCreating visualization charts...")
# create_morphology_visualizations(df_morphology, "morphology_analysis")

# print("\nAnalysis completed!")
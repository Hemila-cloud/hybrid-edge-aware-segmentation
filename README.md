# Hybrid Edge-Aware Framework for Rooftop Photovoltaic Potential Assessment Using Aerial Imagery

## 1. Project Overview

Rooftop photovoltaic (PV) systems are an important approach to utilizing renewable solar energy in urban environments. However, identifying suitable rooftop regions for solar panel installation requires the analysis of several factors, including usable roof area, roof orientation, shadows, and rooftop obstacles.

This project proposes a **Hybrid Edge-Aware Framework for Rooftop Photovoltaic Potential Assessment Using Aerial Imagery**. It combines classical image processing techniques with DeepLabV3+ semantic segmentation to identify rooftop structures, extract geometric and environmental features, and estimate rooftop suitability for photovoltaic installation.

The framework integrates image enhancement, edge detection, semantic segmentation, geometric analysis, and rule-based suitability assessment to provide preliminary recommendations for rooftop solar planning.

## 2. Problem Statement

Manual rooftop assessment is time-consuming and may not provide consistent estimates of usable roof area, orientation, shadows, and obstacles. Conventional image-processing methods may also struggle with complex rooftop structures and variations in illumination.

Therefore, an integrated image-based framework is required to combine classical image-processing techniques with deep learning-based segmentation and rooftop feature analysis for more effective and systematic PV suitability assessment.

## 3. Project Objectives

- To preprocess and enhance aerial rooftop images for improved feature extraction.
- To identify rooftop boundaries and structural features using classical image-processing techniques.
- To perform semantic segmentation of rooftop regions and superstructures using DeepLabV3+.
- To estimate rooftop characteristics such as usable area, dominant orientation, shadow coverage, and obstacle coverage.
- To integrate image-processing and deep-learning outputs for rooftop analysis.
- To calculate a rule-based PV suitability score and classify rooftops into suitability categories.
- To generate preliminary recommendations for prioritizing rooftops for photovoltaic installation.

## 4. Dataset

The project uses the **Roof Information Dataset (RID)**, which contains aerial rooftop images and corresponding semantic segmentation annotations.

**Dataset:** Roof Information Dataset for Identifying Roof Segments and Roof Superstructures in Aerial Images (RID)


**Dataset paper:** [https://doi.org/10.14459/2022mp1655470](https://doi.org/10.14459/2022mp1655470)

Dataset characteristics:

- Total images: 1,880
- Image dimensions: 512 × 512 pixels
- Image type: Aerial rooftop imagery
- Annotations: Roof segment masks and rooftop superstructure masks
- Semantic classes: Multiple rooftop structural and environmental categories

The dataset annotations are used for training and evaluating the semantic segmentation model and for extracting rooftop-related information.


## 5. Proposed Methodology

The proposed framework consists of the following stages:

### 5.1 Image Preprocessing and Enhancement

Aerial images are preprocessed to reduce noise and improve image quality.

The following techniques are applied:

- **Gaussian Filtering:** Reduces image noise through Gaussian smoothing.
- **Median Filtering:** Removes noise while preserving important structural boundaries.
- **CLAHE:** Enhances local contrast to improve the visibility of rooftop features.
- **Intensity Normalization:** Scales image intensity values to a normalized range.

These operations prepare the images for subsequent edge detection, feature extraction, and semantic segmentation.

### 5.2 Classical Image Processing and Edge Extraction

Classical image-processing techniques are used to identify rooftop boundaries and structural information.

The implemented operations include:

- **Canny Edge Detection:** Identifies significant edges and rooftop boundaries.
- **Adaptive Thresholding:** Separates foreground rooftop regions from surrounding areas based on local intensity.
- **Morphological Operations:** Refine binary regions and reduce unwanted noise.
- **Contour Detection:** Extracts structural boundaries and geometric information.

These techniques provide edge-based and region-based information that contributes to rooftop feature analysis.

### 5.3 Deep Learning-Based Semantic Segmentation

DeepLabV3+ is used to perform semantic segmentation of rooftop images.

The model uses a ResNet-50 encoder to extract image features and identify rooftop regions and superstructures at the pixel level.

The segmentation output supports the identification and analysis of different rooftop components, including PV modules and other structural elements.

### 5.4 Rooftop Feature Extraction and Geometric Analysis

The segmented outputs and image-processing results are used to estimate important rooftop characteristics.

The extracted features include:

- **Roof Area:** Estimates the rooftop region available for potential PV installation.
- **Roof Coverage:** Determines the proportion of the image occupied by identified rooftop regions.
- **Roof Orientation:** Estimates the dominant rooftop direction using geometric analysis.
- **Shadow Coverage:** Estimates the proportion of rooftop regions affected by shadows.
- **Obstacle Coverage:** Measures the rooftop area occupied by identified obstacles and superstructures.
- **PV Module Coverage:** Estimates the area occupied by identified PV module regions.

These features provide information about rooftop geometry and environmental conditions that influence preliminary PV suitability.

### 5.5 Hybrid Integration

The proposed framework combines classical edge-aware image processing with DeepLabV3+ semantic segmentation.

Classical image processing provides edge and structural information, while DeepLabV3+ identifies rooftop regions and semantic components. Their outputs are used alongside geometric and environmental feature extraction to support a more comprehensive rooftop assessment.

The integrated framework is evaluated against the individual classical image-processing and DeepLabV3+ approaches.

### 5.6 PV Suitability Assessment

A rule-based scoring mechanism is used to estimate the suitability of each rooftop for photovoltaic installation.

The suitability score is calculated using the following weighted formula:

**S = 0.40A + 0.30O + 0.20H + 0.10B**

Where:

- **A:** Normalized usable rooftop area.
- **O:** Orientation suitability.
- **H:** Shadow-related suitability.
- **B:** Obstacle-related suitability.

The weights represent the relative contribution of each feature to the overall suitability assessment.

The resulting scores are classified into five categories:

| Suitability Score | Category |
|---|---|
| 80 and above | Excellent |
| 65 to less than 80 | Good |
| 50 to less than 65 | Moderate |
| 35 to less than 50 | Poor |
| Below 35 | Very Poor |

The categories help prioritize rooftops based on their estimated suitability.

## 6. System Workflow

The overall workflow of the proposed framework is:

1. Input aerial rooftop images and corresponding annotations.
2. Apply image preprocessing and enhancement.
3. Perform edge detection, thresholding, morphological processing, and contour extraction.
4. Perform semantic segmentation using DeepLabV3+.
5. Integrate image-processing and segmentation information.
6. Extract rooftop geometric and environmental features.
7. Calculate rooftop PV suitability scores.
8. Classify rooftops and generate preliminary recommendations.

## 7. Experimental Results

The framework was evaluated using the RID dataset.

### 7.1 Segmentation Performance

The segmentation performance of the individual methods and the integrated framework is summarized below.

| Method | IoU | Dice Score | Precision | Recall |
|---|---:|---:|---:|---:|
| Classical Image Processing | 0.2976 | 0.4480 | 0.4158 | 0.5482 |
| DeepLabV3+ | 0.9184 | 0.9568 | 0.9315 | 0.9850 |
| Integrated Framework | 0.9300 | 0.9600 | 0.9400 | 0.9912 |

The results indicate that DeepLabV3+ achieves substantially higher segmentation performance than classical image processing alone. The integrated framework reports further improvements across the evaluated metrics.

### 7.2 Rooftop Suitability Assessment

A total of **607 rooftop records** were assessed using the extracted geometric and environmental features.

The suitability classification results are:

| Category | Number of Rooftops | Percentage |
|---|---:|---:|
| Excellent | 79 | 13.01% |
| Good | 225 | 37.07% |
| Moderate | 200 | 32.95% |
| Poor | 78 | 12.85% |
| Very Poor | 25 | 4.12% |
| **Total** | **607** | **100%** |

Approximately **50.08% of the evaluated rooftops** were classified as Good or Excellent, indicating their higher estimated suitability for preliminary PV planning.

## 8. Technologies and Tools

The project uses the following technologies and techniques:

- **Programming Language:** Python
- **Image Processing:** OpenCV
- **Numerical Processing:** NumPy
- **Data Analysis:** Pandas
- **Deep Learning:** PyTorch
- **Semantic Segmentation:** DeepLabV3+
- **Feature Extraction:** Image processing and geometric analysis
- **Development Environment:** Visual Studio Code 





## 12. Conclusion

This project presents a hybrid approach for preliminary rooftop photovoltaic suitability assessment using aerial imagery. By combining classical image processing, DeepLabV3+ semantic segmentation, geometric feature extraction, and rule-based suitability scoring, the framework provides an integrated approach to rooftop analysis.
The experimental results demonstrate the effectiveness of deep learning-based segmentation compared with classical image processing alone, while the integrated framework provides segmentation results and rooftop suitability classifications. The proposed system can support preliminary rooftop prioritization for solar planning, with further validation required before real-world installation decisions.


**Outputs:**
https://drive.google.com/drive/folders/1Z1RGCyVJB2E6ABkAuXw4lkh4gdmINBVY?usp=sharing



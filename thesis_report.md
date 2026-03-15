# Object Detection with Deep Learning: Project 3
**Authors:** Yehonatan Gurevich (206607962), Yehuda Frist (316484476)

## Infrastructure Links
- **Input Video:** [Big Cats of the World Part II](https://www.youtube.com/watch?v=YOUR_LINK_HERE)
- **Output Video Inference:** [Part 2 Output Video](https://www.youtube.com/watch?v=YOUR_LINK_HERE)

## Architecture Selection (ID Summation)
The project required selecting a convolutional backbone based on the summation of the group members' identification numbers.
- ID 1 (206607962) Sum: `2+0+6+6+0+7+9+6+2 = 38`
- ID 2 (316484476) Sum: `3+1+6+4+8+4+4+7+6 = 43`
- Combined Sum: `38 + 43 = 81`
- Final Digit Sum: `8 + 1 = 9`
- **Result:** Value 9 dictates the selection of the **MobileNetV3-Small** architecture.

---

# Part 1: Classification Backbone Report

## 1. Architecture Overview
MobileNetV3-Small serves as a lightweight convolutional neural network optimized for low-latency inference on hardware-constrained environments. The architecture was derived utilizing automated Neural Architecture Search (NAS) combined with NetAdapt, distinguishing it from manually designed models.
- **Model Size**: Approximately 2.5 million parameters.
- **Objective**: Minimization of structural latency and power consumption while preserving representational capacity.
- **Structure**: The framework employs a sequence of inverted residual blocks interspersed with variable kernel dimensions (3x3 and 5x5) and Squeeze-and-Excitation (SE) modules, culminating in an efficient global average pooling mechanism and classification head.

## 2. Key Architectural Innovations
### 2.1. Network Architecture Search (NAS)
Rather than relying on empirical human design, MobileNetV3 incorporates a platform-aware ML optimization approach. The search reward function was explicitly parameterized to incorporate inference latency on mobile CPU topologies, yielding a heterogeneous structure wherein layer dimensionalities and kernel sizes vary irregularly to maximize computational throughput.

### 2.2. Squeeze-and-Excitation (SE) Modules
Integrated directly within the bottleneck blocks, SE modules facilitate dynamic channel attention.
- **Function**: These modules compute channel-wise weights by aggregating global spatial information into a descriptor, allowing the network to recalibrate feature map focus dynamically.
- **Optimization**: To strictly bound computational overhead, the SE blocks in MobileNetV3 utilize a reduced dimensionality projection (25% of the input channel size), successfully boosting classification accuracy at minimal FLOP cost.

### 2.3. Hard-Swish Activation Function
While the standard Swish activation (x * sigma(x)) demonstrably improves accuracy in deep networks, the calculation of the Sigmoid function introduces a severe latency penalty on edge devices.
- **Solution**: MobileNetV3 substitutes this with the "Hard-Swish" approximation: h-swish(x) = x * ReLU6(x+3) / 6.
- **Benefit**: This formulation successfully mimics Swish utilizing solely standard arithmetic and ReLU6, facilitating rapid computation and hardware-level quantization viability.

### 2.4. Integration of 5x5 Depthwise Convolutions
Whereas preceding iterations of the MobileNet family relied almost exclusively on 3x3 kernels, MobileNetV3 selectively re-introduces 5x5 convolutions within deeper layers. NAS analysis indicated that expanded receptive fields are critical for extracting high-level semantic features in later stages. Given the computationally inexpensive nature of depthwise operations, the 5x5 kernels induced significant accuracy improvements at a negligible latency cost.

### 2.5. Efficient Classification Head
The terminal layers were structurally revised to mitigate latency. The computationally expensive 1x1 expansion layer was repositioned post-global average pooling. This architectural shift ensures the expansion operates on a 1x1 feature map rather than a 7x7 spatial resolution, reducing computational operations significantly prior to the classifier without performance degradation.

## 3. Inference Analysis and Limitations
### 3.1. Empirical Inference Performance
To evaluate the zero-shot capacity of the pre-trained ImageNet weights, inference was executed across a subset of highly diverse image classes. Observation of the resulting probability distributions reveals absolute canonical certainty for distinct morphological shapes:
- **High Confidence Classifications:** The network correctly classified an image of a "cheeseburger" with **99.9%** confidence, and a "panda" with **100.0%** confidence. Such stark probability peaks indicate the architecture's exceptional capacity to extract globally unambiguous feature descriptors.

### 3.2. Structural Limitations and Failure Cases
- **Fine-Grained Misclassification:** The constrained channel capacity (maximum 576 output features) inherently restricts fine-grained differentiation within similar biological subclasses. For example, during testing, an image of a **goat** resulted in an erroneous top prediction of a "ram" (with only **21%** confidence), and an image of a **horse** was misclassified as a "water buffalo" (with **57%** confidence). 
- **Analysis:** This uncertainty underscores the architectural trade-off: MobileNetV3 sacrifices deep-channel redundancy utilized by broader models (e.g., ResNet) in favor of latency optimization, resulting in vulnerability when parsing nuanced inter-class textures.

[IMAGE_56]
Figure 1: Inference classification results demonstrating high-confidence successes and low-confidence morphological cross-classifications.

**Conclusion**: MobileNetV3-Small demonstrates highly effective machine-optimized architectural design. By trading structural redundancy for deployment efficiency, it provides a stable and highly performant lightweight backbone suitable for subsequent object detection adaptations.

## References
[1] Howard, A., et al. (2019). "Searching for MobileNetV3". arXiv:1905.02244.
[2] Hu, J., et al. (2018). "Squeeze-and-Excitation Networks".

# Part 2: Single Object Detection Engineering Report
## 1. Core Architecture and Problem Formulation
For the single object (tiger) detection task, an edge-capable detection pipeline was developed utilizing Transfer Learning embedded atop a lightweight backbone architecture.

### 1.1 Architectural Configuration
- **Backbone**: MobileNetV3-Small (ImageNet pre-trained), selected to ensure high processing frame-rates while preserving hierarchical spatial resolution.
- **Dimensionality Reduction**: To avoid the spatial data destruction inherent to 1x1 global average pooling, `AdaptiveAvgPool2d(4)` was applied to the backbone's 576-channel termination. This yielded a dense 9216-dimensional feature vector, effectively preserving spatial quadrant geometries essential for accurate bounding box localization.
- **Regression Head**: A dense Multilayer Perceptron (512 -> 256 -> 128 -> 4) was constructed. It incorporated Xavier weight initialization, Batch Normalization, and Dropout layers (0.3/0.2) to systematically restrain overfitting. The terminal layer adopted a Sigmoid activation, mathematically constraining the (cx, cy, w, h) coordinate predictions strictly within a normalized [0, 1] spatial domain.
- **Optimization Strategy**: The model utilized the `AdamW` optimizer (weight decay 1e-4) modulated by a `ReduceLROnPlateau` learning rate scheduler.

### 1.2 Objective Function
Application of standard regression losses (L1/L2) introduces bias, as coordinate error fails to explicitly penalize scale variance. Consequently, the network was optimized using Efficient Intersection over Union (EIoU) Loss. EIoU imposes penalties on three distinct geometric axes:
1. Bounding box planar overlap (IoU).
2. Euclidean distance between center-points.
3. Variance in bounding box aspect ratio and specific width-height dimensionalities.

## 2. Data Strategy and Curation
### 2.1 Multi-Object Dataset Preprocessing
The initial dataset distribution included images populated with multiple object annotations. Attempting to fit a single-object regression head against multi-object spatial targets induces destructive gradient phenomena (the network mathematically regresses toward the geometric centroid between disparate objects).
- **Resolution**: A semantic parsing script (`filter_single_object.py`) was implemented to rigidly isolate and retain solely images satisfying `len(annotations) == 1`.

### 2.2 Validation Split Re-Engineering
The original dataset partition defined a validation subset of only 374 images. Evaluating convergence metrics across such a miniscule sample generates significant statistical noise, frequently triggering premature early stopping.
- **Resolution**: As out-of-distribution evaluation relies upon external zero-shot video inference, the independent static test set represented redundant isolation. Thus, test annotations and images were computationally integrated into the validation split, resolving ID collisions and duplicate data pointers.
- **Result**: The restructuring provided a statistically significant validation set of 752 images, yielding a robust 7.7:1 Train-to-Validation sample ratio.

### 2.3 Data Augmentation
To mitigate structural overfitting, horizontal image flipping (P=0.5) was integrated. Tigers exhibit horizontal symmetry; therefore, mirrored orientations increase variance without corrupting semantic meaning. Bounding box coordinates were mathematically inverted along the x-axis to synchronize with augmented frames. Furthermore, to address severe illumination variance expected in wild deployments, programmatic Color Jittering (modulating brightness, contrast, saturation, and hue) was applied.

## 3. Training Dynamics and Loss Analysis
The network was trained for 140 epochs employing a staggered "Silent Initialization" paradigm: the pre-trained backbone parameters were frozen during the first 5 epochs (LR=0.0) while the randomly initialized head absorbed massive initial gradient magnitude, effectively preventing catastrophic forgetting within the backbone features.

[IMAGE_115]
Figure 2: TensorBoard Loss Curves demonstrating optimization through epoch 140.

### 3.1 Loss Curve Analytics
Observation of the telemetric data reveals key training behaviors:
- **Generalization Capacity**: The `Loss/val` trajectory mirrors the `Loss/train` loss vector with strong correlation through 140 epochs, ultimately yielding an optimal validation Intersection over Union (**IoU**) of **90.0%**. The absence of upward divergence confirms that the applied dropout and weight-decay regularization strategies successfully mitigated overfitting.
- **Localization versus Scaling Error**: The `Loss/val_coord` (representing absolute center-point variance) decays to ~0.0004 virtually immediately, indicating the MobileNet backbone extracts focal localization with trivial difficulty. Conversely, `Loss/val_iou` reduces logarithmically over the full training cycle. This split proves the primary optimization challenge lies not in localizing the object center, but in precise scalar geometry estimation (object width and height resolution).

## 4. Zero-Shot Video Tracking and Inference
### 4.1 Static Evaluation
The architecture was evaluated across both the training distributions and validation set subsets to empirically confirm bounding box tightness and prediction confidence.

[IMAGE_126]
Figure 3: Static image inference demonstrating accurate bounding box geometry.

### 4.2 Tracking Pipeline Engineering
Applying a static-image object detection model natively across continuous video frames generates high-frequency bounding box temporal jitter. To stabilize inference on the external dataset sequence, a "Tracking-by-Detection" pipeline was implemented:
- **Exponential Moving Average (EMA) Smoothing**: A mathematical scalar momentum (alpha = 0.35) was embedded into the coordinate estimation loop:
Box_current = (alpha * Box_predicted) + ((1 - alpha) * Box_history)
This operation geometrically filters high-frequency frame-to-frame coordinate noise, yielding visually stabilized bounding tracking.
- **Physics Plausibility Gating**: Hard kinematic constraints were appended to reject anomalous model hallucinations induced by motion blur. If an absolute prediction exceeded standard physical frame-to-frame translation limits (Center delta > 50%) or impossible morphological shifts (Shape delta > 80%), the raw prediction was rejected. 

### 4.3 Video Inference Limitations
During live sequence rendering over the continuous external video, the network sustained highly contiguous tracking, governed successfully by the EMA momentum logic. However, an empirical vulnerability materialized: when the target traversed terrain featuring a background statistically contiguous in local color spectra and structural shape (e.g., highly textured yellow/brown jungle scrub), performance degraded marginally. The semantic boundary distinguishing the foreground mass from the background eroded rapidly across the limited 576-channel topology, causing temporary dimensional fluctuation before restabilizing in clearer contrast frames.

[IMAGE_136]
Figure 4: Zero-shot video inference demonstrating temporal EMA tracking.

## 5. Conclusion
The implementation of the MobileNetV3-Small framework, coupled with EIoU loss optimization and rigid data filtering operations, yielded a highly performant classification-to-detection transition achieving a 90% peak validation IoU. Empirical loss analytics confirmed that while spatial localization is computed rapidly by the backbone, bounding box scalar geometry governs the primary learning phase. Finally, encapsulating the stateless neural network within an EMA Plausibility tracking filter successfully resolved native inter-frame jitter, creating a robust, continuous video tracking system constrained only by foreground-background structural isomorphism scenarios.

# Glossary of Technical Terms

This glossary explains technical terms used in the RF-DETR Inference project README.

---

**RF-DETR**: A transformer-based object detection model developed by Roboflow, designed for efficient and accurate detection of objects in images. DETR stands for "DEtection TRansformer".

**ONNX Runtime**: An open-source runtime for executing machine learning models in the ONNX (Open Neural Network Exchange) format, enabling interoperability between different frameworks.

**OpenCV**: An open-source computer vision and image processing library widely used for tasks such as image manipulation, object detection, and video analysis.

**C++20**: The 2020 revision of the C++ programming language standard, introducing new features and improvements for modern C++ development.

**CMake**: A cross-platform build system generator that manages the build process for software projects using simple configuration files.

**Google Test**: A popular C++ testing framework for writing and running unit tests.

**Ninja**: A small, fast build system focused on speed, often used as a backend for CMake to accelerate compilation.

**COCO Labels**: A set of object class names defined by the COCO (Common Objects in Context) dataset, commonly used for training and evaluating object detection models.

**Bounding Box**: A rectangle that defines the location of an object detected in an image, typically specified by its coordinates.

**Model File (ONNX)**: The file containing the trained weights and architecture of a machine learning model, saved in the ONNX format for compatibility.

**Sigmoid Function**: A mathematical function that maps any real-valued number into the range (0, 1). In machine learning, it is often used to convert model outputs (logits) into probabilities. The formula is $\sigma(x) = \frac{1}{1 + e^{-x}}$. Used for binary classification, multi-label classification, and in object detection models to convert class logits to independent probabilities.

**Segmentation Masks**: In computer vision, a segmentation mask is an image where each pixel is labeled to indicate the object or region it belongs to. Used for tasks like semantic or instance segmentation, where the goal is to delineate object boundaries at the pixel level.

**Instance Segmentation**: A computer vision task where each detected object is not only classified and localized (as in object detection) but also segmented at the pixel level, producing a mask for each instance.

**cxcywh Format**: A way to represent bounding boxes using the center coordinates (cx, cy), width (w), and height (h) of the box, typically normalized to the image size.

**xyxy Format**: A bounding box format using the coordinates of the top-left (x1, y1) and bottom-right (x2, y2) corners of the box.

**Logits**: The raw, unnormalized output values from a neural network's final layer before applying any activation function (such as sigmoid or softmax). Logits are the direct result of a linear transformation (logits = W·x + b) and can be any real number from -∞ to +∞. In well-trained networks, logits typically range from -3 to +3, though very confident predictions may reach -10 to +10. Logits must be converted to probabilities using activation functions: sigmoid for binary/multi-label classification (each output independent), or softmax for multi-class classification (outputs sum to 1).

**Applying Sigmoid to Class Logits**: The process of passing raw logits through the sigmoid function to convert them into probabilities between 0 and 1. Used primarily for multi-label classification where each class is evaluated independently, allowing an instance to belong to multiple classes simultaneously (e.g., an image tagged as both "cat" AND "indoors"). The sigmoid activation is $\sigma(x) = \frac{1}{1 + e^{-x}}$.

**Object Detection**: A computer vision task that identifies and localizes objects in images by predicting bounding boxes and class labels for each detected object.

**Confidence Threshold**: A minimum probability score that predictions must exceed to be considered valid detections. Predictions below this threshold are filtered out. The default value in this project is 0.5.

**Mask Threshold**: A value used to convert predicted mask probabilities into binary masks. Pixels above the threshold are considered part of the object; those below are background.

**Normalization**: The process of adjusting image pixel values using mean and standard deviation (e.g., ImageNet statistics) to improve model performance and stability.

**Top-k Selection**: Selecting the k highest scoring predictions (e.g., detections) from a set, often used to keep only the most confident results.

**Bilinear Interpolation**: A method for resizing images or masks that uses linear interpolation in two directions, resulting in smoother scaling compared to nearest-neighbor methods.

**Transparency Overlay**: Drawing masks or other visual elements with partial transparency (alpha blending) so that both the mask and the underlying image are visible.

**Tensor**: A multi-dimensional array used to represent data in machine learning. Model inputs and outputs are tensors with specific shapes, e.g., `float32[batch, num_queries, 4]` represents a 3-dimensional tensor.

**Tensor Shape Notation**: The notation `float32[batch, num_queries, mask_h, mask_w]` describes a tensor's properties: the data type (float32 = 32-bit floating-point) followed by dimensions in brackets. For segmentation masks like `float32[1, 300, 108, 108]`, this means: 1 image (batch), 300 possible detections (queries), each with a 108×108 pixel mask. The masks store probability values and are later resized to the original image dimensions using bilinear interpolation.

**Batch**: The number of samples processed simultaneously by a model. In the context of model outputs like `float32[batch, num_queries, 4]`, batch refers to the first dimension, typically 1 for single-image inference.

**Queries**: In transformer-based models like DETR, queries are learned embeddings that the model uses to detect objects. Each query can potentially detect one object, with `num_queries` determining the maximum number of detections (e.g., 300).

**BGR to RGB**: Color format conversion where Blue-Green-Red channel order (used by OpenCV) is converted to Red-Green-Blue order (expected by most neural networks).

**CHW Format**: Image data format where dimensions are ordered as Channel-Height-Width (e.g., 3×640×640 for an RGB image). Neural networks typically expect CHW format, while image libraries often use HWC (Height-Width-Channel).

**Binary Mask**: A mask where each pixel is either 0 (background) or 1 (object), created by applying a threshold to continuous mask probabilities.

**ImageNet**: A large-scale image dataset commonly used for training computer vision models. ImageNet statistics (mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`) are widely used for normalizing images before feeding them to pretrained models.

**Alpha Blending**: A technique for compositing images by combining foreground and background colors using an alpha (transparency) value, typically ranging from 0 (fully transparent) to 1 (fully opaque). Used to overlay segmentation masks on images.

**Softmax**: An activation function that converts a vector of logits into a probability distribution where all values sum to 1. Used for multi-class classification where each instance belongs to exactly one class. Formula: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$.

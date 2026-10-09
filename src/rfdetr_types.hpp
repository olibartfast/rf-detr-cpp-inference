#pragma once

#include <array>

/// Axis-aligned bounding box in xyxy pixel format.
struct BoundingBox {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
};

/// A single detected keypoint with associated metadata.
struct KeypointResult {
    float x;                  ///< Pixel x-coordinate
    float y;                  ///< Pixel y-coordinate
    float findability;        ///< Sigmoid(findability_logit) - [0, 1], radius multiplier
    float visibility;         ///< Sigmoid(visibility_logit) - [0, 1], occlusion flag
    std::array<float, 4> cov; ///< 2x2 pixel covariance matrix, row-major: [a, b, b, c]
};

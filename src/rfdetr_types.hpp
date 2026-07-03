#pragma once

/// A single detected keypoint with associated metadata.
struct KeypointResult {
    float x;           ///< Pixel x-coordinate
    float y;           ///< Pixel y-coordinate
    float findability; ///< Sigmoid(findability_logit) - [0, 1], radius multiplier
    float visibility;  ///< Sigmoid(visibility_logit) - [0, 1], occlusion flag
    float cov[4];      ///< 2x2 pixel covariance matrix, row-major: [a, b, b, c]
};

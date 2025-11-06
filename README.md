# JABuff: Josh's Audio Buffer Library

JABuff is a minimal, high-performance, header-only C++17 library for framing circular buffers, ideal for real-time audio applications.

It provides templated classes for managing blocks and frames of data, supporting overlapping reads (e.g., for STFT processing).

## Features
- Header-only: Just include the headers and go.
- Templated: Works with any data type (float, double, int, etc.).
- Block & Frame Based: Designed to write blocks of data (e.g., from an audio callback) and read overlapping frames.
- Performance: Uses std::memcpy for fast, efficient data copies.
- CMake-Ready: Includes a modern CMake setup for easy integration and examples.


## Classes

- `JABuff::2DFramingRingBuffer<T>`: A circular buffer for 2D data (e.g., num_channels x time). Ideal for standard audio samples.
- `JABuff::3DFramingRingBuffer<T>`: A circular buffer for 3D data (e.g., num_channels x time x feature_dim). Ideal for spectrograms or other feature matrices.

## Repository Organization
```
JABuff/
├── build/                  # (Created by you) CMake build output
├── include/
│   └── JABuff/
│       ├── 2DFramingRingBuffer.hpp
│       └── 3DFramingRingBuffer.hpp
├── src/
│   ├── CMakeLists.txt      # CMake config for the example
│   └── main.cpp            # Example usage and tests
├── CMakeLists.txt          # Top-level CMake config for the library
└── README.md
```

## Building the Example
This project includes a simple example in the `src/` directory.
```
# Clone the repository
git clone <your-repo-url>
cd JABuff

# Create a build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .

# Run the example
./src/jabuff_example
```

## Basic Usage
The buffers are designed to be written to in blocks and read from in overlapping frames.
```
#include "JABuff/2DFramingRingBuffer.hpp"
#include <vector>
#include <iostream>

int main() {
    size_t num_channels = 2;
    size_t capacity_features = 1024; // 1024 features/samples per channel
    size_t frame_size_features = 512;
    size_t hop_size_features = 128;

    // Create a buffer for 2 channels, 1024-sample capacity,
    // with a frame size of 512 and hop size of 128.
    JABuff::2DFramingRingBuffer<float> buffer(
        num_channels, 
        capacity_features, 
        frame_size_features, 
        hop_size_features
    );

    // Create some input data (2 channels x 256 samples)
    std::vector<std::vector<float>> input_data(num_channels, std::vector<float>(256, 1.0f));

    // Write the data
    if (buffer.write(input_data)) {
        std::cout << "Wrote 256 samples." << std::endl;
    }
    
    // Write more data to have enough for a read (total 512)
    buffer.write(input_data);
    std::cout << "Wrote another 256 samples." << std::endl;
    std::cout << "Available frames to read: " << buffer.getAvailableFramesRead() << std::endl;

    // Prepare an output frame (must be pre-sized)
    std::vector<std::vector<float>> output_frame(
        num_channels, 
        std::vector<float>(frame_size_features)
    );

    // Read the frame
    if (buffer.read(output_frame)) {
        std::cout << "Read one 512-sample frame." << std::endl;
        std::cout << "Available features after read: " << buffer.getAvailableFeaturesRead() << std::endl;
        std::cout << "Available frames after read: " << buffer.getAvailableFramesRead() << std::endl;
    }

    return 0;
}
```

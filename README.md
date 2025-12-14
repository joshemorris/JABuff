# JABuff: Josh's Audio Buffer Library

JABuff is a minimal, high-performance, header-only C++17 library for framing circular buffers, ideal for real-time audio applications.

It provides templated classes for managing blocks and frames of data, handling the logic of buffering input blocks and producing continuous blocks of data covering requested frames (handling overlaps internally).

## Features
- **Header-only:** Just include the headers and go.
- **Templated:** Works with any data type (float, double, int, etc.).
- **Block & Frame Based:** Write blocks of data (e.g., from audio callback). Read contiguous blocks covering specific frames.
- **Performance:** Uses `std::memcpy` for fast, efficient data copies.
- **Latency Control:** Includes a `prime()` method to pre-fill buffers for latency alignment
- **CMake-Ready:** Includes a modern CMake setup for easy integration and examples.
- **Robust Testing:** Includes a full CTest suite to verify logic, wrap-arounds, and exception handling.


## Classes

- `JABuff::FramingRingBuffer2D<T>`: A circular buffer for 2D data (e.g., num_channels x time). Ideal for standard audio samples.
- `JABuff::FramingRingBuffer3D<T>`: A circular buffer for 3D data (e.g., num_channels x time x feature_dim). Ideal for spectrograms or other feature matrices.

## Repository Organization
```
JABuff/
├── build/                  # (Created by you) CMake build output
├── include/
│   └── JABuff/
│       ├── FramingRingBuffer2D.hpp
│       └── FramingRingBuffer3D.hpp
├── src/
│   ├── CMakeLists.txt      # CMake config for the example
│   └── main.cpp            # Example usage and tests
├── tests/
│   ├── CMakeLists.txt      # CMake config for the test suite
│   ├── test_utils.hpp      # Testing helper macros
│   ├── test_2d.cpp         # Tests for 2D Buffer
│   ├── test_3d.cpp         # Tests for 3D Buffer
│   └── test_exceptions.cpp # Tests for error handling
├── CMakeLists.txt          # Top-level CMake config for the library
└── README.md
```

## Building
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
```

## Tests
The library includes a CTest suite to verify functionality.

### Linux / macOS (Single-Config Generators):
```
ctest --output-on-failure
```

### Windows (Visual Studio / Multi-Config Generators):
You must specify the configuration (Debug/Release) used during the build:
```
ctest -C Debug --output-on-failure
````

## Basic Usage
The buffers are designed to be written to in blocks. Reading requests a number of frames, and the buffer returns a contiguous block of memory covering the time span of those frames (without duplicating overlapping samples).
```
#include "JABuff/FramingRingBuffer2D.hpp"
#include <vector>
#include <iostream>

int main() {
    size_t num_channels = 2;
    size_t capacity_features = 1024; // 1024 features/samples per channel
    size_t frame_size_features = 512;
    size_t hop_size_features = 128;
    // Optional: size_t min_frames = 1; // Default is 1 (requires full frame)
    // Optional: size_t keep_frames = 0; // Default is 0

    // Create a buffer for 2 channels, 1024-sample capacity,
    // with a frame size of 512 and hop size of 128.
    JABuff::FramingRingBuffer2D<float> buffer(
        num_channels, 
        capacity_features, 
        frame_size_features, 
        hop_size_features
    );

    // Optional: Prime the buffer.
    // This fills the buffer with enough silence (0.0f) such that the *next*
    // write of exactly 'hop_size' will make a full frame available.
    // Useful for minimizing initial latency or aligning block processing.
    buffer.prime(0.0f);

    // Create some input data (2 channels x 128 samples - exactly one hop)
    std::vector<std::vector<float>> input_data(num_channels, std::vector<float>(128, 1.0f));

    // Write the data
    // Because we primed the buffer, this single hop write makes the first frame ready!
    if (buffer.write(input_data)) {
        std::cout << "Wrote 128 samples." << std::endl;
    }
    
    std::cout << "Available frames to read: " << buffer.getAvailableFramesRead() << std::endl;

    // Prepare an output buffer. It will be resized automatically.
    // Layout: [Channel][Contiguous Samples]
    // The output contains the union of samples for the requested frames.
    // Size = (num_frames - 1) * hop_size + frame_size
    std::vector<std::vector<float>> output_buffer;

    // Read the frame
    if (buffer.read(output_buffer)) {
        std::cout << "Read one 512-sample frame." << std::endl;
        std::cout << "Available features after read: " << buffer.getAvailableFeaturesRead() << std::endl;
        
        // Accessing data: output_buffer[channel][sample_index]
        float val = output_buffer[0][0];
    }

    return 0;
}
```

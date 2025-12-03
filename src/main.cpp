#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <limits>  // For max()

#include "JABuff/FramingRingBuffer2D.hpp"
#include "JABuff/FramingRingBuffer3D.hpp"

void Test2DBuffer() {
    std::cout << "--- Testing FramingRingBuffer2D ---" << std::endl;
    
    size_t num_channels = 2;
    size_t capacity = 1024; // 1024 features/samples per channel
    size_t frame_size = 512;
    size_t hop_size = 128;

    JABuff::FramingRingBuffer2D<float> buffer(num_channels, capacity, frame_size, hop_size);

    std::cout << "Buffer created. Capacity: " << buffer.getCapacity() << " features." << std::endl;
    std::cout << "Frame Size: " << buffer.getFrameSizeFeatures() << ", Hop Size: " << buffer.getHopSizeFeatures() << std::endl;
    std::cout << "Is empty? " << std::boolalpha << buffer.isEmpty() << std::endl;

    // --- Test Write ---
    size_t write_size = 256;
    std::vector<std::vector<float>> input_data(num_channels, std::vector<float>(write_size));
    // Fill with 0, 1, 2... for channel 0
    std::iota(input_data[0].begin(), input_data[0].end(), 0.0f);
    // Fill with 1000, 1001... for channel 1
    std::iota(input_data[1].begin(), input_data[1].end(), 1000.0f);


    if (buffer.write(input_data)) {
        std::cout << "Wrote " << write_size << " features." << std::endl;
    } else {
        std::cout << "Failed to write features." << std::endl;
    }
    
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // Should be 256
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // Should be 0
    std::cout << "Is empty? " << std::boolalpha << buffer.isEmpty() << std::endl;

    // --- Test Read (not enough data) ---
    // Note: Frames vector will be resized by read()
    std::vector<std::vector<std::vector<float>>> frames_out;

    if (!buffer.read(frames_out)) {
        std::cout << "Read failed (as expected, not enough data for frame)." << std::endl;
    }

    // --- Write more data ---
    if (buffer.write(input_data)) {
        std::cout << "Wrote another " << write_size << " features." << std::endl;
    }
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // Should be 512
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // Should be 1

    // --- Test Read (should succeed) ---
    if (buffer.read(frames_out)) { // Defaults to 1 frame
        size_t read_count = frames_out.size();
        // Access the first frame (index 0)
        std::cout << "Read " << read_count << " frame(s)." << std::endl;
        std::cout << "Frame size: " << (frames_out[0].empty() ? 0 : frames_out[0][0].size()) << std::endl;
        std::cout << "Channel 0, first element: " << frames_out[0][0][0] << std::endl; // 0.0
        std::cout << "Channel 1, first element: " << frames_out[0][1][0] << std::endl; // 1000.0
        std::cout << "Channel 0, last element: " << frames_out[0][0].back() << std::endl; // 255.0 (from 2nd write)
        std::cout << "Available features after read: " << buffer.getAvailableFeaturesRead() << std::endl; // Should be 512 - 128 = 384
    }

    std::cout << "Available features after read: " << buffer.getAvailableFeaturesRead() << std::endl; // 384
    std::cout << "Available frames after read: " << buffer.getAvailableFramesRead() << std::endl; // 0

    // --- Test Full ---
    std::cout << "Filling buffer..." << std::endl;
    buffer.clear();
    std::cout << "Available to write: " << buffer.getAvailableWrite() << std::endl;
    for(int i = 0; i < 4; ++i) { // 4 * 256 = 1024
        buffer.write(input_data);
    }
    std::cout << "Available to write: " << buffer.getAvailableWrite() << std::endl; // 0
    std::cout << "Is full? " << std::boolalpha << buffer.isFull() << std::endl;
    
    // This one should fail
    if (!buffer.write(input_data)) {
        std::cout << "Write failed (as expected, buffer is full)." << std::endl;
    }

    // --- Test wrap-around read ---
    std::cout << "Testing wrap-around read..." << std::endl;
    buffer.clear();
    // 1. Write 768 features
    std::vector<std::vector<float>> partial_data(num_channels, std::vector<float>(768));
    std::iota(partial_data[0].begin(), partial_data[0].end(), 0.0f);
    std::iota(partial_data[1].begin(), partial_data[1].end(), 0.0f);
    buffer.write(partial_data); // write_index = 768, available = 768
    
    // 2. Read 512, hop 256
    buffer.read(frames_out); // read_index = 256, available = 512
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 512
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1
    
    // 3. Write 512 features. This will wrap.
    std::vector<std::vector<float>> wrap_data(num_channels, std::vector<float>(512));
    std::iota(wrap_data[0].begin(), wrap_data[0].end(), 10000.0f);
    std::iota(wrap_data[1].begin(), wrap_data[1].end(), 10000.0f);
    buffer.write(wrap_data); // write_index = 256, available = 1024
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 1024
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (1024-512)/128 = 1 + 512/128 = 5
    
    // 4. Read 512, hop 256. Starts from read_index 256.
    //    Should read data [256..767]
    buffer.read(frames_out); // read_index = 512, available = 768
    std::cout << "Read frame. First val: " << frames_out[0][0][0] << std::endl; // 256.0
    std::cout << "Read frame. Last val: " << frames_out[0][0].back() << std::endl; // 767.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 768
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (768-512)/128 = 1 + 256/128 = 3

    // 5. Read 512, hop 256. Starts from read_index 512.
    //    Should read data [512..1023] (wraps in buffer)
    buffer.read(frames_out); // read_index = 768, available = 512
    std::cout << "Read wrapped frame." << std::endl;
    std::cout << "First val: " << frames_out[0][0][0] << std::endl; // 512.0
    std::cout << "Val at 255: " << frames_out[0][0][255] << std::endl; // 767.0
    std::cout << "Val at 256: " << frames_out[0][0][256] << std::endl; // 10000.0
    std::cout << "Last val: " << frames_out[0][0].back() << std::endl; // 10255.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 512
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1

    // 6. Read 512, hop 256. Starts from read_index 768.
    //    Should read data [768..1023] and [0..255] (wraps in buffer)
    buffer.read(frames_out); // read_index = 0, available = 256
    std::cout << "Read wrapped frame 2." << std::endl;
    std::cout << "First val: " << frames_out[0][0][0] << std::endl; // 10000.0
    std::cout << "Val at 255: " << frames_out[0][0][255] << std::endl; // 10255.0
    std::cout << "Val at 256: " << frames_out[0][0][256] << std::endl; // 10256.0
    std::cout << "Last val: " << frames_out[0][0].back() << std::endl; // 10511.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 256
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 0
    
    // 7. Read 512, hop 256.
    if (!buffer.read(frames_out)) {
        std::cout << "Read failed (as expected, not enough data)." << std::endl;
    }
}

void Test3DBuffer() {
    std::cout << "\n--- Testing FramingRingBuffer3D ---" << std::endl;
    
    size_t num_channels = 2;
    size_t feature_dim = 4;
    size_t capacity_time = 100;
    size_t frame_size_time = 10;
    size_t hop_size_time = 5;

    JABuff::FramingRingBuffer3D<float> buffer(
        num_channels, feature_dim, capacity_time, frame_size_time, hop_size_time);

    std::cout << "3D Buffer created." << std::endl;
    std::cout << "Capacity: " << buffer.getCapacity() << " time steps." << std::endl;
    std::cout << "Frame Size: " << buffer.getFrameSizeTime() << " time steps." << std::endl;
    std::cout << "Feature Dim: " << buffer.getFeatureDim() << std::endl;
    std::cout << "Is empty? " << std::boolalpha << buffer.isEmpty() << std::endl;

    // --- Test Write ---
    size_t write_time_steps = 8;
    std::vector<std::vector<std::vector<float>>> input_data(
        num_channels, 
        std::vector<std::vector<float>>(
            write_time_steps, 
            std::vector<float>(feature_dim)
        )
    );

    // Fill with 0, 1, 2, 3...
    float val = 0.0f;
    for (size_t c = 0; c < num_channels; ++c) {
        for (size_t t = 0; t < write_time_steps; ++t) {
            std::iota(input_data[c][t].begin(), input_data[c][t].end(), val);
            val += 10.0f; // Each time step starts 10 higher
        }
    }

    if (buffer.write(input_data)) {
        std::cout << "Wrote " << write_time_steps << " time steps." << std::endl;
    } else {
        std::cout << "Failed to write data." << std::endl;
    }

    std::cout << "Available time: " << buffer.getAvailableTimeRead() << std::endl; // 8
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 0

    // --- Test Read (not enough data) ---
    // [Frames][Channel][Time][Feature]
    std::vector<std::vector<std::vector<std::vector<float>>>> frames_out;

    if (!buffer.read(frames_out)) {
        std::cout << "Read failed (as expected, not enough data)." << std::endl;
    }

    // --- Write more data ---
    if (buffer.write(input_data)) {
        std::cout << "Wrote " << write_time_steps << " time steps." << std::endl;
    }
    std::cout << "Available time: " << buffer.getAvailableTimeRead() << std::endl; // 16
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (16-10)/5 = 2

    // --- Test Read (should succeed) ---
    if (buffer.read(frames_out)) {
        std::cout << "Read one 3D frame." << std::endl;
        // Check first value: Frame 0, channel 0, time 0, feature 0
        std::cout << "Ch 0, T 0, F 0: " << frames_out[0][0][0][0] << std::endl; // 0.0
        // Check last value of first block: ch 0, time 7, feature 3
        std::cout << "Ch 0, T 7, F 3: " << frames_out[0][0][7][3] << std::endl; // 73.0
        // Check first value of second block: ch 0, time 8, feature 0
        std::cout << "Ch 0, T 8, F 0: " << frames_out[0][0][8][0] << std::endl; // 80.0
        // Check last value of frame: ch 0, time 9, feature 3
        std::cout << "Ch 0, T 9, F 3: " << frames_out[0][0][9][3] << std::endl; // 93.0
        // Check a value from channel 1
        std::cout << "Ch 1, T 0, F 0: " << frames_out[0][1][0][0] << std::endl; // 80.0
        std::cout << "Ch 1, T 9, F 3: " << frames_out[0][1][9][3] << std::endl; // 173.0
    } else {
        std::cout << "Failed to read 3D frame (unexpected)." << std::endl;
    }

    std::cout << "Available time after read: " << buffer.getAvailableTimeRead() << std::endl; // 16 - 5 = 11
    std::cout << "Available frames after read: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (11-10)/5 = 1
}

void TestOffsetWrite() {
    std::cout << "\n--- Testing Offset Writes (2D) ---" << std::endl;
    
    size_t num_channels = 1;
    size_t capacity = 100;
    size_t frame_size = 10;
    size_t hop_size = 5;

    JABuff::FramingRingBuffer2D<float> buffer(num_channels, capacity, frame_size, hop_size);

    // Create a vector of 20 elements: 0, 1, 2... 19
    std::vector<std::vector<float>> input_data(num_channels, std::vector<float>(20));
    std::iota(input_data[0].begin(), input_data[0].end(), 0.0f);

    // Write only elements 5 through 9 (offset 5, length 5)
    // Expected to write: 5, 6, 7, 8, 9
    buffer.write(input_data, 5, 5);

    std::cout << "Available features after slice write: " << buffer.getAvailableFeaturesRead() << std::endl; // 5

    // Write elements 15 to end (offset 15, length 0 -> auto calc)
    // Expected to write: 15, 16, 17, 18, 19
    buffer.write(input_data, 15); 
    
    std::cout << "Available features after 2nd slice write: " << buffer.getAvailableFeaturesRead() << std::endl; // 10

    // Read a frame (size 10)
    // Should be: 5, 6, 7, 8, 9, 15, 16, 17, 18, 19
    std::vector<std::vector<std::vector<float>>> frames_out;
    if (buffer.read(frames_out)) {
        std::cout << "Read Frame: ";
        for (float f : frames_out[0][0]) {
            std::cout << f << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Failed to read frame." << std::endl;
    }
}

void TestOffsetWrite3D() {
    std::cout << "\n--- Testing Offset Writes (3D) ---" << std::endl;
    
    size_t num_channels = 1;
    size_t feature_dim = 2;
    // Capacity 100, Frame 10, Hop 5
    JABuff::FramingRingBuffer3D<float> buffer(num_channels, feature_dim, 100, 10, 5);

    // Create 20 time steps of data
    // t=0..19. val = t
    std::vector<std::vector<std::vector<float>>> input(
        num_channels, 
        std::vector<std::vector<float>>(
            20, 
            std::vector<float>(feature_dim)
        )
    );

    for(size_t t=0; t<20; ++t) {
        input[0][t][0] = (float)t;       // Feature 0 = t
        input[0][t][1] = (float)t + 0.5f; // Feature 1 = t + 0.5
    }

    // 1. Write slice: offset 5, length 5. (Indices 5, 6, 7, 8, 9)
    buffer.write(input, 5, 5);
    std::cout << "Written 5 steps (offset 5). Available time: " << buffer.getAvailableTimeRead() << std::endl;

    // 2. Write slice: offset 15, length 5. (Indices 15, 16, 17, 18, 19)
    buffer.write(input, 15, 5);
    std::cout << "Written 5 steps (offset 15). Available time: " << buffer.getAvailableTimeRead() << std::endl;

    // Total 10 steps. Frame size is 10. Should be able to read 1 frame.
    // That frame should contain [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    std::vector<std::vector<std::vector<std::vector<float>>>> frames;
    if (buffer.read(frames, 1)) {
        std::cout << "Read 1 frame." << std::endl;
        // Verify contents
        bool match = true;
        
        // Check index 0 (was input index 5)
        if (frames[0][0][0][0] != 5.0f) match = false;
        
        // Check index 4 (was input index 9)
        if (frames[0][0][4][0] != 9.0f) match = false;

        // Check index 5 (was input index 15)
        if (frames[0][0][5][0] != 15.0f) match = false;

        // Check index 9 (was input index 19)
        if (frames[0][0][9][0] != 19.0f) match = false;

        if (match) {
            std::cout << "[Success] Data matches expected slices." << std::endl;
        } else {
             std::cout << "[Fail] Data mismatch." << std::endl;
             std::cout << "Index 0 val: " << frames[0][0][0][0] << " (Expected 5.0)" << std::endl;
             std::cout << "Index 5 val: " << frames[0][0][5][0] << " (Expected 15.0)" << std::endl;
        }
    } else {
        std::cout << "[Fail] Could not read frame." << std::endl;
    }
}

void TestExceptions() {
    std::cout << "\n--- Testing Exception Handling ---" << std::endl;

    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5);
    std::vector<std::vector<float>> input(1, std::vector<float>(20, 1.0f));

    // 1. Test Offset out of range
    try {
        buffer.write(input, 50); // Size is 20, offset 50 is invalid
    } catch (const std::out_of_range& e) {
        std::cout << "[Success] Caught expected out_of_range: " << e.what() << std::endl;
    }

    // 2. Test Offset + Count out of range
    try {
        buffer.write(input, 15, 10); // Offset 15 + 10 = 25 > 20
    } catch (const std::out_of_range& e) {
        std::cout << "[Success] Caught expected out_of_range: " << e.what() << std::endl;
    }

    // 3. Test Channel Mismatch
    std::vector<std::vector<float>> bad_channel_input(2, std::vector<float>(10));
    try {
        buffer.write(bad_channel_input);
    } catch (const std::invalid_argument& e) {
        std::cout << "[Success] Caught expected invalid_argument: " << e.what() << std::endl;
    }

    // 4. Test Buffer Full (Should NOT throw, should return false)
    buffer.clear();
    // Fill buffer (capacity 100)
    buffer.write(std::vector<std::vector<float>>(1, std::vector<float>(100)));
    
    // Try to write more
    bool result = buffer.write(input);
    if (!result) {
        std::cout << "[Success] Buffer full returned false (no exception thrown)." << std::endl;
    } else {
        std::cout << "[Fail] Buffer full returned true?" << std::endl;
    }
}

void TestVariableRead2D() {
    std::cout << "\n--- Testing Variable Read (2D) ---" << std::endl;
    // Capacity 100, Frame 10, Hop 5
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5);
    
    // Write 20 items: 0..19
    std::vector<std::vector<float>> input(1, std::vector<float>(20));
    std::iota(input[0].begin(), input[0].end(), 0.0f);
    buffer.write(input);

    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // Should be 3: [0-9], [5-14], [10-19]

    std::vector<std::vector<std::vector<float>>> frames;
    
    // 1. Read specific count (2)
    // Should succeed because 3 available > 2 requested
    if (buffer.read(frames, 2)) {
        std::cout << "Requested 2, read " << frames.size() << " frames." << std::endl;
        std::cout << "Frame 0 start: " << frames[0][0][0] << " (Expected 0)" << std::endl;
        std::cout << "Frame 1 start: " << frames[1][0][0] << " (Expected 5)" << std::endl;
    } else {
        std::cout << "[Fail] Failed to read 2 frames." << std::endl;
    }

    // 2. Read remaining STRICT (Request 100)
    // Should FAIL because only 1 available < 100 requested
    std::cout << "Requesting 100 frames (Strict check)..." << std::endl;
    if (!buffer.read(frames, 100)) { 
        std::cout << "[Success] Strict read failed (returned false) as expected." << std::endl;
    } else {
        std::cout << "[Fail] Strict read returned true?" << std::endl;
    }

    // 3. Read remaining ALL (Request 0)
    // Should SUCCEED and read the 1 remaining frame
    std::cout << "Requesting ALL frames (0)..." << std::endl;
    if (buffer.read(frames, 0)) {
         std::cout << "[Success] Read " << frames.size() << " frame(s)." << std::endl;
         std::cout << "Frame 0 start: " << frames[0][0][0] << " (Expected 10)" << std::endl;
    } else {
        std::cout << "[Fail] Read All failed." << std::endl;
    }
}

void TestVariableRead3D() {
    std::cout << "\n--- Testing Variable Read (3D) ---" << std::endl;
    // Capacity 100, Frame 10, Hop 5, Feature Dim 2
    size_t num_channels = 1;
    size_t feature_dim = 2;
    JABuff::FramingRingBuffer3D<float> buffer(num_channels, feature_dim, 100, 10, 5);
    
    // Write 20 time steps
    std::vector<std::vector<std::vector<float>>> input(
        num_channels, 
        std::vector<std::vector<float>>(
            20, 
            std::vector<float>(feature_dim)
        )
    );

    // Fill data: Time t, Feature f -> val = t * 10 + f
    for(size_t t=0; t<20; ++t) {
        input[0][t][0] = (float)t * 10.0f;     // Feature 0
        input[0][t][1] = (float)t * 10.0f + 1.0f; // Feature 1
    }

    buffer.write(input);

    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // Should be 3: [0-9], [5-14], [10-19]

    std::vector<std::vector<std::vector<std::vector<float>>>> frames;
    
    // 1. Read specific count (2)
    // Should succeed
    if (buffer.read(frames, 2)) {
        std::cout << "Requested 2, read " << frames.size() << " frames." << std::endl;
        // Frame 0 starts at t=0. Val at t=0,f=0 is 0.
        std::cout << "Frame 0, T=0, F=0: " << frames[0][0][0][0] << " (Expected 0)" << std::endl;
        // Frame 1 starts at t=5. Val at t=0(relative),f=0 is 50.
        std::cout << "Frame 1, T=0, F=0: " << frames[1][0][0][0] << " (Expected 50)" << std::endl;
    } else {
        std::cout << "[Fail] Failed to read 2 frames." << std::endl;
    }

    // 2. Read remaining STRICT (Request 100)
    // Should FAIL
    std::cout << "Requesting 100 frames (Strict check)..." << std::endl;
    if (!buffer.read(frames, 100)) { 
        std::cout << "[Success] Strict read failed (returned false) as expected." << std::endl;
    } else {
        std::cout << "[Fail] Strict read returned true?" << std::endl;
    }

    // 3. Read remaining ALL (Request 0)
    // Should SUCCEED and read the 1 remaining frame (starts at t=10)
    std::cout << "Requesting ALL frames (0)..." << std::endl;
    if (buffer.read(frames, 0)) {
         std::cout << "[Success] Read " << frames.size() << " frame(s)." << std::endl;
         // Frame starts at t=10. Val is 100.
         std::cout << "Frame 0, T=0, F=0: " << frames[0][0][0][0] << " (Expected 100)" << std::endl;
    } else {
        std::cout << "[Fail] Read All failed." << std::endl;
    }
}

int main() {
    std::cout << "JABuff Example Application" << std::endl;
    
    Test2DBuffer();
    Test3DBuffer();
    TestOffsetWrite();
    TestOffsetWrite3D();
    TestExceptions();
    TestVariableRead2D();
    TestVariableRead3D();
    
    return 0;
}

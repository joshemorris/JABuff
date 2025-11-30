#include <iostream>
#include <vector>
#include <numeric> // For std::iota

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
    std::vector<std::vector<float>> output_frame(num_channels, std::vector<float>(frame_size));


    if (!buffer.read(output_frame)) {
        std::cout << "Read failed (as expected, not enough data for frame)." << std::endl;
    }

    // --- Write more data ---
    if (buffer.write(input_data)) {
        std::cout << "Wrote another " << write_size << " features." << std::endl;
    }
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // Should be 512
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // Should be 1

    // --- Test Read (should succeed) ---
    if (buffer.read(output_frame)) {
        std::cout << "Read one frame. Frame size: " << (output_frame.empty() ? 0 : output_frame[0].size()) << std::endl;
        std::cout << "Channel 0, first element: " << output_frame[0][0] << std::endl; // 0.0
        std::cout << "Channel 1, first element: " << output_frame[1][0] << std::endl; // 1000.0
        std::cout << "Channel 0, last element: " << output_frame[0].back() << std::endl; // 255.0 (from 2nd write)
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
    buffer.read(output_frame); // read_index = 256, available = 512
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
    buffer.read(output_frame); // read_index = 512, available = 768
    std::cout << "Read frame. First val: " << output_frame[0][0] << std::endl; // 256.0
    std::cout << "Read frame. Last val: " << output_frame[0].back() << std::endl; // 767.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 768
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (768-512)/128 = 1 + 256/128 = 3

    // 5. Read 512, hop 256. Starts from read_index 512.
    //    Should read data [512..1023] (wraps in buffer)
    //    Data [512..767] from partial_data (vals 512..767)
    //    Data [768..1023] from wrap_data (vals 10000..10255)
    buffer.read(output_frame); // read_index = 768, available = 512
    std::cout << "Read wrapped frame." << std::endl;
    std::cout << "First val: " << output_frame[0][0] << std::endl; // 512.0
    std::cout << "Val at 255: " << output_frame[0][255] << std::endl; // 767.0
    std::cout << "Val at 256: " << output_frame[0][256] << std::endl; // 10000.0
    std::cout << "Last val: " << output_frame[0].back() << std::endl; // 10255.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 512
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1

    // 6. Read 512, hop 256. Starts from read_index 768.
    //    Should read data [768..1023] and [0..255] (wraps in buffer)
    //    Data [768..1023] from wrap_data (vals 10000..10255)
    //    Data [0..255] from wrap_data (vals 10256..10511)
    buffer.read(output_frame); // read_index = 0, available = 256
    std::cout << "Read wrapped frame 2." << std::endl;
    std::cout << "First val: " << output_frame[0][0] << std::endl; // 10000.0
    std::cout << "Val at 255: " << output_frame[0][255] << std::endl; // 10255.0
    std::cout << "Val at 256: " << output_frame[0][256] << std::endl; // 10256.0
    std::cout << "Last val: " << output_frame[0].back() << std::endl; // 10511.0
    std::cout << "Available features: " << buffer.getAvailableFeaturesRead() << std::endl; // 256
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 0
    
    // 7. Read 512, hop 256.
    //    Fails, available is 256, frame_size is 512.
    if (!buffer.read(output_frame)) {
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
    std::vector<std::vector<std::vector<float>>> output_frame(
        num_channels,
        std::vector<std::vector<float>>(
            frame_size_time,
            std::vector<float>(feature_dim)
        )
    );

    if (!buffer.read(output_frame)) {
        std::cout << "Read failed (as expected, not enough data)." << std::endl;
    }

    // --- Write more data ---
    if (buffer.write(input_data)) {
        std::cout << "Wrote " << write_time_steps << " time steps." << std::endl;
    }
    std::cout << "Available time: " << buffer.getAvailableTimeRead() << std::endl; // 16
    std::cout << "Available frames: " << buffer.getAvailableFramesRead() << std::endl; // 1 + (16-10)/5 = 2

    // --- Test Read (should succeed) ---
    if (buffer.read(output_frame)) {
        std::cout << "Read one 3D frame." << std::endl;
        // Check first value: channel 0, time 0, feature 0
        std::cout << "Ch 0, T 0, F 0: " << output_frame[0][0][0] << std::endl; // 0.0
        // Check last value of first block: ch 0, time 7, feature 3
        std::cout << "Ch 0, T 7, F 3: " << output_frame[0][7][3] << std::endl; // 73.0
        // Check first value of second block: ch 0, time 8, feature 0
        std::cout << "Ch 0, T 8, F 0: " << output_frame[0][8][0] << std::endl; // 80.0
        // Check last value of frame: ch 0, time 9, feature 3
        std::cout << "Ch 0, T 9, F 3: " << output_frame[0][9][3] << std::endl; // 93.0
        // Check a value from channel 1
        std::cout << "Ch 1, T 0, F 0: " << output_frame[1][0][0] << std::endl; // 80.0
        std::cout << "Ch 1, T 9, F 3: " << output_frame[1][9][3] << std::endl; // 173.0
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
    std::vector<std::vector<float>> output_frame(num_channels, std::vector<float>(frame_size));
    if (buffer.read(output_frame)) {
        std::cout << "Read Frame: ";
        for (float f : output_frame[0]) {
            std::cout << f << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Failed to read frame." << std::endl;
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

int main() {
    std::cout << "JABuff Example Application" << std::endl;
    
    Test2DBuffer();
    Test3DBuffer();
    TestOffsetWrite();
    TestExceptions();
    
    return 0;
}

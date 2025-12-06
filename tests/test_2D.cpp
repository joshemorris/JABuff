#include "JABuff/FramingRingBuffer2D.hpp"
#include "test_utils.hpp"
#include <numeric>

void TestBasicFlow() {
    print_header("TestBasicFlow");
    size_t num_channels = 2;
    size_t capacity = 1024;
    size_t frame_size = 512;
    size_t hop_size = 128;

    JABuff::FramingRingBuffer2D<float> buffer(num_channels, capacity, frame_size, hop_size);

    ASSERT(buffer.isEmpty(), "Buffer should be empty on init");
    ASSERT(buffer.getCapacity() == 1024, "Capacity incorrect");

    // Write 256
    std::vector<std::vector<float>> input(num_channels, std::vector<float>(256));
    std::iota(input[0].begin(), input[0].end(), 0.0f);
    std::iota(input[1].begin(), input[1].end(), 1000.0f);

    bool res = buffer.write(input);
    ASSERT(res, "Write failed");
    ASSERT(buffer.getAvailableFeaturesRead() == 256, "Available features mismatch (1)");
    ASSERT(buffer.getAvailableFramesRead() == 0, "Should have 0 frames");

    // Write another 256 (Total 512)
    buffer.write(input);
    ASSERT(buffer.getAvailableFeaturesRead() == 512, "Available features mismatch (2)");
    ASSERT(buffer.getAvailableFramesRead() == 1, "Should have 1 frame");

    // Read
    std::vector<std::vector<float>> out;
    res = buffer.read(out);
    ASSERT(res, "Read failed");
    ASSERT(out[0].size() == 512, "Output size mismatch");
    ASSERT_NEAR(out[0][0], 0.0f, 0.001f, "Ch0 Data mismatch");
    ASSERT_NEAR(out[1][0], 1000.0f, 0.001f, "Ch1 Data mismatch");
}

void TestWrapAround() {
    print_header("TestWrapAround");
    // Setup: Capacity 1024, Frame 512, Hop 128
    JABuff::FramingRingBuffer2D<float> buffer(2, 1024, 512, 128);
    
    // 1. Write 768 features
    // [0.........767] (Empty: 768..1023)
    std::vector<std::vector<float>> data(2, std::vector<float>(768, 1.0f));
    buffer.write(data);
    
    // 2. Read 1 frame
    // Consumes Hop Size (128).
    // Remaining Features: 768 - 128 = 640.
    // Read Ptr moves to 128. Write Ptr is at 768.
    std::vector<std::vector<float>> out;
    buffer.read(out); 
    
    // 3. Write more data
    // Available Space = Capacity (1024) - Available (640) = 384.
    // We write exactly 384 to fill the buffer and force a wrap-around of the write pointer.
    // Write Ptr 768 + 384 = 1152. 1152 % 1024 = 128.
    // New data wraps: [768..1023] and [0..127].
    std::vector<std::vector<float>> data2(2, std::vector<float>(384, 2.0f));
    bool write_success = buffer.write(data2);
    
    ASSERT(write_success, "Write failed (should fit exactly)");
    ASSERT(buffer.getAvailableFeaturesRead() == 1024, "Buffer should be full");
    
    // 4. Read wrapped data
    // We want to verify we can read valid data now that the buffer is full and wrapped.
    // Current Read Ptr: 128.
    // We want to read 4 frames.
    // 4 Frames = (3 hops * 128) + 1 frame (512) = 384 + 512 = 896 samples.
    // This is valid (896 < 1024).
    buffer.read(out, 4);
    
    ASSERT(out[0].size() == 896, "Read size mismatch");
}

void TestOffsetWrite() {
    print_header("TestOffsetWrite");
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5);
    std::vector<std::vector<float>> input(1, std::vector<float>(20));
    std::iota(input[0].begin(), input[0].end(), 0.0f);

    // Write index 5..9 (5 items)
    buffer.write(input, 5, 5);
    ASSERT(buffer.getAvailableFeaturesRead() == 5, "Offset write size fail");

    std::vector<std::vector<float>> out;
    // Not enough for a frame (need 10)
    ASSERT(!buffer.read(out), "Should not be able to read");

    // Write index 15..19 (5 items)
    buffer.write(input, 15);
    ASSERT(buffer.getAvailableFeaturesRead() == 10, "Total size fail");

    // Read frame
    buffer.read(out);
    ASSERT_NEAR(out[0][0], 5.0f, 0.001f, "Data index 0 incorrect");
    ASSERT_NEAR(out[0][4], 9.0f, 0.001f, "Data index 4 incorrect");
    ASSERT_NEAR(out[0][5], 15.0f, 0.001f, "Data index 5 incorrect");
}

void TestVariableRead() {
    print_header("TestVariableRead");
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5);
    std::vector<std::vector<float>> input(1, std::vector<float>(20, 1.0f));
    buffer.write(input);

    // Available: 3 frames ([0-9], [5-14], [10-19])
    std::vector<std::vector<float>> out;
    
    // Request 2
    ASSERT(buffer.read(out, 2), "Read 2 frames failed");
    ASSERT(out[0].size() == 15, "Read 2 frames size calc failed");

    // Request 100 (Fail)
    ASSERT(!buffer.read(out, 100), "Read 100 should fail");

    // Request All (0) -> Should be 1 remaining
    ASSERT(buffer.read(out, 0), "Read all failed");
    ASSERT(out[0].size() == 10, "Read all size calc failed");
}

void TestKeepFrames() {
    print_header("TestKeepFrames");
    // Keep 1 frame
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5, 1, 1);
    std::vector<std::vector<float>> input(1, std::vector<float>(20));
    std::iota(input[0].begin(), input[0].end(), 0.0f);
    buffer.write(input);

    std::vector<std::vector<float>> out;
    
    // Read 1, Keep 1 -> Consumes 0
    buffer.read(out, 1);
    ASSERT_NEAR(out[0][0], 0.0f, 0.001f, "Peek frame data mismatch");
    ASSERT(buffer.getAvailableFeaturesRead() == 20, "Should not consume data");

    // Read 2, Keep 1 -> Consumes 1 frame (5 samples)
    buffer.read(out, 2);
    ASSERT_NEAR(out[0][0], 0.0f, 0.001f, "Read frame 0 data mismatch");
    ASSERT_NEAR(out[0][5], 5.0f, 0.001f, "Read frame 1 data mismatch");
    ASSERT(buffer.getAvailableFeaturesRead() == 15, "Consumption mismatch");
}

void TestPush() {
    print_header("TestPush");
    JABuff::FramingRingBuffer2D<float> buffer(2, 10, 5, 2);
    std::vector<float> frame(2);
    
    for(int i=0; i<5; ++i) {
        frame[0] = (float)i;
        frame[1] = (float)i + 10.0f;
        buffer.push(frame);
    }
    
    ASSERT(buffer.getAvailableFeaturesRead() == 5, "Push count mismatch");
    
    std::vector<std::vector<float>> out;
    ASSERT(buffer.read(out, 1), "Read pushed data failed");
    ASSERT_NEAR(out[0].back(), 4.0f, 0.001f, "Data validation failed");
}

void TestReady() {
    print_header("TestReady");
    // Min frames = 2
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5, 2);
    
    std::vector<std::vector<float>> input(1, std::vector<float>(10, 1.0f)); // 1 frame
    
    // 0 frames
    ASSERT(!buffer.ready(), "Should not be ready (empty)");
    
    // 1 frame
    buffer.write(input);
    ASSERT(!buffer.ready(), "Should not be ready (1 frame < min 2)");
    
    // 2 frames
    buffer.write(input);
    ASSERT(buffer.ready(), "Should be ready (2 frames == min 2)");
}

int main() {
    TestBasicFlow();
    TestWrapAround();
    TestOffsetWrite();
    TestVariableRead();
    TestKeepFrames();
    TestPush();
    TestReady();
    print_pass();
    return 0;
}

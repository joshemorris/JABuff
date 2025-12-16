#include "JABuff/OLARingBuffer2D.hpp"
#include "test_utils.hpp"
#include <vector>
#include <numeric>

// Helper to create a block of data
template <typename T>
std::vector<std::vector<T>> create_block(size_t channels, size_t samples, T value) {
    return std::vector<std::vector<T>>(channels, std::vector<T>(samples, value));
}

// Helper to create a ramping block
template <typename T>
std::vector<std::vector<T>> create_ramp(size_t channels, size_t samples, T start_val) {
    std::vector<std::vector<T>> data(channels, std::vector<T>(samples));
    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < samples; ++i) {
            data[c][i] = start_val + static_cast<T>(i);
        }
    }
    return data;
}

// --- Tests ---

void TestInitialization() {
    print_header("TestInitialization");
    size_t channels = 2;
    size_t capacity = 1000;
    size_t frame = 100;
    size_t overlap = 25;

    JABuff::OLARingBuffer2D<float> buffer(channels, capacity, frame, overlap);

    ASSERT(buffer.getNumChannels() == channels, "Channel count mismatch");
    ASSERT(buffer.getCapacity() == capacity, "Capacity mismatch");
    ASSERT(buffer.getFrameSize() == frame, "Frame size mismatch");
    ASSERT(buffer.getOverlapSize() == overlap, "Overlap size mismatch");
    ASSERT(buffer.getHopSize() == frame - overlap, "Hop size mismatch");
    ASSERT(buffer.getAvailableSamplesRead() == 0, "Initial samples available mismatch");
    ASSERT(buffer.getAvailableFramesRead() == 0, "Initial frames available mismatch");
}

void TestZeroOverlapFIFO() {
    print_header("TestZeroOverlapFIFO");
    // With Overlap = 0, this should behave like a standard FIFO ring buffer
    // Write 100 -> Read 100 -> Exact match expected.
    
    size_t channels = 1;
    size_t capacity = 1024;
    size_t frame = 128;
    size_t overlap = 0;

    JABuff::OLARingBuffer2D<float> buffer(channels, capacity, frame, overlap);

    auto input = create_ramp<float>(channels, 128, 0.0f); // 0..127
    
    // Write
    bool w_success = buffer.write(input);
    ASSERT(w_success, "Write failed");
    ASSERT(buffer.getAvailableSamplesRead() == 128, "Available samples mismatch (128 - 0 overlap)");
    ASSERT(buffer.getAvailableFramesRead() == 1, "Available frames mismatch");

    // Read
    std::vector<std::vector<float>> output;
    bool r_success = buffer.read(output, 1);
    
    ASSERT(r_success, "Read failed");
    ASSERT(output.size() == channels, "Output channel count mismatch");
    ASSERT(output[0].size() == 128, "Output sample count mismatch");
    
    // Verify content (Exact match expected for overlap 0)
    for(size_t i=0; i<128; ++i) {
        ASSERT_NEAR(output[0][i], static_cast<float>(i), 1e-5f, "Content verification failed");
    }

    ASSERT(buffer.getAvailableSamplesRead() == 0, "Buffer should be empty after full read");
}

void TestCrossfadeConstraints() {
    print_header("TestCrossfadeConstraints");
    // The write input MUST be >= overlap size
    size_t overlap = 10;
    JABuff::OLARingBuffer2D<float> buffer(1, 100, 20, overlap);

    // Try writing too small block
    auto small_input = create_block<float>(1, 5, 1.0f); // 5 < 10
    ASSERT(buffer.write(small_input) == false, "Write should fail for input smaller than overlap");

    // Try writing exact size block
    auto exact_input = create_block<float>(1, 10, 1.0f); // 10 == 10
    ASSERT(buffer.write(exact_input) == true, "Write should succeed for input equal to overlap");
    
    // Net advance should be 0 (Input - Overlap)
    ASSERT(buffer.getAvailableSamplesRead() == 0, "Available samples should remain 0");
}

void TestCrossfadeLogic() {
    print_header("TestCrossfadeLogic");
    size_t channels = 1;
    size_t capacity = 100;
    size_t frame = 20;
    size_t overlap = 10; 
    // Hop = 10

    JABuff::OLARingBuffer2D<float> buffer(channels, capacity, frame, overlap);

    // 1. Write Block A (Size 20, Value 1.0)
    // Advance = 20 - 10 = 10.
    // Buffer state (roughly): [0..9: FadeIn(1.0)], [10..19: FadeOut(1.0)]
    // Available: 10 samples (0..9)
    auto blockA = create_block<float>(channels, 20, 1.0f);
    buffer.write(blockA);
    ASSERT(buffer.getAvailableSamplesRead() == 10, "Available mismatch after block A");

    // 2. Write Block B (Size 20, Value 1.0)
    // This splices onto the tail of Block A.
    // The overlap region (10..19) gets: FadeOut(BlockA) + FadeIn(BlockB).
    // Available increases by 10 -> Total 20.
    auto blockB = create_block<float>(channels, 20, 1.0f);
    buffer.write(blockB);
    ASSERT(buffer.getAvailableSamplesRead() == 20, "Available mismatch after block B");

    // 3. Read 1 Frame (Size 20)
    // This covers indices 0..19.
    // 0..9 was the start of Block A.
    // 10..19 is the crossfade region (Tail A + Head B).
    std::vector<std::vector<float>> out;
    buffer.read(out, 1);

    ASSERT(out[0].size() == 20, "Read size mismatch");

    // Check Start (Pure Fade In of A)
    // Note: The very first write fades in from 0 (silence). 
    // So index 0 should be 0.0, index 5 approx 0.5 * curve, etc.
    ASSERT(out[0][0] == 0.0f, "Start of first block should be 0.0 (fade in from silence)");
    ASSERT(out[0][5] > 0.0f && out[0][5] < 1.0f, "Mid-fade-in value check");

    // Check Crossfade Region (indices 10..19)
    // This is where A fades out and B fades in.
    float mid_val = out[0][15];
    ASSERT(mid_val > 0.5f && mid_val < 1.5f, "Crossfade midpoint value check");
}

void TestVariableWritesAndWrapping() {
    print_header("TestVariableWritesAndWrapping");
    // Capacity 100.
    // We will write enough data to wrap around.
    size_t capacity = 100;
    size_t overlap = 5;
    size_t frame = 10;
    // Hop = 5
    JABuff::OLARingBuffer2D<float> buffer(1, capacity, frame, overlap);

    // Write 1: 55 samples. Advance = 50. Available = 50.
    // Writes indices [0..54].
    auto b1 = create_block<float>(1, 55, 1.0f);
    buffer.write(b1);

    // Write 2: 55 samples. Advance = 50. Available = 100. Full.
    // Wraps around. 
    auto b2 = create_block<float>(1, 55, 2.0f);
    buffer.write(b2);

    ASSERT(buffer.getAvailableSamplesRead() == 100, "Should be full (100 samples)");
    ASSERT(buffer.getAvailableSpaceWrite() == 0, "Space should be 0");

    // Read everything
    std::vector<std::vector<float>> out;
    buffer.read(out, 0); // Read all available frames
    // Available is 100. Frame size 10, Hop 5.
    // Num frames = 1 + (100 - 10)/5 = 1 + 18 = 19 frames.
    // Samples = (18 * 5) + 10 = 100.
    // Advance = 19 * 5 = 95.
    
    ASSERT(out[0].size() == 100, "Read all size mismatch");
    
    // Check that we extracted all frames
    ASSERT(buffer.getAvailableFramesRead() == 0, "Should have 0 frames remaining");
    
    // With overlap, samples remain because the read advances by hop size, not frame size.
    // Remaining = Total - Advance = 100 - 95 = 5.
    ASSERT(buffer.getAvailableSamplesRead() == overlap, "Remaining samples should match overlap");
    
    // Check that we can write again after reading
    // Advance write by 20-5=15. Available becomes 5+15=20.
    auto b3 = create_block<float>(1, 20, 3.0f);
    ASSERT(buffer.write(b3) == true, "Write after read failed");
    ASSERT(buffer.getAvailableSamplesRead() == 20, "Available samples check failed");
}

void TestPrimeSilence() {
    print_header("TestPrimeSilence");
    size_t overlap = 5;
    JABuff::OLARingBuffer2D<float> buffer(1, 100, 10, overlap);
    
    // Prime 2 frames (hops). Hop = 5.
    // Should advance available by 10.
    buffer.primeWithSilence(2);
    
    ASSERT(buffer.getAvailableSamplesRead() == 10, "Prime amount incorrect");
    
    std::vector<std::vector<float>> out;
    buffer.read(out, 0);
    
    // Should be all zeros
    for(float val : out[0]) {
        ASSERT_NEAR(val, 0.0f, 1e-6f, "Prime silence value check");
    }
}

int main() {
    TestInitialization();
    TestZeroOverlapFIFO();
    TestCrossfadeConstraints();
    TestCrossfadeLogic();
    TestVariableWritesAndWrapping();
    TestPrimeSilence();
    
    print_pass();
    return 0;
}

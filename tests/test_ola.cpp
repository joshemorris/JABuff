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
    ASSERT(buffer.getHopSize() == frame, "Hop size mismatch (should equal frame size)");
    
    ASSERT(buffer.getAvailableSamplesRead() == 0, "Initial samples available mismatch");
    ASSERT(buffer.getAvailableFramesRead() == 0, "Initial frames available mismatch");
}

void TestZeroOverlapFIFO() {
    print_header("TestZeroOverlapFIFO");
    // With Overlap = 0, this should behave like a standard FIFO ring buffer
    
    size_t channels = 1;
    size_t capacity = 1024;
    size_t frame = 128;
    size_t overlap = 0;

    JABuff::OLARingBuffer2D<float> buffer(channels, capacity, frame, overlap);

    // Write constraint is > 2 * overlap. 128 > 0 is True.
    auto input = create_ramp<float>(channels, 128, 0.0f); // 0..127
    
    // Write
    bool w_success = buffer.write(input);
    ASSERT(w_success, "Write failed");
    
    // Net advance = 128 - 0 = 128
    ASSERT(buffer.getAvailableSamplesRead() == 128, "Available samples mismatch");
    ASSERT(buffer.getAvailableFramesRead() == 1, "Available frames mismatch");

    // Read
    std::vector<std::vector<float>> output;
    bool r_success = buffer.read(output, 1);
    
    ASSERT(r_success, "Read failed");
    ASSERT(output.size() == channels, "Output channel count mismatch");
    ASSERT(output[0].size() == 128, "Output sample count mismatch");
    
    // Verify content
    for(size_t i=0; i<128; ++i) {
        ASSERT_NEAR(output[0][i], static_cast<float>(i), 1e-5f, "Content verification failed");
    }

    ASSERT(buffer.getAvailableSamplesRead() == 0, "Buffer should be empty after full read");
}

void TestCrossfadeConstraints() {
    print_header("TestCrossfadeConstraints");
    // The write input MUST be > 2 * overlap size
    size_t overlap = 10;
    JABuff::OLARingBuffer2D<float> buffer(1, 100, 20, overlap);

    // Threshold is 2 * 10 = 20. Input must be > 20.

    // 1. Try writing <= threshold
    auto small_input = create_block<float>(1, 20, 1.0f); // 20 is not > 20
    ASSERT(buffer.write(small_input) == false, "Write should fail for input <= 2*overlap");

    // 2. Try writing > threshold
    auto valid_input = create_block<float>(1, 21, 1.0f); // 21 > 20
    ASSERT(buffer.write(valid_input) == true, "Write should succeed for input > 2*overlap");
    
    // Net advance should be 21 - 10 = 11.
    ASSERT(buffer.getAvailableSamplesRead() == 11, "Available samples should match net advance");
}

void TestCrossfadeLogic() {
    print_header("TestCrossfadeLogic");
    size_t channels = 1;
    size_t capacity = 100;
    size_t frame = 20;
    size_t overlap = 10; 
    // Hop = Frame = 20
    // Write constraint > 20.

    JABuff::OLARingBuffer2D<float> buffer(channels, capacity, frame, overlap);

    // 1. Write Block A
    // Use size 30 (> 20). Net advance = 30 - 10 = 20.
    // Layout: [0..9 FadeIn], [10..19 Body], [20..29 FadeOut]
    // Available: 20 samples (Indices 0..19).
    auto blockA = create_block<float>(channels, 30, 1.0f);
    buffer.write(blockA);
    ASSERT(buffer.getAvailableSamplesRead() == 20, "Available mismatch after block A");

    // 2. Write Block B
    // Size 30. Net advance = 20. Total Available = 40.
    // Splices at index 20 (where Block A's FadeOut started).
    // Layout at splice: [20..29] = FadeOut(A) + FadeIn(B).
    auto blockB = create_block<float>(channels, 30, 1.0f);
    buffer.write(blockB);
    ASSERT(buffer.getAvailableSamplesRead() == 40, "Available mismatch after block B");

    // 3. Read Frame 1
    // Reads indices 0..19 (Size 20).
    // This is strictly Block A (FadeIn + Body). No crossfade here (splice is at 20).
    std::vector<std::vector<float>> out1;
    buffer.read(out1, 1);
    ASSERT(out1[0].size() == 20, "Read 1 size mismatch");
    ASSERT(out1[0][0] == 0.0f, "Start of first block should be 0.0 (fade in)");
    ASSERT(out1[0][10] == 1.0f, "Body of Block A should be 1.0");

    // 4. Read Frame 2
    // Reads indices 20..39.
    // Indices 20..29 are the Crossfade Region.
    // Indices 30..39 are Body of B.
    std::vector<std::vector<float>> out2;
    buffer.read(out2, 1);
    ASSERT(out2[0].size() == 20, "Read 2 size mismatch");

    // Check Crossfade Region (indices 0..9 of this frame)
    float mid_val = out2[0][5]; // Middle of crossfade
    ASSERT(mid_val > 0.5f && mid_val < 1.5f, "Crossfade midpoint value check");
    
    // Check Body of B (indices 10..19 of this frame)
    ASSERT(out2[0][15] == 1.0f, "Body of Block B should be 1.0");
}

void TestVariableWritesAndWrapping() {
    print_header("TestVariableWritesAndWrapping");
    // Capacity 100.
    size_t capacity = 100;
    size_t overlap = 5;
    size_t frame = 10; 
    // New: Hop = 10.
    // Constraint: Input > 10.

    JABuff::OLARingBuffer2D<float> buffer(1, capacity, frame, overlap);

    // Write 1: 55 samples. Advance = 55 - 5 = 50. Available = 50.
    auto b1 = create_block<float>(1, 55, 1.0f);
    buffer.write(b1);

    // Write 2: 55 samples. Advance = 50. Available = 100. Full.
    auto b2 = create_block<float>(1, 55, 2.0f);
    buffer.write(b2);

    ASSERT(buffer.getAvailableSamplesRead() == 100, "Should be full (100 samples)");
    ASSERT(buffer.getAvailableSpaceWrite() == 0, "Space should be 0");

    // Read everything
    std::vector<std::vector<float>> out;
    buffer.read(out, 0); // Read all available frames
    
    // Available 100. Frame 10. Hop 10.
    // Num frames = 100 / 10 = 10 frames.
    // Total samples read = 10 * 10 = 100.
    
    ASSERT(out[0].size() == 100, "Read all size mismatch");
    
    // Check that we extracted all frames
    ASSERT(buffer.getAvailableFramesRead() == 0, "Should have 0 frames remaining");
    
    // With Hop == Frame, reading all frames drains samples exactly (if divisible).
    // Remaining = 100 - 100 = 0.
    ASSERT(buffer.getAvailableSamplesRead() == 0, "Remaining samples should be 0");
    
    // Check that we can write again after reading
    // Write 25 (> 10). Advance 20. Available 20.
    auto b3 = create_block<float>(1, 25, 3.0f);
    ASSERT(buffer.write(b3) == true, "Write after read failed");
    ASSERT(buffer.getAvailableSamplesRead() == 20, "Available samples check failed");
}

void TestPrimeSilence() {
    print_header("TestPrimeSilence");
    size_t overlap = 5;
    size_t frame = 10; 
    JABuff::OLARingBuffer2D<float> buffer(1, 100, frame, overlap);
    
    // 1. Write a block of 1s.
    // Input 20. Advance 15. Available 15.
    // The tail (size 5) contains the Fade-Out of the 1s block.
    // If we wrote another 1s block now, it would maintain volume approx 1.0.
    auto b1 = create_block<float>(1, 20, 1.0f);
    buffer.write(b1);
    
    // 2. Prime with Silence.
    // This should clear the tail.
    // Available should NOT change.
    buffer.primeWithSilence();
    
    ASSERT(buffer.getAvailableSamplesRead() == 15, "Prime should not change available count");
    
    // 3. Write another block of 1s.
    // Input 20. Advance 15. Total Available 30.
    // The splice happens at the previous tail location.
    // Since we primed, the 'Previous Tail' is now 0.0.
    // So Splice = 0.0 + FadeIn(NewBlock).
    // This means the signal should drop to 0 and fade up, instead of staying at 1.0.
    buffer.write(b1);
    
    // 4. Read the splice region.
    std::vector<std::vector<float>> out;
    // We want to read 30 samples.
    buffer.read(out, 3);
    
    // Frame 1 (0..9): First block (FadeIn + Body).
    // Frame 2 (10..19):
    //   Indices 10..14: Body of first block (1.0).
    //   Indices 15..19: SPLICE REGION.
    //   Since we primed, Splice = 0 + FadeIn.
    //   FadeIn starts at 0.
    //   So index 15 should be near 0.
    
    float splice_val = out[0][15];
    
    // If we hadn't primed, splice would be FadeOut(1) + FadeIn(1) ~= 1.
    // Since we primed, it is 0 + FadeIn(1).
    // FadeIn(0) is 0. FadeIn(0.2) is small.
    // So splice_val should be small (< 0.5).
    ASSERT(splice_val < 0.5f, "Splice should be silence+fadein (small), not crossfade (near 1)");
    
    // Ensure it's not identically zero (it is fading in)
    // Actually, index 15 corresponds to window index 0.
    // window[0] is usually 0.
    // So it might be exactly 0.
    ASSERT(splice_val >= 0.0f, "Splice val positive");
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

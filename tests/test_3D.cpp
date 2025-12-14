#include "JABuff/FramingRingBuffer3D.hpp"
#include "test_utils.hpp"
#include <numeric>

void TestBasic3D() {
    print_header("TestBasic3D");
    size_t num_channels = 2;
    size_t feature_dim = 4;
    size_t capacity = 100;
    size_t frame_size = 10;
    size_t hop_size = 5;

    JABuff::FramingRingBuffer3D<float> buffer(num_channels, feature_dim, capacity, frame_size, hop_size);

    ASSERT(buffer.getFeatureDim() == 4, "Feature dim mismatch");

    // Write 8 steps
    std::vector<std::vector<std::vector<float>>> input(
        num_channels, std::vector<std::vector<float>>(8, std::vector<float>(feature_dim, 1.0f))
    );
    buffer.write(input);
    ASSERT(buffer.getAvailableTimeRead() == 8, "Available time mismatch (1)");
    ASSERT(buffer.getAvailableFramesRead() == 0, "Frame count mismatch");

    // Write 8 more -> Total 16
    buffer.write(input);
    ASSERT(buffer.getAvailableTimeRead() == 16, "Available time mismatch (2)");
    ASSERT(buffer.getAvailableFramesRead() == 2, "Frame count mismatch (should be 2)"); // (16-10)/5 + 1 = 2 (floored? no integer math)
    // 16 available. Frame 10. Hop 5.
    // Frame 1: 0..9.
    // Frame 2: 5..14.
    // Frame 3 needs 10..19. (Not available).
    // Correct.

    std::vector<std::vector<std::vector<float>>> out;
    buffer.read(out);
    
    // Check dimensions [Channel][Time][Feature]
    ASSERT(out.size() == 2, "Out Channel dim");
    ASSERT(out[0].size() == 10, "Out Time dim"); // 1 frame
    ASSERT(out[0][0].size() == 4, "Out Feature dim");
}

void TestOffsetWrite3D() {
    print_header("TestOffsetWrite3D");
    JABuff::FramingRingBuffer3D<float> buffer(1, 2, 100, 10, 5);
    
    std::vector<std::vector<std::vector<float>>> input(
        1, std::vector<std::vector<float>>(20, std::vector<float>(2))
    );
    
    for(int t=0; t<20; ++t) input[0][t][0] = (float)t;

    // Write 5..9
    buffer.write(input, 5, 5);
    // Write 15..19
    buffer.write(input, 15, 5);
    
    std::vector<std::vector<std::vector<float>>> out;
    ASSERT(buffer.read(out, 1), "Read failed");
    
    ASSERT_NEAR(out[0][0][0], 5.0f, 0.001f, "Start val mismatch");
    ASSERT_NEAR(out[0][4][0], 9.0f, 0.001f, "Mid val mismatch");
    ASSERT_NEAR(out[0][5][0], 15.0f, 0.001f, "Gap jump val mismatch");
}

void TestPush3D() {
    print_header("TestPush3D");
    JABuff::FramingRingBuffer3D<float> buffer(2, 2, 10, 5, 2);
    
    // [Channel][Feature]
    std::vector<std::vector<float>> step(2, std::vector<float>(2));
    
    for(int t=0; t<5; ++t) {
        step[0][0] = (float)t;
        step[1][1] = (float)t + 10.0f;
        buffer.push(step);
    }
    
    ASSERT(buffer.getAvailableTimeRead() == 5, "Push count mismatch");
    
    std::vector<std::vector<std::vector<float>>> out;
    buffer.read(out, 1);
    
    ASSERT_NEAR(out[0][4][0], 4.0f, 0.001f, "Val verification");
    ASSERT_NEAR(out[1][4][1], 14.0f, 0.001f, "Val verification ch2");
}

void TestReady3D() {
    print_header("TestReady3D");
    JABuff::FramingRingBuffer3D<float> buffer(1, 2, 100, 10, 5, 2); // Min frames 2
    
    std::vector<std::vector<std::vector<float>>> input(
        1, std::vector<std::vector<float>>(10, std::vector<float>(2, 1.0f))
    );

    // 0 frames
    ASSERT(!buffer.ready(), "Should not be ready (empty)");

    // 1 frame
    buffer.write(input);
    ASSERT(!buffer.ready(), "Should not be ready (1 frame < min 2)");

    // 2 frames
    buffer.write(input);
    ASSERT(buffer.ready(), "Should be ready (2 frames == min 2)");
}

void TestPrime3D() {
    print_header("TestPrime3D");

    // Case 1: Standard Latency Correction (MinFrames = 1)
    // Frame=10, Hop=5. Need 10 time steps total.
    // We want the NEXT Hop (5 steps) to trigger readiness.
    // Prime needs: 10 - 5 = 5 time steps.
    {
        JABuff::FramingRingBuffer3D<float> buffer(1, 2, 100, 10, 5, 1);
        
        // Prime with 0.5
        buffer.prime(0.5f);
        
        ASSERT(buffer.getAvailableTimeRead() == 5, "Prime amount incorrect (MinFrames=1)");
        ASSERT(!buffer.ready(), "Should not be ready yet");

        // Write 1 hop (5 steps)
        std::vector<std::vector<std::vector<float>>> input(
            1, std::vector<std::vector<float>>(5, std::vector<float>(2, 1.0f))
        );
        buffer.write(input);
        
        ASSERT(buffer.ready(), "Should be ready");
        
        std::vector<std::vector<std::vector<float>>> out;
        buffer.read(out);
        
        ASSERT(out[0].size() == 10, "Time dim");
        ASSERT_NEAR(out[0][0][0], 0.5f, 0.001f, "Prime val check");
        ASSERT_NEAR(out[0][5][0], 1.0f, 0.001f, "Input val check");
    }

    // Case 2: High MinFrames (MinFrames = 2)
    // Frame=10, Hop=5. Need 15 time steps total ((2-1)*5 + 10).
    // We want NEXT Hop (5 steps) to trigger readiness.
    // Prime needs: 15 - 5 = 10 steps.
    {
        JABuff::FramingRingBuffer3D<float> buffer(1, 2, 100, 10, 5, 2);
        buffer.prime(9.0f);
        
        ASSERT(buffer.getAvailableTimeRead() == 10, "Prime amount incorrect (MinFrames=2)");
        
        // Write 1 hop
         std::vector<std::vector<std::vector<float>>> input(
            1, std::vector<std::vector<float>>(5, std::vector<float>(2, 2.0f))
        );
        buffer.write(input);
        
        ASSERT(buffer.ready(), "Should be ready");
        ASSERT(buffer.getAvailableFramesRead() == 2, "Should have 2 frames");
    }
}

int main() {
    TestBasic3D();
    TestOffsetWrite3D();
    TestPush3D();
    TestReady3D();
    TestPrime3D();
    print_pass();
    return 0;
}

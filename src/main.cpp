#include <iostream>
#include <vector>
#include <numeric> // For std::iota
#include <cmath>   // For std::sin

#include "JABuff/FramingRingBuffer2D.hpp"
#include "JABuff/FramingRingBuffer3D.hpp"

// Helper to print separator
void PrintHeader(const std::string& name) {
    std::cout << "\n========================================\n";
    std::cout << "  " << name << "\n";
    std::cout << "========================================" << std::endl;
}

/**
 * @brief Demonstrates using the 2D buffer for standard audio processing.
 * Scenario: 
 * - 2 Channels of Audio
 * - Input arrives in blocks of 256 samples (e.g., from sound card callback).
 * - We need overlapping analysis frames of 512 samples, hopping by 128 samples.
 */
void RunAudioExample() {
    PrintHeader("2D Audio Buffering Example");

    // 1. Setup Buffer Parameters
    size_t num_channels = 2;
    size_t capacity = 4096;      // Buffer capacity per channel
    size_t frame_size = 512;     // Window size for analysis (e.g., FFT size)
    size_t hop_size = 128;       // Hop size (75% overlap)

    JABuff::FramingRingBuffer2D<float> audioBuffer(num_channels, capacity, frame_size, hop_size);

    std::cout << "Created Buffer: " << num_channels << " Channels, " 
              << capacity << " Sample Capacity.\n";
    std::cout << "Framing: Size " << frame_size << ", Hop " << hop_size << "\n\n";

    // 2. Simulate Audio Callback (Incoming Data)
    // Let's simulate receiving 3 blocks of audio (3 * 256 = 768 samples total)
    size_t input_block_size = 256;
    
    for (int block = 0; block < 3; ++block) {
        // Generate synthetic audio data (sawtooth wave)
        std::vector<std::vector<float>> input_data(num_channels, std::vector<float>(input_block_size));
        float start_val = (float)(block * input_block_size);
        
        for (size_t c = 0; c < num_channels; ++c) {
            std::iota(input_data[c].begin(), input_data[c].end(), start_val + (c * 1000.0f)); 
        }

        std::cout << "[Input] Writing audio block " << block 
                  << " (" << input_block_size << " samples)..." << std::endl;

        // Write the block to the circular buffer
        if (!audioBuffer.write(input_data)) {
            std::cerr << "Buffer overflow!" << std::endl;
        }

        // 3. Process Available Frames
        // In a real app, this might happen in the same thread or a processing thread.
        std::vector<std::vector<float>> process_frame;
        
        // While we have enough data for a full frame, read it.
        int frames_processed = 0;
        while (audioBuffer.read(process_frame)) {
            // process_frame is now [Channel][Time] (size: 2 x 512)
            frames_processed++;
            
            // Example processing: Just print the first sample of the frame
            std::cout << "  -> [Process] Read Frame. Ch0 Start Sample: " 
                      << process_frame[0][0] << std::endl;
        }
        
        if (frames_processed == 0) {
            std::cout << "  -> [Buffering] Not enough data for a full frame yet." << std::endl;
        }
    }

    // 4. Demonstrate Single Sample Push
    std::cout << "\n[Input] Pushing single samples (e.g., loop based)..." << std::endl;
    std::vector<float> sample(num_channels);
    for(int i = 0; i < 128; ++i) {
        sample[0] = 0.5f; 
        sample[1] = 0.5f;
        audioBuffer.push(sample);
    }
    std::cout << "Available features after push: " << audioBuffer.getAvailableFeaturesRead() << std::endl;
}

/**
 * @brief Demonstrates using the 3D buffer for feature buffering (e.g., Spectrograms).
 * Scenario:
 * - 1 Channel of Data
 * - Each time step contains a feature vector of dimension 64 (e.g., Mel Bands).
 * - We want to buffer these time steps to feed a neural network a block of context.
 * - Context window: 10 time steps.
 */
void RunFeatureExample() {
    PrintHeader("3D Feature Buffering Example");

    size_t num_channels = 1;
    size_t feature_dim = 64;     // e.g., 64 Mel bands
    size_t capacity_time = 100;  // Store 100 time steps
    size_t context_window = 10;  // Neural Net requires 10 frames context
    size_t hop = 5;              // Advance 5 steps

    JABuff::FramingRingBuffer3D<float> featureBuffer(
        num_channels, feature_dim, capacity_time, context_window, hop
    );

    std::cout << "Created 3D Buffer: Dim " << feature_dim 
              << ", Context Window " << context_window << "\n\n";

    // 1. Simulate incoming feature vectors (e.g., computed from FFT)
    // Let's write 15 time steps.
    size_t time_steps_to_write = 15;
    
    // Structure: [Channel][Time][Feature]
    std::vector<std::vector<std::vector<float>>> input_features(
        num_channels, 
        std::vector<std::vector<float>>(time_steps_to_write, std::vector<float>(feature_dim, 0.0f))
    );

    // Fill with dummy data
    for(size_t t=0; t<time_steps_to_write; ++t) {
        input_features[0][t][0] = (float)t; // Set 0th feature to time index for visualization
    }

    std::cout << "[Input] Writing " << time_steps_to_write << " feature time steps..." << std::endl;
    featureBuffer.write(input_features);

    // 2. Read Batches
    std::vector<std::vector<std::vector<float>>> batch_out;
    
    while(featureBuffer.read(batch_out)) {
        std::cout << "  -> [Process] Read Context Window." << std::endl;
        std::cout << "     Shape: [" << batch_out.size() << "][" 
                  << batch_out[0].size() << "][" 
                  << batch_out[0][0].size() << "]" << std::endl;
        
        std::cout << "     Time Step 0, Feat 0: " << batch_out[0][0][0] << std::endl;
        std::cout << "     Time Step 9, Feat 0: " << batch_out[0][9][0] << std::endl;
    }
}

int main() {
    RunAudioExample();
    RunFeatureExample();
    return 0;
}

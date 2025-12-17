#pragma once

#include <vector>       // For std::vector
#include <stdexcept>    // For std::invalid_argument, std::out_of_range
#include <cstring>      // For std::memcpy, std::memset
#include <cstddef>      // For size_t
#include <string>       // For std::to_string
#include <algorithm>    // For std::min
#include <cmath>        // For std::sqrt

namespace JABuff {

/**
 * @brief A templated circular buffer for Crossfaded Concatenation (Write) and Framed Reading.
 *
 * Write Behavior:
 * - Accepts variable-sized blocks of audio.
 * - Splicing: Crossfades the overlap_size region of the new block with the tail of the previous block.
 * - Constraint: Input block size must be > 2 * overlap_size.
 * - Uses a "Cheap Energy-Preserving" crossfade curve.
 * * Read Behavior:
 * - Reads contiguous fixed-size frames (frame_size).
 * - Read Hop Size is equal to Frame Size (0% overlap on read).
 * - Only allows reading samples that have been fully resolved (passed the splice point).
 * - "Yet to be overlapped" tail samples are not available for reading.
 *
 * Reference for crossfade: https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
 *
 * @tparam T The data type to be stored (e.g., float, double).
 */
template <typename T>
class OLARingBuffer2D {
public:
    /**
     * @brief Construct a new OLA Ring Buffer.
     *
     * @param num_channels The number of channels.
     * @param capacity_samples The total capacity of the internal buffer per channel.
     * @param frame_size The size of the OUTPUT frames to be read.
     * @param overlap_size The size of the overlap used for WRITING (splicing).
     * * Note: Read operations will use frame_size as the hop size (contiguous frames).
     */
    OLARingBuffer2D(size_t num_channels, size_t capacity_samples, size_t frame_size, size_t overlap_size);

    /**
     * @brief Writes a variable-sized block of data to the buffer.
     * * This method performs a crossfade splice:
     * 1. It 'rewinds' to the end of the previous valid data.
     * 2. It sums the 'Fade In' of the new data with the 'Fade Out' of the previous data (the tail).
     * 3. It overwrites the subsequent buffer area with the body and new 'Fade Out' tail of this data.
     *
     * @param data_in Input data [channel][sample]. Size must be > 2 * overlap_size.
     * @return true if write succeeded.
     * @return false if buffer full or input too small.
     */
    bool write(const std::vector<std::vector<T>>& data_in);

    /**
     * @brief Reads contiguous frames of audio from the buffer.
     * * Reads 'num_frames' of size 'frame_size'.
     * Advances the read head by 'frame_size' for each frame.
     * Does NOT clear the buffer.
     *
     * @param buffer_out Output vector [channel][sample]. Resized automatically.
     * @param num_frames The number of frames to read. 0 = Read all available.
     * @return true if the frames were successfully read.
     * @return false if there were not enough available frames.
     */
    bool read(std::vector<std::vector<T>>& buffer_out, size_t num_frames = 1);

    /**
     * @brief Primes the buffer's tail with silence.
     * * This Zeros out the 'overlap_size' samples at the current write index.
     * * It ensures the NEXT write will crossfade with silence (Fade In from 0)
     * * rather than crossfading with whatever garbage or previous data was there.
     * * It does NOT advance the write index or available samples.
     */
    void primeWithSilence();

    /**
     * @brief Resets read/write pointers and clears the buffer memory.
     */
    void clear();

    size_t getAvailableFramesRead() const;
    size_t getAvailableSamplesRead() const;
    size_t getAvailableSpaceWrite() const; // In terms of samples
    size_t getNumChannels() const;
    size_t getFrameSize() const;
    size_t getOverlapSize() const;
    size_t getHopSize() const;
    size_t getCapacity() const;

private:
    // --- Helpers ---
    void precomputeWindow();
    
    // The "Cheap Energy-Preserving" curve function
    T crossfadeCurve(T x) const;

    // --- Member Variables ---
    std::vector<std::vector<T>> m_buffer; 
    std::vector<T> m_crossfade_window; // Size = overlap_size

    size_t m_num_channels;
    size_t m_capacity_samples;
    size_t m_frame_size;    // For Reading
    size_t m_overlap_size;  // For Writing (Splice Size)
    size_t m_hop_size;      // Equal to m_frame_size (Contiguous reading)

    size_t m_write_index;   // Points to the start of the current "Overlap Region" (where we Add)
    size_t m_read_index;    // Points to the next sample to be read
    size_t m_available_samples; // Samples safely fully written and ready to read
};

// ===================================================================
// --- Implementation ---
// ===================================================================

template <typename T>
OLARingBuffer2D<T>::OLARingBuffer2D(size_t num_channels, size_t capacity_samples, size_t frame_size, size_t overlap_size)
    : m_num_channels(num_channels),
      m_capacity_samples(capacity_samples),
      m_frame_size(frame_size),
      m_overlap_size(overlap_size),
      m_write_index(0),
      m_read_index(0),
      m_available_samples(0) {

    if (num_channels == 0 || capacity_samples == 0) {
        throw std::invalid_argument("Channels and capacity must be non-zero.");
    }
    if (m_frame_size > m_capacity_samples) {
        throw std::invalid_argument("Frame size cannot be larger than capacity.");
    }
    // Note: Overlap size is now independent of frame size (only affects writing).
    // Read hop size is implicitly the frame size (contiguous reading).
    m_hop_size = m_frame_size;

    // Allocate buffer
    m_buffer.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        m_buffer[c].resize(m_capacity_samples, static_cast<T>(0));
    }

    precomputeWindow();
}

template <typename T>
T OLARingBuffer2D<T>::crossfadeCurve(T x) const {
    // Signalsmith "Cheap Energy-Preserving" Crossfade
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;

    const T k = static_cast<T>(1.4186);
    T v = x * (static_cast<T>(1.0) - x);
    T term = v * (static_cast<T>(1.0) + k * v) + x;
    return term * term; // Returns Amplitude Gain
}

template <typename T>
void OLARingBuffer2D<T>::precomputeWindow() {
    m_crossfade_window.resize(m_overlap_size);
    for (size_t i = 0; i < m_overlap_size; ++i) {
        T t = static_cast<T>(i) / static_cast<T>(m_overlap_size);
        m_crossfade_window[i] = crossfadeCurve(t);
    }
}

template <typename T>
bool OLARingBuffer2D<T>::write(const std::vector<std::vector<T>>& data_in) {
    if (data_in.empty()) return true;

    // 1. Validate Dimensions
    if (data_in.size() != m_num_channels) {
        throw std::invalid_argument("Input channel count mismatch.");
    }

    size_t input_len = data_in[0].size();
    
    // Constraint: Writing can be any length longer than two times the overlap size.
    // This ensures distinct regions: Overlap In (Fade In) -> Body -> Overlap Out (Fade Out)
    if (input_len <= 2 * m_overlap_size) {
        return false;
    }

    // 2. Check Capacity
    // We effectively advance the buffer by (input_len - overlap_size).
    // The overlap region is "rewritten/summed", but the net growth is input - overlap.
    size_t net_advance = input_len - m_overlap_size;
    
    if (m_available_samples + net_advance > m_capacity_samples) {
        return false;
    }

    // 3. Perform Crossfaded Splice
    for (size_t c = 0; c < m_num_channels; ++c) {
        const T* input_ptr = data_in[c].data();
        T* buffer_ptr = m_buffer[c].data();

        // Part A: Overlap Region (Add Fade-In to existing buffer content)
        // Existing buffer content at m_write_index is assumed to be the Fade-Out of the previous block.
        for (size_t i = 0; i < m_overlap_size; ++i) {
            size_t idx = (m_write_index + i) % m_capacity_samples;
            
            // Apply Fade-In Window to Input
            T input_sample = input_ptr[i] * m_crossfade_window[i];
            
            // Add to existing (Overlap Add)
            buffer_ptr[idx] += input_sample;
        }

        // Part B: Body and New Tail (Overwrite)
        // We write the rest of the data. The end of this data becomes the new Fade-Out tail.
        size_t remaining_len = input_len - m_overlap_size;
        size_t body_start_offset = m_overlap_size;
        
        // Where to start writing the body in the buffer?
        // It starts immediately after the overlap region.
        size_t buffer_body_start_idx = (m_write_index + m_overlap_size) % m_capacity_samples;

        for (size_t i = 0; i < remaining_len; ++i) {
            size_t idx = (buffer_body_start_idx + i) % m_capacity_samples;
            size_t input_idx = body_start_offset + i;
            
            T sample = input_ptr[input_idx];

            // If this sample is part of the NEW tail (last overlap_size samples), fade it out.
            // Distance from end of input block:
            size_t samples_from_end = input_len - 1 - input_idx;
            
            if (samples_from_end < m_overlap_size) {
                // Apply Fade-Out (Reverse Window)
                // Window index 0 is silence (start of fade in).
                // Window index overlap-1 is full vol (end of fade in).
                size_t window_idx = samples_from_end; 
                if (window_idx >= m_overlap_size) window_idx = m_overlap_size - 1;
                
                sample *= m_crossfade_window[window_idx];
            }

            // Overwrite garbage/old history with new data
            buffer_ptr[idx] = sample;
        }
    }

    // 4. Update Indices
    // The next write should start adding at the beginning of the NEW tail.
    // The new tail starts at: current_write + input_len - overlap.
    m_write_index = (m_write_index + net_advance) % m_capacity_samples;
    
    // We can now safely read the data up to the start of the new tail.
    // The tail itself is "incomplete" (yet to be overlapped) and not counted.
    m_available_samples += net_advance;

    return true;
}

template <typename T>
bool OLARingBuffer2D<T>::read(std::vector<std::vector<T>>& buffer_out, size_t num_frames) {
    size_t available_frames = getAvailableFramesRead();

    // Check availability
    size_t count_to_read = num_frames;
    if (num_frames == 0) count_to_read = available_frames;

    if (count_to_read == 0 || available_frames < count_to_read) {
        return false;
    }

    // Output setup
    // Output size per channel = count_to_read * frame_size (Contiguous)
    size_t total_samples = count_to_read * m_frame_size;
    
    buffer_out.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        buffer_out[c].resize(total_samples);
        T* dest = buffer_out[c].data();
        const T* src = m_buffer[c].data();

        // Contiguous copy logic
        size_t space_to_end = m_capacity_samples - m_read_index;
        if (total_samples > space_to_end) {
            std::memcpy(dest, src + m_read_index, space_to_end * sizeof(T));
            std::memcpy(dest + space_to_end, src, (total_samples - space_to_end) * sizeof(T));
        } else {
            std::memcpy(dest, src + m_read_index, total_samples * sizeof(T));
        }
    }

    // Advance Read Head
    // We advance by frames (hop = frame).
    size_t advance = count_to_read * m_hop_size;
    m_read_index = (m_read_index + advance) % m_capacity_samples;
    m_available_samples -= advance;

    return true;
}

template <typename T>
void OLARingBuffer2D<T>::primeWithSilence() {
    // We strictly want to clear the 'tail' (overlap region) at the current write head.
    // This ensures the NEXT write sums with silence (0.0) instead of existing data.
    // We do NOT advance indices, because this region is 'waiting' to be overlapped.
    
    for (size_t c = 0; c < m_num_channels; ++c) {
        for (size_t i = 0; i < m_overlap_size; ++i) {
            size_t idx = (m_write_index + i) % m_capacity_samples;
            m_buffer[c][idx] = static_cast<T>(0);
        }
    }
}

template <typename T>
void OLARingBuffer2D<T>::clear() {
    m_write_index = 0;
    m_read_index = 0;
    m_available_samples = 0;
    
    for (auto& ch_buffer : m_buffer) {
        std::fill(ch_buffer.begin(), ch_buffer.end(), static_cast<T>(0));
    }
}

template <typename T>
size_t OLARingBuffer2D<T>::getAvailableFramesRead() const {
    if (m_available_samples < m_frame_size) return 0;
    // Simple division since we read contiguous blocks
    return m_available_samples / m_frame_size;
}

template <typename T>
size_t OLARingBuffer2D<T>::getAvailableSamplesRead() const {
    return m_available_samples;
}

template <typename T>
size_t OLARingBuffer2D<T>::getAvailableSpaceWrite() const {
    return m_capacity_samples - m_available_samples;
}

template <typename T>
size_t OLARingBuffer2D<T>::getNumChannels() const { return m_num_channels; }

template <typename T>
size_t OLARingBuffer2D<T>::getFrameSize() const { return m_frame_size; }

template <typename T>
size_t OLARingBuffer2D<T>::getOverlapSize() const { return m_overlap_size; }

template <typename T>
size_t OLARingBuffer2D<T>::getHopSize() const { return m_hop_size; }

template <typename T>
size_t OLARingBuffer2D<T>::getCapacity() const { return m_capacity_samples; }

} // namespace JABuff

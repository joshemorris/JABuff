#pragma once

#include <vector>       // For std::vector
#include <stdexcept>    // For std::invalid_argument, std::out_of_range
#include <cstring>      // For std::memcpy
#include <cstddef>      // For size_t
#include <string>       // For std::to_string
#include <algorithm>    // For std::min

namespace JABuff {

/**
 * @brief A templated circular buffer for 2D data (e.g., Channels x Features).
 *
 * This class is designed for single-process access. It stores data in a
 * separate circular buffer for each channel.
 *
 * It allows writing blocks of features and reading overlapping frames.
 *
 * @tparam T The data type to be stored (e.g., float, double).
 */
template <typename T>
class FramingRingBuffer2D {
public:
    /**
     * @brief Construct a new 2D Framing Ring Buffer.
     *
     * @param num_channels The number of channels (dimension 1).
     * @param capacity_features The total number of features (e.g., samples)
     * the buffer can hold *per channel* (dimension 2).
     * @param frame_size_features The number of features to read per frame.
     * @param hop_size_features The number of features to advance after each read.
     * @param min_frames The minimum number of available frames required to perform a read.
     * Defaults to 1 (requires at least one full frame to be available).
     * @param keep_frames The number of frames to keep in the buffer after a read operation.
     * These frames will be available for the *next* read operation. Defaults to 0.
     */
    FramingRingBuffer2D(size_t num_channels, size_t capacity_features, size_t frame_size_features, size_t hop_size_features, size_t min_frames = 1, size_t keep_frames = 0);

    /**
     * @brief Writes a block of data to the buffer.
     * @param data_in Input data [channel][feature].
     * @param offset Start index in data_in.
     * @param num_to_write Number of features to write. 0 = auto-detect.
     * @return true if write succeeded, false if buffer full.
     * @throws std::invalid_argument if dimensions mismatch.
     * @throws std::out_of_range if offset/count exceeds input bounds.
     */
    bool write(const std::vector<std::vector<T>>& data_in, size_t offset = 0, size_t num_to_write = 0);

    /**
     * @brief Reads a contiguous block of data covering the requested frames.
     * * The output is organized as [channel][samples].
     * * Unlike previous versions, this does NOT duplicate overlapping samples.
     * It returns a single continuous vector representing the union of the requested frames.
     * * Total output size = (num_frames - 1) * hop_size + frame_size.
     * To access Frame 'i' from this buffer, read starting at index (i * hop_size).
     * * @param buffer_out Output vector [channel][samples]. Resized automatically.
     * @param num_frames The number of frames to read. 
     * If 0, reads ALL available frames.
     * If > 0, strictly requires that number of frames to be available.
     * @return true if the frames were successfully read.
     * @return false if there were not enough frames available (or < min_frames available).
     */
    bool read(std::vector<std::vector<T>>& buffer_out, size_t num_frames = 1);

    size_t getAvailableFramesRead() const;
    size_t getAvailableFeaturesRead() const;
    size_t getAvailableWrite() const;
    size_t getCapacity() const;
    size_t getNumChannels() const;
    size_t getFrameSizeFeatures() const;
    size_t getHopSizeFeatures() const;
    size_t getMinFrames() const;
    size_t getKeepFrames() const;
    bool isFull() const;
    bool isEmpty() const;
    void clear();

private:
    // --- Helpers ---
    void validateWriteInput(const std::vector<std::vector<T>>& data_in, size_t offset, size_t num_to_write, size_t& calculated_write_size) const;

    // --- Member Variables ---
    std::vector<std::vector<T>> m_buffer; 
    size_t m_num_channels;
    size_t m_capacity_features;
    size_t m_frame_size_features;
    size_t m_hop_size_features;
    size_t m_min_frames;
    size_t m_keep_frames;
    size_t m_write_index_features;
    size_t m_read_index_features;
    size_t m_available_features;
};

// ===================================================================
// --- Implementation ---
// ===================================================================

template <typename T>
FramingRingBuffer2D<T>::FramingRingBuffer2D(size_t num_channels, size_t capacity_features, size_t frame_size_features, size_t hop_size_features, size_t min_frames, size_t keep_frames)
    : m_num_channels(num_channels),
      m_capacity_features(capacity_features),
      m_frame_size_features(frame_size_features),
      m_hop_size_features(hop_size_features),
      m_min_frames(min_frames),
      m_keep_frames(keep_frames),
      m_write_index_features(0),
      m_read_index_features(0),
      m_available_features(0) {

    if (num_channels == 0 || capacity_features == 0) {
        throw std::invalid_argument("Channels and capacity must be non-zero.");
    }
    if (m_frame_size_features > m_capacity_features) {
        throw std::invalid_argument("Frame size cannot be larger than capacity.");
    }
     if (m_hop_size_features == 0) {
        throw std::invalid_argument("Hop size must be non-zero.");
    }

    m_buffer.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        m_buffer[c].resize(m_capacity_features);
    }
}

template <typename T>
void FramingRingBuffer2D<T>::validateWriteInput(const std::vector<std::vector<T>>& data_in, size_t offset, size_t num_to_write, size_t& calculated_write_size) const {
    if (data_in.size() != m_num_channels) {
        throw std::invalid_argument("Input data channel count (" + std::to_string(data_in.size()) + 
                                    ") does not match buffer channels (" + std::to_string(m_num_channels) + ").");
    }

    if (data_in.empty()) return; // Should be caught by channel check usually, but for safety

    size_t input_size = data_in[0].size();

    // Verify all channels have the same size
    for(size_t c = 1; c < m_num_channels; ++c) {
        if(data_in[c].size() != input_size) {
            throw std::invalid_argument("Input channels have inconsistent sizes.");
        }
    }

    if (offset >= input_size && input_size > 0) {
        throw std::out_of_range("Write offset (" + std::to_string(offset) + ") exceeds input vector size (" + std::to_string(input_size) + ").");
    }

    // Logic for auto-size
    calculated_write_size = num_to_write;
    if (calculated_write_size == 0) {
        calculated_write_size = input_size - offset;
    }

    if (offset + calculated_write_size > input_size) {
        throw std::out_of_range("Write request (Offset: " + std::to_string(offset) + 
                                ", Count: " + std::to_string(calculated_write_size) + 
                                ") exceeds input vector bounds (" + std::to_string(input_size) + ").");
    }
}

template <typename T>
bool FramingRingBuffer2D<T>::write(const std::vector<std::vector<T>>& data_in, size_t offset, size_t num_to_write) {
    if (data_in.empty()) return true;

    // 1. Validate Input (Logic Errors -> Exception)
    size_t actual_write_size = 0;
    validateWriteInput(data_in, offset, num_to_write, actual_write_size);

    if (actual_write_size == 0) return true;

    // 2. Check Capacity (Runtime State -> Return False)
    if (actual_write_size > getAvailableWrite()) {
        return false; 
    }

    // 3. Perform Write
    for (size_t c = 0; c < m_num_channels; ++c) {
        const T* source_data = data_in[c].data() + offset;
        T* buffer_data = m_buffer[c].data();

        size_t write_pos = m_write_index_features;
        size_t space_to_end = m_capacity_features - write_pos;

        if (actual_write_size > space_to_end) {
            std::memcpy(buffer_data + write_pos, source_data, space_to_end * sizeof(T));
            std::memcpy(buffer_data, source_data + space_to_end, (actual_write_size - space_to_end) * sizeof(T));
        } else {
            std::memcpy(buffer_data + write_pos, source_data, actual_write_size * sizeof(T));
        }
    }

    m_write_index_features = (m_write_index_features + actual_write_size) % m_capacity_features;
    m_available_features += actual_write_size;

    return true;
}

template <typename T>
bool FramingRingBuffer2D<T>::read(std::vector<std::vector<T>>& buffer_out, size_t num_frames) {
    size_t available = getAvailableFramesRead();

    // Check minimum frames requirement
    if (available < m_min_frames) {
        return false;
    }

    size_t count_to_read = 0;

    // Logic for "Read All" vs "Read Specific Amount"
    if (num_frames == 0) {
        count_to_read = available;
    } else {
        if (available < num_frames) {
            // Strict check failed
            return false;
        }
        count_to_read = num_frames;
    }

    if (count_to_read == 0) {
        buffer_out.clear();
        return false;
    }

    // Calculate total continuous samples needed to cover these frames
    // Size = (N-1) * hop + frame_size
    // This creates a contiguous block of memory with NO duplicates.
    size_t total_samples_per_channel = (count_to_read - 1) * m_hop_size_features + m_frame_size_features;
    
    buffer_out.resize(m_num_channels);
    for(size_t c = 0; c < m_num_channels; ++c) {
        buffer_out[c].resize(total_samples_per_channel);
        
        T* dest_data = buffer_out[c].data();
        const T* buffer_data = m_buffer[c].data();

        size_t space_to_end = m_capacity_features - m_read_index_features;

        // Perform single continuous copy (with wrap check)
        if (total_samples_per_channel > space_to_end) {
            std::memcpy(dest_data, buffer_data + m_read_index_features, space_to_end * sizeof(T));
            std::memcpy(dest_data + space_to_end, buffer_data, (total_samples_per_channel - space_to_end) * sizeof(T));
        } else {
            std::memcpy(dest_data, buffer_data + m_read_index_features, total_samples_per_channel * sizeof(T));
        }
    }

    // Update actual member variables based on consumption logic (hop size & keep frames)
    size_t frames_consumed = 0;
    if (count_to_read > m_keep_frames) {
        frames_consumed = count_to_read - m_keep_frames;
    }

    size_t features_consumed = frames_consumed * m_hop_size_features;

    m_read_index_features = (m_read_index_features + features_consumed) % m_capacity_features;
    m_available_features -= features_consumed;

    return true;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableFramesRead() const {
    if (m_available_features < m_frame_size_features) return 0;
    return 1 + (m_available_features - m_frame_size_features) / m_hop_size_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableFeaturesRead() const { return m_available_features; }

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableWrite() const { return m_capacity_features - m_available_features; }

template <typename T>
size_t FramingRingBuffer2D<T>::getCapacity() const { return m_capacity_features; }

template <typename T>
size_t FramingRingBuffer2D<T>::getNumChannels() const { return m_num_channels; }

template <typename T>
size_t FramingRingBuffer2D<T>::getFrameSizeFeatures() const { return m_frame_size_features; }

template <typename T>
size_t FramingRingBuffer2D<T>::getHopSizeFeatures() const { return m_hop_size_features; }

template <typename T>
size_t FramingRingBuffer2D<T>::getMinFrames() const { return m_min_frames; }

template <typename T>
size_t FramingRingBuffer2D<T>::getKeepFrames() const { return m_keep_frames; }

template <typename T>
bool FramingRingBuffer2D<T>::isFull() const { return getAvailableWrite() == 0; }

template <typename T>
bool FramingRingBuffer2D<T>::isEmpty() const { return getAvailableFeaturesRead() == 0; }

template <typename T>
void FramingRingBuffer2D<T>::clear() {
    m_write_index_features = 0;
    m_read_index_features = 0;
    m_available_features = 0;
}

} // namespace JABuff

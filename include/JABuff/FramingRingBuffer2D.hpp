#pragma once

#include <vector>       // For std::vector
#include <stdexcept>    // For std::invalid_argument, std::out_of_range
#include <cstring>      // For std::memcpy
#include <cstddef>      // For size_t
#include <string>       // For std::to_string

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
     */
    FramingRingBuffer2D(size_t num_channels, size_t capacity_features, size_t frame_size_features, size_t hop_size_features);

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
     * @brief Reads a frame of data.
     * @return true if read succeeded, false if insufficient data.
     * @throws std::invalid_argument if frame_out dimensions are incorrect.
     */
    bool read(std::vector<std::vector<T>>& frame_out);

    size_t getAvailableFramesRead() const;
    size_t getAvailableFeaturesRead() const;
    size_t getAvailableWrite() const;
    size_t getCapacity() const;
    size_t getNumChannels() const;
    size_t getFrameSizeFeatures() const;
    size_t getHopSizeFeatures() const;
    bool isFull() const;
    bool isEmpty() const;
    void clear();

private:
    // --- Helpers ---
    void validateWriteInput(const std::vector<std::vector<T>>& data_in, size_t offset, size_t num_to_write, size_t& calculated_write_size) const;
    void validateReadOutput(const std::vector<std::vector<T>>& frame_out) const;

    // --- Member Variables ---
    std::vector<std::vector<T>> m_buffer; 
    size_t m_num_channels;
    size_t m_capacity_features;
    size_t m_frame_size_features;
    size_t m_hop_size_features;
    size_t m_write_index_features;
    size_t m_read_index_features;
    size_t m_available_features;
};

// ===================================================================
// --- Implementation ---
// ===================================================================

template <typename T>
FramingRingBuffer2D<T>::FramingRingBuffer2D(size_t num_channels, size_t capacity_features, size_t frame_size_features, size_t hop_size_features)
    : m_num_channels(num_channels),
      m_capacity_features(capacity_features),
      m_frame_size_features(frame_size_features),
      m_hop_size_features(hop_size_features),
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
void FramingRingBuffer2D<T>::validateReadOutput(const std::vector<std::vector<T>>& frame_out) const {
    if (frame_out.size() != m_num_channels) {
        throw std::invalid_argument("Output frame channel count (" + std::to_string(frame_out.size()) + 
                                    ") does not match buffer channels (" + std::to_string(m_num_channels) + ").");
    }
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (frame_out[c].size() != m_frame_size_features) {
            throw std::invalid_argument("Output frame size (" + std::to_string(frame_out[c].size()) + 
                                        ") does not match buffer frame size (" + std::to_string(m_frame_size_features) + ").");
        }
    }
}

template <typename T>
bool FramingRingBuffer2D<T>::read(std::vector<std::vector<T>>& frame_out) {
    // 1. Check Availability (Runtime State -> Return False)
    if (m_frame_size_features > m_available_features) return false;
    if (m_hop_size_features > m_available_features) return false;

    // 2. Validate Output Container (Logic Errors -> Exception)
    validateReadOutput(frame_out);

    // 3. Perform Read
    for (size_t c = 0; c < m_num_channels; ++c) {
        T* dest_data = frame_out[c].data();
        const T* buffer_data = m_buffer[c].data();

        size_t read_pos = m_read_index_features;
        size_t space_to_end = m_capacity_features - read_pos;

        if (m_frame_size_features > space_to_end) {
            std::memcpy(dest_data, buffer_data + read_pos, space_to_end * sizeof(T));
            std::memcpy(dest_data + space_to_end, buffer_data, (m_frame_size_features - space_to_end) * sizeof(T));
        } else {
            std::memcpy(dest_data, buffer_data + read_pos, m_frame_size_features * sizeof(T));
        }
    }

    m_read_index_features = (m_read_index_features + m_hop_size_features) % m_capacity_features;
    m_available_features -= m_hop_size_features;

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

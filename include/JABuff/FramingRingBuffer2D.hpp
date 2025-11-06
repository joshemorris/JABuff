#pragma once

#include <vector>       // For std::vector
#include <stdexcept>    // For std::stdexcept
#include <cstring>      // For std::memcpy
#include <cstddef>      // For size_t

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
    // ===================================================================
    // --- Class Declaration ---
    // ===================================================================

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
     * This is a non-blocking call.
     * @param data_in A vector of vectors [channel][feature] containing the
     * data to write. All channels must have the same number
     * of features.
     * @return true if the write was successful, false if the buffer was full
     * or if dimensions mismatch.
     */
    bool write(const std::vector<std::vector<T>>& data_in);

    /**
     * @brief Reads a frame of data from the buffer and consumes the hop size.
     * This is a non-blocking call.
     * @param frame_out The vector to read data into. Must be pre-sized
     * to [num_channels][frame_size_features].
     * @return true if the read was successful, false if the buffer had
     * insufficient data or if dimensions mismatch.
     */
    bool read(std::vector<std::vector<T>>& frame_out);

    /** @return The number of full frames available to be read. */
    size_t getAvailableFramesRead() const;

    /** @return The number of features currently available to be read. */
    size_t getAvailableFeaturesRead() const;

    /** @return The number of empty features available to be written. */
    size_t getAvailableWrite() const;

    /** @return The total feature capacity of the buffer *per channel*. */
    size_t getCapacity() const;

    /** @return The number of channels. */
    size_t getNumChannels() const;

    /** @return The frame size in features. */
    size_t getFrameSizeFeatures() const;

    /** @return The hop size in features. */
    size_t getHopSizeFeatures() const;

    /** @return true if the buffer is full (available write == 0). */
    bool isFull() const;

    /** @return true if the buffer is empty (available features read == 0). */
    bool isEmpty() const;

    /** @brief Clears the buffer, resetting all pointers. Not thread-safe. */
    void clear();

private:
    // --- Member Variables ---
    std::vector<std::vector<T>> m_buffer; // Storage: [Channel][Feature]

    size_t m_num_channels;
    size_t m_capacity_features;
    size_t m_frame_size_features;
    size_t m_hop_size_features;

    // Indices are in features
    size_t m_write_index_features;
    size_t m_read_index_features;
    size_t m_available_features;
};

// ===================================================================
// --- Class Implementation ---
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

    // Allocate the 2D buffer
    m_buffer.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        m_buffer[c].resize(m_capacity_features);
    }
}

template <typename T>
bool FramingRingBuffer2D<T>::write(const std::vector<std::vector<T>>& data_in) {
    if (data_in.empty() || data_in.size() != m_num_channels) {
        return false; // Channel mismatch
    }

    const size_t num_features_to_write = data_in[0].size();
    if (num_features_to_write == 0) {
        return true; // Nothing to write
    }

    if (num_features_to_write > getAvailableWrite()) {
        return false; // Not enough space
    }

    // Write data, looping per channel
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (data_in[c].size() != num_features_to_write) {
            return false; // All channels must have same feature count
        }

        const T* source_data = data_in[c].data();
        T* buffer_data = m_buffer[c].data();

        // Handle wrap-around
        size_t write_pos = m_write_index_features;
        size_t space_to_end = m_capacity_features - write_pos;

        if (num_features_to_write > space_to_end) {
            // First part: to the end of the buffer
            std::memcpy(buffer_data + write_pos, source_data, space_to_end * sizeof(T));
            // Second part: from the start of the buffer
            std::memcpy(buffer_data, source_data + space_to_end, (num_features_to_write - space_to_end) * sizeof(T));
        } else {
            // No wrap-around
            std::memcpy(buffer_data + write_pos, source_data, num_features_to_write * sizeof(T));
        }
    }

    m_write_index_features = (m_write_index_features + num_features_to_write) % m_capacity_features;
    m_available_features += num_features_to_write;

    return true;
}

template <typename T>
bool FramingRingBuffer2D<T>::read(std::vector<std::vector<T>>& frame_out) {
    if (m_frame_size_features > m_available_features) {
        return false; // Not enough data to read a full frame
    }
    
    if (m_hop_size_features > m_available_features) {
        // This case can happen if frame_size > hop_size, but we've
        // consumed data such that available < hop.
        return false;
    }

    // --- Validation ---
    // User must provide a pre-sized output vector
    if (frame_out.size() != m_num_channels) {
        return false; // Incorrect channel count
    }
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (frame_out[c].size() != m_frame_size_features) {
            return false; // Incorrect frame size
        }
    }
    // --- End Validation ---


    for (size_t c = 0; c < m_num_channels; ++c) {
        T* dest_data = frame_out[c].data();
        const T* buffer_data = m_buffer[c].data();

        // Handle wrap-around
        size_t read_pos = m_read_index_features;
        size_t space_to_end = m_capacity_features - read_pos;

        if (m_frame_size_features > space_to_end) {
            // First part: from read_pos to the end
            std::memcpy(dest_data, buffer_data + read_pos, space_to_end * sizeof(T));
            // Second part: from the start of the buffer
            std::memcpy(dest_data + space_to_end, buffer_data, (m_frame_size_features - space_to_end) * sizeof(T));
        } else {
            // No wrap-around
            std::memcpy(dest_data, buffer_data + read_pos, m_frame_size_features * sizeof(T));
        }
    }

    // Consume hop_size_features
    m_read_index_features = (m_read_index_features + m_hop_size_features) % m_capacity_features;
    m_available_features -= m_hop_size_features;

    return true;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableFramesRead() const {
    if (m_available_features < m_frame_size_features) {
        return 0;
    }
    // We have at least one frame.
    // Calculate how many *more* frames can be read based on the hop.
    return 1 + (m_available_features - m_frame_size_features) / m_hop_size_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableFeaturesRead() const {
    return m_available_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getAvailableWrite() const {
    return m_capacity_features - m_available_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getCapacity() const {
    return m_capacity_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getNumChannels() const {
    return m_num_channels;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getFrameSizeFeatures() const {
    return m_frame_size_features;
}

template <typename T>
size_t FramingRingBuffer2D<T>::getHopSizeFeatures() const {
    return m_hop_size_features;
}

template <typename T>
bool FramingRingBuffer2D<T>::isFull() const {
    return getAvailableWrite() == 0;
}

template <typename T>
bool FramingRingBuffer2D<T>::isEmpty() const {
    return getAvailableFeaturesRead() == 0;
}

template <typename T>
void FramingRingBuffer2D<T>::clear() {
    m_write_index_features = 0;
    m_read_index_features = 0;
    m_available_features = 0;
}

} // namespace JABuff

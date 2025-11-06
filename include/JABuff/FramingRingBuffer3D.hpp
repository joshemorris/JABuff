#pragma once

#include <vector>       // For std::vector
#include <stdexcept>    // For std::stdexcept
#include <cstring>      // For std::memcpy
#include <cstddef>      // For size_t

namespace JABuff {

/**
 * @brief A templated circular buffer for 3D data (e.g., Channels x Time x Features).
 *
 * This class is designed for single-threaded or externally synchronized
 * access. It stores data in a separate circular buffer for each channel.
 * The circular buffer behavior is along the 'Time' dimension.
 *
 * It allows writing blocks of 'time' steps and reading overlapping frames.
 *
 * @tparam T The data type to be stored (e.g., float, double).
 */
template <typename T>
class FramingRingBuffer3D {
public:
    // ===================================================================
    // --- Class Declaration ---
    // ===================================================================

    /**
     * @brief Construct a new 3D Framing Ring Buffer.
     *
     * @param num_channels The number of channels (dimension 1).
     * @param feature_dim The size of the feature dimension (dimension 3). This
     * is fixed for all time steps.
     * @param capacity_time The total number of time steps the buffer can hold
     * *per channel* (dimension 2).
     * @param frame_size_time The number of time steps to read per frame.
     * @param hop_size_time The number of time steps to advance after each read.
     */
    FramingRingBuffer3D(size_t num_channels, size_t feature_dim, size_t capacity_time, size_t frame_size_time, size_t hop_size_time);

    /**
     * @brief Writes a block of data to the buffer.
     * This is a non-blocking call.
     * @param data_in A vector of vectors [channel][time][feature] containing the
     * data to write. All channels must have the same number
     * of time steps, and all time steps must have 'feature_dim' features.
     * @return true if the write was successful, false if the buffer was full
     * or if dimensions mismatch.
     */
    bool write(const std::vector<std::vector<std::vector<T>>>& data_in);

    /**
     * @brief Reads a frame of data from the buffer and consumes the hop size.
     * This is a non-blocking call.
     * @param frame_data_out The vector to read data into. Must be pre-sized
     * to [num_channels][frame_size_time][feature_dim].
     * @return true if the read was successful, false if the buffer had
     * insufficient data or if dimensions mismatch.
     */
    bool read(std::vector<std::vector<std::vector<T>>>& frame_data_out);

    /** @return The number of full frames available to be read. */
    size_t getAvailableFramesRead() const;

    /** @return The number of time steps currently available to be read. */
    size_t getAvailableTimeRead() const;

    /** @return The number of empty time steps available to be written. */
    size_t getAvailableWrite() const;

    /** @return The total time step capacity of the buffer *per channel*. */
    size_t getCapacity() const;

    /** @return The number of channels. */
    size_t getNumChannels() const;

    /** @return The size of the feature dimension. */
    size_t getFeatureDim() const;

    /** @return The frame size in time steps. */
    size_t getFrameSizeTime() const;

    /** @return The hop size in time steps. */
    size_t getHopSizeTime() const;

    /** @return true if the buffer is full (available write == 0). */
    bool isFull() const;

    /** @return true if the buffer is empty (available time read == 0). */
    bool isEmpty() const;
    
    /** @brief Clears the buffer, resetting all pointers. Not thread-safe. */
    void clear();

private:
    // --- Member Variables ---
    // Storage: [Channel][Time][Feature]
    std::vector<std::vector<std::vector<T>>> m_buffers; 
    
    size_t m_num_channels;
    size_t m_feature_dim;
    size_t m_capacity_time;
    size_t m_frame_size_time;
    size_t m_hop_size_time;

    // Indices are in "time steps"
    size_t m_write_index_time;
    size_t m_read_index_time;
    size_t m_available_time;
};

// ===================================================================
// --- Class Implementation ---
// ===================================================================

template <typename T>
FramingRingBuffer3D<T>::FramingRingBuffer3D(size_t num_channels, size_t feature_dim, size_t capacity_time, size_t frame_size_time, size_t hop_size_time)
    : m_num_channels(num_channels),
      m_feature_dim(feature_dim),
      m_capacity_time(capacity_time),
      m_frame_size_time(frame_size_time),
      m_hop_size_time(hop_size_time),
      m_write_index_time(0),
      m_read_index_time(0),
      m_available_time(0) {
    
    if (num_channels == 0 || feature_dim == 0 || capacity_time == 0) {
        throw std::invalid_argument("Channels, feature dim, and capacity must be non-zero.");
    }
    if (m_frame_size_time > m_capacity_time) {
        throw std::invalid_argument("Frame size cannot be larger than capacity.");
    }
    if (m_hop_size_time == 0) {
        throw std::invalid_argument("Hop size must be non-zero.");
    }

    // Allocate the 3D buffer
    m_buffers.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        m_buffers[c].resize(m_capacity_time);
        for (size_t t = 0; t < m_capacity_time; ++t) {
            m_buffers[c][t].resize(m_feature_dim);
        }
    }
}

template <typename T>
bool FramingRingBuffer3D<T>::write(const std::vector<std::vector<std::vector<T>>>& data_in) {
    if (data_in.empty() || data_in.size() != m_num_channels) {
        return false; // Channel mismatch
    }

    const size_t num_time_steps_to_write = data_in[0].size();
    if (num_time_steps_to_write == 0) {
        return true; // Nothing to write
    }

    if (num_time_steps_to_write > getAvailableWrite()) {
        return false; // Not enough space
    }

    // Write data, looping one time step at a time to handle wrap-around
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (data_in[c].size() != num_time_steps_to_write) {
            return false; // All channels must have same number of time steps
        }

        for (size_t t = 0; t < num_time_steps_to_write; ++t) {
            if (data_in[c][t].size() != m_feature_dim) {
                return false; // Feature dimension mismatch
            }

            // Calculate destination time index
            size_t write_pos_time = (m_write_index_time + t) % m_capacity_time;
            
            // Copy the 1D feature vector
            T* dest_ptr = m_buffers[c][write_pos_time].data();
            const T* src_ptr = data_in[c][t].data();
            std::memcpy(dest_ptr, src_ptr, m_feature_dim * sizeof(T));
        }
    }

    m_write_index_time = (m_write_index_time + num_time_steps_to_write) % m_capacity_time;
    m_available_time += num_time_steps_to_write;

    return true;
}

template <typename T>
bool FramingRingBuffer3D<T>::read(std::vector<std::vector<std::vector<T>>>& frame_data_out) {
     if (m_frame_size_time > m_available_time) {
        return false; // Not enough data to read a full frame
    }
    
    if (m_hop_size_time > m_available_time) {
        return false; // Not enough data to consume a hop
    }

    // --- Validation ---
    if (frame_data_out.size() != m_num_channels) {
        return false; // Incorrect channel count
    }
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (frame_data_out[c].size() != m_frame_size_time) {
            return false; // Incorrect frame time size
        }
        for (size_t t = 0; t < m_frame_size_time; ++t) {
            if (frame_data_out[c][t].size() != m_feature_dim) {
                return false; // Incorrect feature dim
            }
        }
    }
    // --- End Validation ---

    // Read data, looping one time step at a time to handle wrap-around
    for (size_t c = 0; c < m_num_channels; ++c) {
        for (size_t t = 0; t < m_frame_size_time; ++t) {
            // Calculate source time index
            size_t read_pos_time = (m_read_index_time + t) % m_capacity_time;

            // Copy the 1D feature vector
            T* dest_ptr = frame_data_out[c][t].data();
            const T* src_ptr = m_buffers[c][read_pos_time].data();
            std::memcpy(dest_ptr, src_ptr, m_feature_dim * sizeof(T));
        }
    }

    // Consume hop_size_time
    m_read_index_time = (m_read_index_time + m_hop_size_time) % m_capacity_time;
    m_available_time -= m_hop_size_time;

    return true;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableFramesRead() const {
    if (m_available_time < m_frame_size_time) {
        return 0;
    }
    // We have at least one frame.
    // Calculate how many *more* frames can be read based on the hop.
    return 1 + (m_available_time - m_frame_size_time) / m_hop_size_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableTimeRead() const {
    return m_available_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableWrite() const {
    return m_capacity_time - m_available_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getCapacity() const {
    return m_capacity_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getNumChannels() const {
    return m_num_channels;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getFeatureDim() const {
    return m_feature_dim;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getFrameSizeTime() const {
    return m_frame_size_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getHopSizeTime() const {
    return m_hop_size_time;
}

template <typename T>
bool FramingRingBuffer3D<T>::isFull() const {
    return getAvailableWrite() == 0;
}

template <typename T>
bool FramingRingBuffer3D<T>::isEmpty() const {
    return getAvailableTimeRead() == 0;
}

template <typename T>
void FramingRingBuffer3D<T>::clear() {
    m_write_index_time = 0;
    m_read_index_time = 0;
    m_available_time = 0;
}

} // namespace JABuff

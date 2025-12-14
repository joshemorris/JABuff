#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstddef>
#include <string>
#include <algorithm> // For std::min

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
     * @param min_frames The minimum number of available frames required to perform a read.
     * Defaults to 1 (requires at least one full frame to be available).
     * @param keep_frames The number of frames to keep in the buffer after a read operation.
     * These frames will be available for the *next* read operation. Defaults to 0.
     */
    FramingRingBuffer3D(size_t num_channels, size_t feature_dim, size_t capacity_time, size_t frame_size_time, size_t hop_size_time, size_t min_frames = 1, size_t keep_frames = 0);

    /**
     * @brief Writes a block of data.
     * @return true if write succeeded, false if buffer full.
     * @throws std::invalid_argument if dimensions mismatch.
     * @throws std::out_of_range if offset/count exceeds input bounds.
     */
    bool write(const std::vector<std::vector<std::vector<T>>>& data_in, size_t offset_time = 0, size_t num_time_steps = 0);

    /**
     * @brief Writes a single time step to the buffer.
     * * @param time_step_data A vector of size [channels], where each element is a vector of size [feature_dim].
     * @return true if write succeeded, false if buffer full.
     * @throws std::invalid_argument if channel count or feature dimensions mismatch.
     */
    bool push(const std::vector<std::vector<T>>& time_step_data);

    /**
     * @brief Primes the buffer with enough time steps (default 0) so that the next write of 'hop_size'
     * time steps will make the buffer ready to read 'min_frames'.
     *
     * Formula: PrimeTimeSteps = ((min_frames - 1) * hop_size + frame_size) - hop_size.
     *
     * @param value The value to fill the buffer with (default 0).
     */
    void prime(T value = 0);

    /**
     * @brief Checks if the buffer has enough data to perform a read.
     * Checks if the number of available frames is greater than or equal to min_frames.
     * @return true if ready to read.
     */
    bool ready() const;

    /**
     * @brief Reads a contiguous block of data covering the requested frames.
     * * The output is organized as [channel][time][feature].
     * * Unlike previous versions, this does NOT duplicate overlapping time steps.
     * It returns a single continuous vector representing the union of the requested frames.
     * * Total time steps = (num_frames - 1) * hop_size + frame_size.
     * To access Frame 'i' from this buffer, read starting at time index (i * hop_size).
     * * @param buffer_out Output vector [channel][time][feature]. Resized automatically.
     * @param num_frames The number of frames to read. 
     * If 0, reads ALL available frames.
     * If > 0, strictly requires that number of frames to be available.
     * @return true if the frames were successfully read.
     * @return false if there were not enough frames available (or < min_frames available).
     */
    bool read(std::vector<std::vector<std::vector<T>>>& buffer_out, size_t num_frames = 1);

    size_t getAvailableFramesRead() const;
    size_t getAvailableTimeRead() const;
    size_t getAvailableWrite() const;
    size_t getCapacity() const;
    size_t getNumChannels() const;
    size_t getFeatureDim() const;
    size_t getFrameSizeTime() const;
    size_t getHopSizeTime() const;
    size_t getMinFrames() const;
    size_t getKeepFrames() const;
    bool isFull() const;
    bool isEmpty() const;
    void clear();

private:
    // --- Helpers ---
    void validateWriteInput(const std::vector<std::vector<std::vector<T>>>& data_in, size_t offset_time, size_t num_time_steps, size_t& calculated_write_size) const;

    // --- Member Variables ---
    std::vector<std::vector<std::vector<T>>> m_buffers; 
    size_t m_num_channels;
    size_t m_feature_dim;
    size_t m_capacity_time;
    size_t m_frame_size_time;
    size_t m_hop_size_time;
    size_t m_min_frames;
    size_t m_keep_frames;        
    size_t m_write_index_time;
    size_t m_read_index_time;
    size_t m_available_time;
};

// ===================================================================
// --- Implementation ---
// ===================================================================

template <typename T>
FramingRingBuffer3D<T>::FramingRingBuffer3D(size_t num_channels, size_t feature_dim, size_t capacity_time, size_t frame_size_time, size_t hop_size_time, size_t min_frames, size_t keep_frames)
    : m_num_channels(num_channels),
      m_feature_dim(feature_dim),
      m_capacity_time(capacity_time),
      m_frame_size_time(frame_size_time),
      m_hop_size_time(hop_size_time),
      m_min_frames(min_frames),
      m_keep_frames(keep_frames),
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

    m_buffers.resize(m_num_channels);
    for (size_t c = 0; c < m_num_channels; ++c) {
        m_buffers[c].resize(m_capacity_time);
        for (size_t t = 0; t < m_capacity_time; ++t) {
            m_buffers[c][t].resize(m_feature_dim);
        }
    }
}

template <typename T>
void FramingRingBuffer3D<T>::validateWriteInput(const std::vector<std::vector<std::vector<T>>>& data_in, size_t offset_time, size_t num_time_steps, size_t& calculated_write_size) const {
    if (data_in.size() != m_num_channels) {
        throw std::invalid_argument("Input data channel count (" + std::to_string(data_in.size()) + 
                                    ") does not match buffer channels (" + std::to_string(m_num_channels) + ").");
    }

    if (data_in.empty()) return;

    size_t input_time_size = data_in[0].size();
    
    // Check channel consistency
    for(size_t c = 1; c < m_num_channels; ++c) {
        if(data_in[c].size() != input_time_size) {
            throw std::invalid_argument("Input channels have inconsistent time lengths.");
        }
    }

    // Check bounds
    if (offset_time >= input_time_size && input_time_size > 0) {
        throw std::out_of_range("Write offset (" + std::to_string(offset_time) + ") exceeds input time size (" + std::to_string(input_time_size) + ").");
    }

    // Auto-calculate size
    calculated_write_size = num_time_steps;
    if (calculated_write_size == 0) {
        calculated_write_size = input_time_size - offset_time;
    }

    // Check bounds with size
    if (offset_time + calculated_write_size > input_time_size) {
        throw std::out_of_range("Write request (Offset: " + std::to_string(offset_time) + 
                                ", Count: " + std::to_string(calculated_write_size) + 
                                ") exceeds input bounds.");
    }

    // Check feature dimensions for the slice we are writing
    // This adds some overhead, but ensures data integrity before partial writes occur.
    for (size_t c = 0; c < m_num_channels; ++c) {
        for (size_t t = 0; t < calculated_write_size; ++t) {
            size_t idx = offset_time + t;
            if (data_in[c][idx].size() != m_feature_dim) {
                throw std::invalid_argument("Feature dimension mismatch at Ch " + std::to_string(c) + 
                                            ", Time " + std::to_string(idx) + ". Expected " + std::to_string(m_feature_dim) + ".");
            }
        }
    }
}

template <typename T>
bool FramingRingBuffer3D<T>::write(const std::vector<std::vector<std::vector<T>>>& data_in, size_t offset_time, size_t num_time_steps) {
    if (data_in.empty()) return true;

    // 1. Validate (Logic Error -> Exception)
    size_t actual_write_time = 0;
    validateWriteInput(data_in, offset_time, num_time_steps, actual_write_time);

    if (actual_write_time == 0) return true;

    // 2. Check Capacity (Runtime State -> Return False)
    if (actual_write_time > getAvailableWrite()) {
        return false;
    }

    // 3. Write
    for (size_t c = 0; c < m_num_channels; ++c) {
        for (size_t t = 0; t < actual_write_time; ++t) {
            size_t input_index = offset_time + t;
            
            size_t write_pos_time = (m_write_index_time + t) % m_capacity_time;
            
            T* dest_ptr = m_buffers[c][write_pos_time].data();
            const T* src_ptr = data_in[c][input_index].data();
            std::memcpy(dest_ptr, src_ptr, m_feature_dim * sizeof(T));
        }
    }

    m_write_index_time = (m_write_index_time + actual_write_time) % m_capacity_time;
    m_available_time += actual_write_time;

    return true;
}

template <typename T>
bool FramingRingBuffer3D<T>::push(const std::vector<std::vector<T>>& time_step_data) {
    // 1. Validate sizes
    if (time_step_data.size() != m_num_channels) {
        throw std::invalid_argument("Input channel count (" + std::to_string(time_step_data.size()) + 
                                    ") does not match buffer channels (" + std::to_string(m_num_channels) + ").");
    }
    
    // 2. Check Capacity
    if (getAvailableWrite() < 1) {
        return false;
    }

    // 3. Perform Write
    for (size_t c = 0; c < m_num_channels; ++c) {
        if (time_step_data[c].size() != m_feature_dim) {
            throw std::invalid_argument("Feature dimension mismatch at Ch " + std::to_string(c) + ".");
        }
        
        T* dest_ptr = m_buffers[c][m_write_index_time].data();
        const T* src_ptr = time_step_data[c].data();
        std::memcpy(dest_ptr, src_ptr, m_feature_dim * sizeof(T));
    }

    m_write_index_time = (m_write_index_time + 1) % m_capacity_time;
    m_available_time++;

    return true;
}

template <typename T>
void FramingRingBuffer3D<T>::prime(T value) {
    // Calculate total time steps needed to satisfy min_frames requirement
    size_t target_time = (m_min_frames - 1) * m_hop_size_time + m_frame_size_time;
    
    // We want the NEXT hop to make it ready, so we need (Target - Hop) now.
    size_t time_to_prime = 0;
    
    if (target_time > m_hop_size_time) {
        time_to_prime = target_time - m_hop_size_time;
    }

    if (time_to_prime > 0) {
        // Create a 3D block: [Channels][Time][Features]
        std::vector<std::vector<std::vector<T>>> prime_data(
            m_num_channels, 
            std::vector<std::vector<T>>(time_to_prime, std::vector<T>(m_feature_dim, value))
        );
        
        // Write it to the buffer (this handles index updates and wrapping)
        write(prime_data);
    }
}

template <typename T>
bool FramingRingBuffer3D<T>::ready() const {
    return getAvailableFramesRead() >= m_min_frames;
}

template <typename T>
bool FramingRingBuffer3D<T>::read(std::vector<std::vector<std::vector<T>>>& buffer_out, size_t num_frames) {
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

    // Calculate total continuous time steps needed to cover these frames
    // Size = (N-1) * hop + frame_size
    size_t total_time_steps = (count_to_read - 1) * m_hop_size_time + m_frame_size_time;
    
    // Resize output vector [Channels][Total Time Steps][Features]
    buffer_out.resize(m_num_channels);
    for(size_t c = 0; c < m_num_channels; ++c) {
        buffer_out[c].resize(total_time_steps);
        for(size_t t = 0; t < total_time_steps; ++t) {
            buffer_out[c][t].resize(m_feature_dim);
        }
    }

    for (size_t c = 0; c < m_num_channels; ++c) {
        // Continuous copy of total_time_steps, handling wrap-around
        for (size_t t = 0; t < total_time_steps; ++t) {
            size_t read_pos_time = (m_read_index_time + t) % m_capacity_time;
            
            T* dest_feature = buffer_out[c][t].data();
            const T* src_feature = m_buffers[c][read_pos_time].data();
            std::memcpy(dest_feature, src_feature, m_feature_dim * sizeof(T));
        }
    }

    // Update actual member variables based on keep_frames
    size_t frames_consumed = 0;
    if (count_to_read > m_keep_frames) {
        frames_consumed = count_to_read - m_keep_frames;
    }

    size_t time_consumed = frames_consumed * m_hop_size_time;

    m_read_index_time = (m_read_index_time + time_consumed) % m_capacity_time;
    m_available_time -= time_consumed;

    return true;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableFramesRead() const {
    if (m_available_time < m_frame_size_time) return 0;
    return 1 + (m_available_time - m_frame_size_time) / m_hop_size_time;
}

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableTimeRead() const { return m_available_time; }

template <typename T>
size_t FramingRingBuffer3D<T>::getAvailableWrite() const { return m_capacity_time - m_available_time; }

template <typename T>
size_t FramingRingBuffer3D<T>::getCapacity() const { return m_capacity_time; }

template <typename T>
size_t FramingRingBuffer3D<T>::getNumChannels() const { return m_num_channels; }

template <typename T>
size_t FramingRingBuffer3D<T>::getFeatureDim() const { return m_feature_dim; }

template <typename T>
size_t FramingRingBuffer3D<T>::getFrameSizeTime() const { return m_frame_size_time; }

template <typename T>
size_t FramingRingBuffer3D<T>::getHopSizeTime() const { return m_hop_size_time; }

template <typename T>
size_t FramingRingBuffer3D<T>::getMinFrames() const { return m_min_frames; }

template <typename T>
size_t FramingRingBuffer3D<T>::getKeepFrames() const { return m_keep_frames; }

template <typename T>
bool FramingRingBuffer3D<T>::isFull() const { return getAvailableWrite() == 0; }

template <typename T>
bool FramingRingBuffer3D<T>::isEmpty() const { return getAvailableTimeRead() == 0; }

template <typename T>
void FramingRingBuffer3D<T>::clear() {
    m_write_index_time = 0;
    m_read_index_time = 0;
    m_available_time = 0;
}

} // namespace JABuff

#include "JABuff/FramingRingBuffer2D.hpp"
#include "test_utils.hpp"

void TestExceptions() {
    print_header("TestExceptions");
    JABuff::FramingRingBuffer2D<float> buffer(1, 100, 10, 5);
    std::vector<std::vector<float>> input(1, std::vector<float>(20));

    // 1. Offset out of range
    bool caught = false;
    try {
        buffer.write(input, 50);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    ASSERT(caught, "Offset > size did not throw");

    // 2. Offset + Count out of range
    caught = false;
    try {
        buffer.write(input, 15, 10);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    ASSERT(caught, "Offset+Count > size did not throw");

    // 3. Channel mismatch
    std::vector<std::vector<float>> bad_ch(2, std::vector<float>(10));
    caught = false;
    try {
        buffer.write(bad_ch);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT(caught, "Channel mismatch did not throw");

    // 4. Push Channel Mismatch
    caught = false;
    try {
        std::vector<float> bad_frame(5); // Buffer expects 1
        buffer.push(bad_frame);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT(caught, "Push size mismatch did not throw");
}

int main() {
    TestExceptions();
    print_pass();
    return 0;
}

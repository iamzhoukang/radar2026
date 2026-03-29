#ifndef SERIAL_DRIVER_HPP
#define SERIAL_DRIVER_HPP

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <mutex>

#include "rm_protocol.hpp"

namespace radar_core {

class SerialDriver {
public:
    using PacketCallback = std::function<void(uint16_t cmd_id, uint8_t* data, uint16_t len)>;

    SerialDriver(const std::string& port_name) 
        : port_name_(port_name), fd_(-1), seq_(0), is_running_(false) {}

    ~SerialDriver() {
        closePort();
    }

    void setCallback(PacketCallback cb) {
        callback_ = cb;
    }

    bool openPort() {
        fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ == -1) return false;

        struct termios options;
        tcgetattr(fd_, &options);

        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);

        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;         
        options.c_cflag &= ~PARENB;     
        options.c_cflag &= ~CSTOPB;     
        options.c_cflag &= ~CRTSCTS;    

        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_iflag &= ~(IXON | IXOFF | IXANY);
        options.c_iflag &= ~(INLCR | ICRNL | IGNCR);
        options.c_oflag &= ~OPOST;

        options.c_cc[VTIME] = 1; 
        options.c_cc[VMIN]  = 0;

        tcflush(fd_, TCIFLUSH);
        if (tcsetattr(fd_, TCSANOW, &options) != 0) {
            close(fd_);
            return false;
        }

        is_running_ = true;
        receive_thread_ = std::thread(&SerialDriver::receiveLoop, this);
        return true;
    }

    void closePort() {
        is_running_ = false;
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
    }

    bool isOpen() const {
        return fd_ != -1;
    }

    bool sendPacket(uint16_t cmd_id, const uint8_t* payload, uint16_t payload_len) {
        if (!isOpen()) return false;

        uint16_t frame_len = 5 + 2 + payload_len + 2;
        std::vector<uint8_t> buffer(frame_len, 0);

        buffer[0] = RM_SOF;                             
        buffer[1] = payload_len & 0xFF;                 
        buffer[2] = (payload_len >> 8) & 0xFF;          
        buffer[3] = seq_++;                             
        Append_CRC8_Check_Sum(buffer.data(), 5);        

        buffer[5] = cmd_id & 0xFF;
        buffer[6] = (cmd_id >> 8) & 0xFF;

        if (payload_len > 0 && payload != nullptr) {
            std::memcpy(buffer.data() + 7, payload, payload_len);
        }

        Append_CRC16_Check_Sum(buffer.data(), frame_len);

        std::lock_guard<std::mutex> lock(write_mutex_);
        int bytes_written = write(fd_, buffer.data(), frame_len);
        return bytes_written == frame_len;
    }

private:
    std::string port_name_;
    int fd_;
    uint8_t seq_;
    std::atomic<bool> is_running_;
    std::thread receive_thread_;
    std::mutex write_mutex_;
    PacketCallback callback_;

    bool readFixed(uint8_t* buf, int expected_len) {
        int total_read = 0;
        while (total_read < expected_len && is_running_) {
            int n = read(fd_, buf + total_read, expected_len - total_read);
            if (n > 0) {
                total_read += n;
            } else if (n < 0) {
                return false;
            }
        }
        return total_read == expected_len;
    }

    void receiveLoop() {
        uint8_t buf[256];
        while (is_running_) {
            int n = read(fd_, buf, 1);
            if (n <= 0) continue; 
            if (buf[0] != RM_SOF) continue; 

            if (!readFixed(buf + 1, 4)) continue;
            if (!Verify_CRC8_Check_Sum(buf, 5)) continue;

            uint16_t data_len = (buf[2] << 8) | buf[1];
            int remain_len = 2 + data_len + 2; 

            if (remain_len > 251) continue; 
            if (!readFixed(buf + 5, remain_len)) continue;

            int total_len = 5 + remain_len;
            if (!Verify_CRC16_Check_Sum(buf, total_len)) continue;

            if (callback_) {
                uint16_t cmd_id = (buf[6] << 8) | buf[5];
                callback_(cmd_id, buf + 7, data_len);
            }
        }
    }
};

} // namespace radar_core

#endif // SERIAL_DRIVER_HPP
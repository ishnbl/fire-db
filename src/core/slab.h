#pragma once
#ifndef FIREDB_SLAB_H
#define FIREDB_SLAB_H
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <filesystem>
enum OpCode : uint8_t {
    OP_INSERT = 1,
    OP_DELETE = 2
};

class IdSlab {
    private:
        std::unordered_map<uint64_t ,uint64_t> user_auto;
        std::vector<int64_t> auto_row;
        std::string fpath;
        std::fstream logfile;
        uint64_t auto_id = 0;

        void replay_log() {
            logfile.clear();
            logfile.seekg(0, std::ios::beg);

            uint8_t op_val;
            uint64_t user_id;
            uint64_t read_aid;
            int64_t row_index;

            while (logfile.read(reinterpret_cast<char*>(&op_val), sizeof(op_val))) {
                OpCode op = static_cast<OpCode>(op_val);
                logfile.read(reinterpret_cast<char*>(&user_id), sizeof(user_id));
                logfile.read(reinterpret_cast<char*>(&read_aid), sizeof(read_aid));
                logfile.read(reinterpret_cast<char*>(&row_index), sizeof(row_index));

                if (op == OP_INSERT) {
                    user_auto[user_id] = read_aid;

                    if (read_aid >= auto_row.size()) {
                        auto_row.resize(read_aid + 1, -1);
                    }
                    auto_row[read_aid] = row_index;

                    if (read_aid >= auto_id) {
                        auto_id = read_aid + 1;
                    }
                }
                else if (op == OP_DELETE) {
                    user_auto.erase(user_id);
                    if (read_aid < auto_row.size()) {
                        auto_row[read_aid] = -1;
                    }
                }
            }

            if (auto_row.size() > auto_id) {
                auto_id = auto_row.size();
            }

            logfile.clear();
            logfile.seekp(0, std::ios::end);
        };

        void write_log_entry(OpCode op, uint64_t uid, uint64_t aid, int64_t row) {
            logfile.write((char*)&op, sizeof(op));
            logfile.write((char*)&uid, sizeof(uid));
            logfile.write((char*)&aid, sizeof(aid));
            logfile.write((char*)&row, sizeof(row));

            logfile.flush();
        };

    public:
        IdSlab(const std::string& path_file) : fpath(path_file) {
            logfile.open(fpath, std::ios::in | std::ios::out | std::ios::app | std::ios::binary);

            if (!logfile.is_open()) {
                logfile.clear();
                logfile.open(fpath, std::ios::out | std::ios::binary);
                logfile.close();
                logfile.open(fpath, std::ios::in | std::ios::out | std::ios::app | std::ios::binary);
            }
            replay_log();
        };

        ~IdSlab() {
            if (logfile.is_open()) {
                logfile.close();
            }
        }

        std::optional<uint64_t> insert(uint64_t user_id, int64_t row_index) {
            if (user_auto.count(user_id)) {
                return std::nullopt;
            }

            uint64_t id = auto_id++;

            user_auto[user_id] = id;
            auto_row.push_back(row_index);

            write_log_entry(OP_INSERT, user_id, id, row_index);
            return id;
        }
        void remove(uint64_t user_id) {
            if (!user_auto.count(user_id)) return;

            uint64_t aid = user_auto[user_id];
            user_auto.erase(user_id);

            if (aid < auto_row.size()) {
                auto_row[aid] = -1;
            }

            write_log_entry(OP_DELETE, user_id, aid, -1);
        }

        int64_t get_row(uint64_t auto_id) {
            if (auto_id >= auto_row.size()) return -1;
            return auto_row[auto_id];
        }
        int64_t get_row_from_user(uint64_t uid) {
            if (user_auto.find(uid) == user_auto.end()) return -1;
            return get_row(user_auto[uid]);
        }
};

struct SlabHeader {
    uint32_t magic = 0x26872687;
    uint32_t version = 1;
    uint64_t count = 0;
    uint64_t dim = 0;
    uint64_t capacity = 0;
    char _pad[96];
};

class MatrixSlab {
    private:
        std::string fpath;
        int fd;
        size_t file_size;
        SlabHeader* header;
        float* data_region;

        const size_t INITIAL_CAPACITY = 1000;

        void map_file(size_t file_size_byes) {
            if (header != nullptr) {
                munmap(header, this->file_size);
            }

            file_size = file_size_byes;

            void* ptr = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr == MAP_FAILED) {
                throw std::runtime_error("mmap failed");
            }

            header = static_cast<SlabHeader*>(ptr);
            data_region = reinterpret_cast<float*>(static_cast<char*>(ptr) + sizeof(SlabHeader));
        }

        void grow_file(size_t new_capacity) {
            size_t new_capacity_bytes = sizeof(SlabHeader) + new_capacity * sizeof(float)*header->dim;
            if (ftruncate(fd, new_capacity_bytes) == -1) {
                throw std::runtime_error("ftruncate failed");
            }
            map_file(new_capacity_bytes);
            header->capacity = new_capacity;
        }
    public:
        MatrixSlab(const std::string& path_file, uint64_t dimension) : fpath(path_file) , header(nullptr) {
            bool is_new = !std::filesystem::exists(fpath);
            fd = open(fpath.c_str(), O_RDWR | O_CREAT , 0644);
            if (is_new) {
                size_t size = sizeof(SlabHeader) + dimension * sizeof(float)*INITIAL_CAPACITY;
                ftruncate(fd, size);
                map_file(size);
                header->magic = 0x26872687;
                header->count = 0;
                header->dim = dimension;
                header->capacity = INITIAL_CAPACITY;
            }else {
                struct stat st;
                fstat(fd, &st);
                map_file(st.st_size);
            }
        }
        ~MatrixSlab() {
                if (header) munmap(header, file_size);
                if (fd != -1) close(fd);
        }

        void add_vector(const float* vector_Data) {
            if (header->count >= header->capacity) {
                grow_file(header->capacity*2);
            }
            size_t offset = header->count * header->dim;
            std::memcpy(&data_region[offset], vector_Data, header->dim * sizeof(float));
            header->count++;
        }
        const float* get_data_ptr() const { return data_region; }
        uint64_t get_count() const { return header->count; }
        uint64_t get_capacity() const { return header->capacity; }
        uint64_t get_dim() const { return header->dim; }

};
#endif

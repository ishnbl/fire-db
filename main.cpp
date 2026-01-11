#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <regex>

#include "src/core/slab.h"
#include "src/core/gpu.h"

int GLOBAL_DIM = 0;
const int MAX_CAPACITY = 1000000;
const int GPU_BATCH_LIMIT = 100;

struct NpyHeader {
    int rows;
    int cols;
    size_t header_size;
    bool is_float32;
};

NpyHeader parse_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");

    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("Invalid NPY file");

    file.seekg(8);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    NpyHeader info{0, 0, 10 + header_len, false};

    if (header.find("<f4") != std::string::npos ||
        header.find("'f4'") != std::string::npos)
        info.is_float32 = true;

    std::regex r(R"(shape['"]?:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))");
    std::smatch m;
    if (!std::regex_search(header, m, r))
        throw std::runtime_error("Could not parse shape");

    info.rows = std::stoi(m[1]);
    info.cols = std::stoi(m[2]);
    return info;
}

std::vector<float> generate_random_vector(int dim) {
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (int i = 0; i < dim; i++) v[i] = dis(gen);
    return v;
}

void print_help() {
    std::cout <<
        "Commands:\n"
        "  status            : Show stats\n"
        "  import <f.npy>    : Import vectors\n"
        "  gen <num>         : Generate random vectors\n"
        "  add <id>          : Add random vector\n"
        "  put <id> <v...>   : Add custom vector\n"
        "  search            : Search with random query\n"
        "  find <id>         : Find neighbors\n"
        "  batch <num>       : Benchmark batch search\n"
        "  exit              : Quit\n";
}

int main() {
    std::cout << "FireDB\n";

    std::string db_name;
    std::getline(std::cin, db_name);
    if (db_name.empty()) db_name = "main";

    std::string vec_file = db_name + ".slab";
    std::string id_file = db_name + ".wal";

    if (std::filesystem::exists(vec_file)) {
        MatrixSlab t(vec_file, 0);
        GLOBAL_DIM = t.get_dim();
        std::cout << "Loading '" << db_name << "' (Dim: " << GLOBAL_DIM << ")\n";
    } else {
        GLOBAL_DIM = 128;
        std::cout << "Creating '" << db_name << "' (Dim: " << GLOBAL_DIM << ")\n";
    }

    IdSlab id_db(id_file);
    MatrixSlab mat_db(vec_file, GLOBAL_DIM);

    std::cout << "[GPU] Allocating Index...\n";
    GpuIndex gpu(GLOBAL_DIM, MAX_CAPACITY);

    if (mat_db.get_count() > 0) {
        std::cout << "[GPU] Uploading " << mat_db.get_count() << " vectors...\n";
        gpu.load_data(mat_db);
    }

    std::cout << "Ready.\n";

    std::string line, cmd;
    while (true) {
        std::cout << db_name << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        std::stringstream ss(line);
        ss >> cmd;

        if (cmd == "exit" || cmd == "quit") break;
        else if (cmd == "help") print_help();

        else if (cmd == "status") {
            std::cout << "Vectors: " << mat_db.get_count() << "\n"
                      << "Dim:     " << GLOBAL_DIM << "\n";
        }

        else if (cmd == "import") {
            std::string path;
            ss >> path;
            if (path.empty()) {
                std::cout << "Usage: import <vectors.npy>\n";
                continue;
            }

            try {
                NpyHeader h = parse_npy(path);
                if (!h.is_float32) {
                    std::cout << "Error: Only float32 supported.\n";
                    continue;
                }
                if (h.cols != GLOBAL_DIM) {
                    std::cout << "Error: NPY dim (" << h.cols
                              << ") != DB dim (" << GLOBAL_DIM << ")\n";
                    continue;
                }

                std::ifstream f(path, std::ios::binary);
                f.seekg(h.header_size);

                std::vector<float> buf(GLOBAL_DIM);
                uint64_t uid = 100000 + mat_db.get_count();

                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < h.rows; i++) {
                    f.read(reinterpret_cast<char*>(buf.data()),
                           GLOBAL_DIM * sizeof(float));
                    if (!f) break;

                    int64_t row = mat_db.get_count();
                    mat_db.add_vector(buf.data());
                    id_db.insert(uid++, row);
                    gpu.add_single_vector(buf.data());
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                std::cout << "Imported " << h.rows << " vectors in "
                          << std::chrono::duration<double>(t1 - t0).count()
                          << "s\n";
            } catch (const std::exception& e) {
                std::cout << "Import failed: " << e.what() << "\n";
            }
        }

        else if (cmd == "add") {
            uint64_t uid;
            if (!(ss >> uid)) continue;
            if (id_db.get_row_from_user(uid) != -1) continue;

            auto v = generate_random_vector(GLOBAL_DIM);
            int64_t row = mat_db.get_count();
            mat_db.add_vector(v.data());
            id_db.insert(uid, row);
            gpu.add_single_vector(v.data());
        }

        else if (cmd == "put") {
            uint64_t uid;
            ss >> uid;

            std::vector<float> v;
            float x;
            while (ss >> x) v.push_back(x);

            if (v.size() != (size_t)GLOBAL_DIM) continue;
            if (id_db.get_row_from_user(uid) != -1) continue;

            int64_t row = mat_db.get_count();
            mat_db.add_vector(v.data());
            id_db.insert(uid, row);
            gpu.add_single_vector(v.data());
        }

        else if (cmd == "gen") {
            int n;
            ss >> n;
            uint64_t uid = 100000 + mat_db.get_count();

            for (int i = 0; i < n; i++) {
                auto v = generate_random_vector(GLOBAL_DIM);
                int64_t row = mat_db.get_count();
                mat_db.add_vector(v.data());
                id_db.insert(uid++, row);
                gpu.add_single_vector(v.data());
            }
        }

        else if (cmd == "search") {
            auto q = generate_random_vector(GLOBAL_DIM);
            auto r = gpu.search_one(q, 5);
            for (auto& x : r)
                std::cout << "Row " << x.id << " | Dist " << x.score << "\n";
        }

        else if (cmd == "find") {
            uint64_t uid;
            ss >> uid;
            int64_t row = id_db.get_row_from_user(uid);
            if (row == -1) continue;

            std::vector<float> q(GLOBAL_DIM);
            std::memcpy(q.data(),
                        mat_db.get_data_ptr() + row * GLOBAL_DIM,
                        GLOBAL_DIM * sizeof(float));

            auto r = gpu.search_one(q, 5);
            for (auto& x : r)
                if (x.id != row)
                    std::cout << "Neighbor row " << x.id
                              << " | Dist " << x.score << "\n";
        }

        else if (cmd == "batch") {
            int n;
            ss >> n;

            std::vector<std::vector<float>> qs(n);
            for (auto& q : qs) q = generate_random_vector(GLOBAL_DIM);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n; i += GPU_BATCH_LIMIT) {
                int c = std::min(GPU_BATCH_LIMIT, n - i);
                gpu.search(
                    std::vector<std::vector<float>>(
                        qs.begin() + i,
                        qs.begin() + i + c
                    ),
                    5
                );
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "QPS: " << int(n / s) << "\n";
        }

        else {
            std::cout << "Unknown command.\n";
        }
    }
    return 0;
}

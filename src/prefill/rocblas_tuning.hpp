// src/prefill/rocblas_tuning.hpp - the rocBLAS solution table of the prompt path's FP16-out GEMMs (HIP, gfx103x).
//
// rocBLAS picks a kernel per GEMM shape from its own table, and on gfx1030 that pick is up to 6x slower than the best
// kernel the library holds for the same shape (N=10240 K=2560 below T=1152: tools/hip/tune_rocblas.cpp).  A table
// made by that tool names, per shape and token bucket, the solution index to run instead; Gemm::f16_inplace reads it
// through STRATA_ROCBLAS_TUNING.  Indices are valid for one architecture and one rocBLAS build (its full version
// string), and the table carries both; any other pair is refused and the default kernels run.
//
// The header may also name the width of the card it was measured on (`mp=40`): HIP's multiProcessorCount, which is
// the compute units on GCN/CDNA and the workgroup processors (two CUs each) on RDNA, so 40 on an RX 6900 XT's 80 CUs.
// One architecture name covers cards of different widths (gfx1030: RX 6800 and Radeon PRO W6800 with 60 CUs,
// mp 30; RX 6800 XT and Radeon PRO V620 72, mp 36; RX 6900 XT and 6950 XT 80, mp 40; and cards run as gfx1030
// through HSA_OVERRIDE_GFX_VERSION), and where several kernels are close, which one is fastest follows the width
// (tiles per wave of the card).  A table of another width still loads (its indices are valid there); the engine
// says so, and a table measured on the card itself (a per-device cache, see cache_path) comes first.
#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::prefill::rocblas_tuning {

struct Row {
    int n = 0;
    int k = 0;
    int ldy = 0;
    int t_bucket = 0;
    bool tuned = false;   // false: the default kernel measured best at this bucket (the row bounds its neighbours' reach)
    int solution = 0;     // rocBLAS solution indices can be negative
};

class Table {
public:
    /// `expected_version`: rocblas_get_version_string() of the library the engine runs with.
    /// mp(): the header's multiprocessor count when it names one, 0 otherwise; the caller compares it.
    bool load(const std::string& path, const std::string& expected_arch, const std::string& expected_version,
              std::string& err) {
        rows_.clear();
        mp_ = 0;
        std::ifstream input(path);
        if (!input) {
            err = "cannot open rocBLAS tuning file";
            return false;
        }
        bool saw_header = false;
        std::string line;
        size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            std::istringstream row(line.substr(first));
            if (!saw_header) {
                std::string magic, arch, version, mp, extra;
                if (!(row >> magic >> arch >> version) || magic != "STRATA_ROCBLAS_TUNING_V1") {
                    err = "invalid rocBLAS tuning header at line " + std::to_string(line_number);
                    return false;
                }
                if (row >> mp) {
                    char* end = nullptr;
                    const long value = mp.rfind("mp=", 0) == 0 ? std::strtol(mp.c_str() + 3, &end, 10) : 0;
                    if (!end || *end || value <= 0 || value > 65535 || (row >> extra)) {
                        err = "invalid rocBLAS tuning header at line " + std::to_string(line_number);
                        return false;
                    }
                    mp_ = (int) value;
                }
                if (arch != expected_arch) {
                    err = "rocBLAS tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                if (version != expected_version) {
                    err = "rocBLAS version mismatch: file=" + version + " runtime=" + expected_version;
                    return false;
                }
                arch_ = std::move(arch);
                version_ = std::move(version);
                saw_header = true;
                continue;
            }
            std::string solution, extra;
            Row parsed;
            std::string type;
            if (!(row >> type >> parsed.n >> parsed.k >> parsed.ldy >> parsed.t_bucket >> solution) || (row >> extra) ||
                type != "f16o" || parsed.n <= 0 || parsed.k <= 0 || parsed.ldy < parsed.n || parsed.t_bucket <= 0) {
                err = "invalid rocBLAS tuning row at line " + std::to_string(line_number);
                return false;
            }
            if (solution != "default") {
                char* end = nullptr;
                const long value = std::strtol(solution.c_str(), &end, 10);
                if (!end || *end || value < INT_MIN || value > INT_MAX) {
                    err = "invalid rocBLAS solution index at line " + std::to_string(line_number);
                    return false;
                }
                parsed.tuned = true;
                parsed.solution = (int) value;
            }
            for (const auto& existing : rows_) {
                if (existing.n == parsed.n && existing.k == parsed.k && existing.ldy == parsed.ldy &&
                    existing.t_bucket == parsed.t_bucket) {
                    err = "duplicate rocBLAS tuning key at line " + std::to_string(line_number);
                    return false;
                }
            }
            rows_.push_back(parsed);
        }
        if (!saw_header) {
            err = "rocBLAS tuning file has no header";
            return false;
        }
        return true;
    }

    /// The row of this shape whose token bucket is nearest to t (the smaller bucket on a tie); null when the shape
    /// has no row at all (never measured).  A `default` row says the default kernel measured best there, so a tuned
    /// neighbour's solution reaches no further than halfway to it.
    const Row* closest(int n, int k, int ldy, int t) const {
        const Row* best = nullptr;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (const auto& row : rows_) {
            if (row.n != n || row.k != k || row.ldy != ldy) continue;
            const int64_t distance = row.t_bucket >= t ? int64_t(row.t_bucket) - t : int64_t(t) - row.t_bucket;
            if (distance < best_distance || (distance == best_distance && row.t_bucket < best->t_bucket)) {
                best = &row;
                best_distance = distance;
            }
        }
        return best;
    }

    const std::vector<Row>& rows() const { return rows_; }
    const std::string& arch() const { return arch_; }
    const std::string& version() const { return version_; }
    int mp() const { return mp_; }

private:
    std::string arch_;
    std::string version_;
    int mp_ = 0;
    std::vector<Row> rows_;
};

/// The version string as it goes into a file name (trailing dots dropped: "5.6.0.8d1ae90e.").
inline std::string version_tag(std::string version) {
    while (!version.empty() && version.back() == '.') version.pop_back();
    return version;
}

/// The per-device cache directory: STRATA_TUNING_CACHE (empty or "0": none), else $XDG_CACHE_HOME/strata, else
/// %LOCALAPPDATA%/Strata/cache (Windows, which sets no HOME), else $HOME/.cache/strata.  Empty when there is none.
inline std::string cache_dir() {
    if (const char* v = std::getenv("STRATA_TUNING_CACHE")) {
        if (!*v || (v[0] == '0' && !v[1])) return {};
        return v;
    }
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/strata";
    if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) return std::string(l) + "/Strata/cache";
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/strata";
    return {};
}

/// The file of one card width and library build in the cache directory, `<arch>-mp<N>-rocblas-<version>[-<what>].txt`
/// (empty without a cache directory).
inline std::string cache_path(const std::string& arch, int mp, const std::string& version, const std::string& what) {
    const std::string dir = cache_dir();
    if (dir.empty()) return {};
    return dir + "/" + arch + "-mp" + std::to_string(mp) + "-rocblas-" + version_tag(version) +
           (what.empty() ? "" : "-" + what) + ".txt";
}

}  // namespace strata::prefill::rocblas_tuning

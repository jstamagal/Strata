// rocBLAS solution table (src/prefill/rocblas_tuning.hpp), synthetic tables: no model, no GPU.  The header's optional
// width (mp=N, HIP's multiProcessorCount) and the per-device cache file names.
#include "rocblas_tuning.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
namespace rt = strata::prefill::rocblas_tuning;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-rocblas-table-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

fs::path write(const fs::path& dir, const std::string& name, const std::string& text) {
    const fs::path p = dir / name;
    std::ofstream(p) << text;
    return p;
}

}  // namespace

int main() {
    TempDir dir;
    const std::string arch = "gfx1030", ver = "5.6.0.8d1ae90e";
    std::string err;

    {   // a header without mp= (tables made before it) still loads, mp() 0
        rt::Table t;
        const auto p = write(dir.path, "v1.txt", "STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e\nf16o 10240 2560 10240 256 -5\n");
        check(t.load(p.string(), arch, ver, err), "header without mp loads");
        check(t.mp() == 0, "header without mp: mp() 0");
        check(t.closest(10240, 2560, 10240, 300) && t.closest(10240, 2560, 10240, 300)->solution == -5, "f16o lookup");
    }
    {   // mp=80 and a trailing comment
        rt::Table t;
        const auto p = write(dir.path, "mp.txt",
                             "# c\nSTRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e mp=80   # an RX 6900 XT\n"
                             "f16o 512 2560 512 1024 default\n");
        check(t.load(p.string(), arch, ver, err), "header with mp loads");
        check(t.mp() == 80, "mp() 80");
    }
    for (const char* bad : {"mp=0", "mp=x", "80", "mp=80 extra"}) {   // malformed width
        rt::Table t;
        const auto p = write(dir.path, "bad.txt", std::string("STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e ") + bad + "\n");
        check(!t.load(p.string(), arch, ver, err), bad);
    }
    {   // architecture and version still gate
        rt::Table t;
        const auto p = write(dir.path, "arch.txt", "STRATA_ROCBLAS_TUNING_V1 gfx1031 5.6.0.8d1ae90e mp=40\n");
        check(!t.load(p.string(), arch, ver, err), "other architecture refused");
        const auto q = write(dir.path, "ver.txt", "STRATA_ROCBLAS_TUNING_V1 gfx1030 5.8.0.0d45e836 mp=80\n");
        check(!t.load(q.string(), arch, ver, err), "other rocBLAS build refused");
    }
    {   // cache paths: per width and build, version's trailing dots dropped; STRATA_TUNING_CACHE=0 turns it off
#if defined(_WIN32)
        _putenv_s("STRATA_TUNING_CACHE", dir.path.string().c_str());
#else
        setenv("STRATA_TUNING_CACHE", dir.path.string().c_str(), 1);
#endif
        check(rt::cache_path(arch, 72, "5.6.0.8d1ae90e.", "select") ==
                  (dir.path / "gfx1030-mp72-rocblas-5.6.0.8d1ae90e-select.txt").string(),
              "cache path");
#if defined(_WIN32)
        _putenv_s("STRATA_TUNING_CACHE", "0");
#else
        setenv("STRATA_TUNING_CACHE", "0", 1);
#endif
        check(rt::cache_path(arch, 72, ver, "select").empty(), "STRATA_TUNING_CACHE=0: no cache");
    }
    if (failures) return 1;
    std::printf("rocblas_tuning_table_test: ok\n");
    return 0;
}

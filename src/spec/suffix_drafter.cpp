// src/spec/suffix_drafter.cpp - see include/strata/spec/suffix_drafter.hpp.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>

namespace strata::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
}

void SuffixDrafter::reset() {
    hist_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    last_match_ = 0;
    last_end_ = -1;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        Slot& s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            return &s;
        }
    }
    return nullptr;                                         // table full: the history outgrew its capacity
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        if (end < 2) continue;
        Slot* s = find_slot(key_at(end), true);
        if (s == nullptr) continue;
        for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) ++s->n;
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    last_end_ = -1;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const Slot* s = find_slot(key_at(cur), false);
    if (s == nullptr) return 0;
    size_t best_end = 0;
    int best_len = 0;
    for (int w = 0; w < s->n; ++w) {
        const size_t p = s->pos[w];
        if (p >= cur) continue;                             // the current suffix itself
        int len = 0;
        while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
        if (len > best_len) { best_len = len; best_end = p; }   // most recent first, so ties keep the newer
    }
    if (best_len < min_match_) return 0;
    last_match_ = best_len;
    last_end_ = (int64_t) best_end;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

int SuffixDrafter::propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    if (n_extra <= 0) return propose(max_k, out);
    // the index holds the history's trigrams only: appended without indexing, the extra tokens just form the suffix
    const size_t n0 = hist_.size();
    hist_.insert(hist_.end(), extra, extra + n_extra);
    const int k = propose(max_k, out);
    hist_.resize(n0);
    return k;
}

int SuffixDrafter::propose_near(const int32_t* extra, int n_extra, int64_t expected_end,
                              size_t source_limit, int max_k, int32_t* out) {
    last_match_ = 0;
    last_end_ = -1;
    const size_t n = hist_.size() + (size_t) std::max(0, n_extra);
    const size_t limit = std::min(source_limit, hist_.size());
    if (n < 2 || limit < 2 || max_k <= 0) return 0;
    auto token = [&](size_t i) { return i < hist_.size() ? hist_[i] : extra[i - hist_.size()]; };
    // Nearest first; equal-length matches retain the closest source position.
    for (int step = 0; step <= 8; ++step) {
        const int delta = step == 0 ? 0 : (step % 2 ? (step + 1) / 2 : -step / 2);
        const int64_t end = expected_end + delta;
        if (end < 1 || end >= (int64_t) limit - 1) continue;
        int len = 0;
        while (len < max_match_ && len <= end && (size_t) len < n &&
               hist_[(size_t) end - len] == token(n - 1 - len)) ++len;
        if (len >= 2 && len > last_match_) { last_match_ = len; last_end_ = end; }
    }
    if (last_end_ < 0) return 0;
    int k = 0;
    for (size_t q = (size_t) last_end_ + 1; q < limit && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

}  // namespace strata::spec

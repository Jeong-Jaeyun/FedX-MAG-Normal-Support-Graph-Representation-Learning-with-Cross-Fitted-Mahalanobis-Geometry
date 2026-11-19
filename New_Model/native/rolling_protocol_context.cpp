#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct Event {
    int64_t time;
    uint64_t source, destination;
    int64_t port;
    double failed, bytes, packets;
    int64_t query;
};
static_assert(sizeof(Event) == 64, "Native event ABI differs");

struct Statistics {
    int64_t count = 0;
    std::vector<int64_t> sums;
    explicit Statistics(int64_t tags) : sums(tags, 0) {}
};

struct History {
    int64_t horizon, oldest = 0, tags;
    std::unordered_map<uint64_t, Statistics> nodes;
    History(int64_t seconds, int64_t dimensions) : horizon(seconds), tags(dimensions) {}
    void expire(const Event* events, const uint8_t* flags, int64_t end, int64_t now) {
        while (oldest < end && events[oldest].time < now - horizon) {
            auto iterator = nodes.find(events[oldest].source);
            auto& state = iterator->second;
            --state.count;
            for (int64_t j = 0; j < tags; ++j) state.sums[j] -= flags[oldest * tags + j];
            if (!state.count) nodes.erase(iterator);
            ++oldest;
        }
    }
    void add(const Event& event, const uint8_t* flags) {
        auto iterator = nodes.try_emplace(event.source, tags).first;
        auto& state = iterator->second;
        ++state.count;
        for (int64_t j = 0; j < tags; ++j) state.sums[j] += flags[j];
    }
    void query(uint64_t source, float* output) const {
        auto iterator = nodes.find(source);
        if (iterator == nodes.end()) return;
        const auto& state = iterator->second;
        for (int64_t j = 0; j < tags; ++j) {
            output[2 * j] = static_cast<float>(std::log1p(static_cast<double>(state.sums[j])));
            output[2 * j + 1] = static_cast<float>(state.sums[j]) / state.count;
        }
    }
};

extern "C" __declspec(dllexport)
int rolling_protocol_context(const Event* events, const uint8_t* flags,
                             int64_t size, int64_t tags, int64_t queries,
                             float* output, uint8_t* seen) {
    try {
        if (size < 0 || tags <= 0 || queries < 0) return -1;
        std::array<History, 2> histories = {History(60, tags), History(300, tags)};
        const int64_t width = 4 * tags;
        int64_t start = 0;
        while (start < size) {
            int64_t stop = start + 1;
            while (stop < size && events[stop].time == events[start].time) ++stop;
            for (auto& history : histories) history.expire(events, flags, start, events[start].time);
            for (int64_t i = start; i < stop; ++i) {
                const int64_t q = events[i].query;
                if (q < 0) continue;
                if (q >= queries || seen[q]) return -2;
                for (int h = 0; h < 2; ++h)
                    histories[h].query(events[i].source, output + q * width + h * 2 * tags);
                seen[q] = 1;
            }
            // No current-second row enters another current-second query.
            for (int64_t i = start; i < stop; ++i)
                for (auto& history : histories) history.add(events[i], flags + i * tags);
            start = stop;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

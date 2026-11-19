#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct Event {
    std::int64_t time;
    std::uint64_t source, destination;
    std::int64_t port;
    double failed, bytes, packets;
    std::int64_t query;
};
static_assert(sizeof(Event) == 64, "Python event ABI differs");

struct Pair {
    std::uint64_t source, destination;
    std::int64_t port;
    bool operator==(const Pair& rhs) const {
        return source == rhs.source && destination == rhs.destination && port == rhs.port;
    }
};
struct PairHash {
    std::size_t operator()(const Pair& key) const {
        std::uint64_t h = key.source;
        h ^= key.destination + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::uint64_t(key.port) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return std::size_t(h);
    }
};
struct Totals {
    std::int64_t count = 0;
    double failed = 0, bytes = 0, packets = 0;
    void update(const Event& event, int sign) {
        count += sign;
        failed += sign * event.failed;
        bytes += sign * event.bytes;
        packets += sign * event.packets;
    }
};
struct Node : Totals {
    std::unordered_map<std::uint64_t, std::int64_t> peers;
    std::unordered_map<std::int64_t, std::int64_t> ports;
};
struct History {
    std::int64_t horizon;
    std::size_t left = 0;
    std::unordered_map<std::uint64_t, Node> outgoing, incoming;
    std::unordered_map<Pair, Totals, PairHash> pairs;

    explicit History(std::int64_t seconds) : horizon(seconds) {}

    static void node_update(std::unordered_map<std::uint64_t, Node>& nodes,
                            std::uint64_t key, std::uint64_t peer,
                            const Event& event, int sign) {
        auto& node = nodes[key];
        node.update(event, sign);
        node.peers[peer] += sign;
        node.ports[event.port] += sign;
        if (node.peers[peer] == 0) node.peers.erase(peer);
        if (node.ports[event.port] == 0) node.ports.erase(event.port);
        if (node.count == 0) nodes.erase(key);
    }
    void update(const Event& event, int sign) {
        node_update(outgoing, event.source, event.destination, event, sign);
        node_update(incoming, event.destination, event.source, event, sign);
        Pair key{event.source, event.destination, event.port};
        auto& pair = pairs[key];
        pair.update(event, sign);
        if (pair.count == 0) pairs.erase(key);
    }
    static void describe(const Totals& total, std::size_t peers, std::size_t ports, float* result) {
        result[0] = float(std::log1p(double(total.count)));
        result[1] = float(std::log1p(double(peers)));
        result[2] = float(std::log1p(double(ports)));
        result[3] = float(total.failed / total.count);
        result[4] = float(std::log1p(std::max(0.0, total.bytes)));
        result[5] = float(std::log1p(std::max(0.0, total.packets)));
    }
    void query(const Event& event, float* result) const {
        auto out = outgoing.find(event.source);
        if (out != outgoing.end())
            describe(out->second, out->second.peers.size(), out->second.ports.size(), result);
        auto in = incoming.find(event.destination);
        if (in != incoming.end())
            describe(in->second, in->second.peers.size(), in->second.ports.size(), result + 6);
        auto pair = pairs.find(Pair{event.source, event.destination, event.port});
        if (pair != pairs.end()) describe(pair->second, 1, 1, result + 12);
    }
};

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

EXPORT int compute_context(const Event* events, std::size_t count,
                           std::size_t queries, float* output, std::uint8_t* seen) {
    try {
        std::array<History, 2> histories{History(60), History(300)};
        std::size_t start = 0;
        while (start < count) {
            const auto now = events[start].time;
            if (start && now < events[start - 1].time) return -1;
            std::size_t stop = start + 1;
            while (stop < count && events[stop].time == now) ++stop;
            for (auto& history : histories) {
                while (history.left < start && events[history.left].time < now - history.horizon)
                    history.update(events[history.left++], -1);
            }
            // Every event in a second sees the same strictly earlier history.
            for (auto i = start; i < stop; ++i) {
                const auto position = events[i].query;
                if (position < 0) continue;
                if (std::uint64_t(position) >= queries) return -2;
                std::array<float, 36> values{};
                for (std::size_t h = 0; h < 2; ++h)
                    histories[h].query(events[i], values.data() + 18 * h);
                auto* destination = output + std::size_t(position) * 36;
                if (seen[position]) {
                    for (std::size_t j = 0; j < 36; ++j)
                        if (std::abs(destination[j] - values[j]) > 1e-6f) return -4;
                }
                std::copy(values.begin(), values.end(), destination);
                seen[position] = 1;
            }
            for (auto i = start; i < stop; ++i)
                for (auto& history : histories) history.update(events[i], 1);
            start = stop;
        }
        return 0;
    } catch (...) { return -3; }
}

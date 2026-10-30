#pragma once

// Versioned, independently compressed observation blocks. Field layouts and
// strings are interned across the file; scalar tags/64-bit values are columns.
// The small record envelope remains JSON. No policy selection happens here.
// The engine defines signed Byte. Rename zlib's unsigned typedef locally.
#define Byte TrainingBinaryZlibByte
#include <zlib.h>
#undef Byte
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace training_binary {
struct Buffer {
    std::vector<uint8_t> tags;
    std::vector<uint64_t> values;
    uint32_t reserve(size_t count) {
        if (count > (512ULL*1024*1024/9) - values.size())
            throw std::runtime_error("Binary observation exceeds block memory budget");
        const auto start = static_cast<uint32_t>(values.size());
        values.resize(values.size()+count);
        tags.resize(values.size());
        return start;
    }
    void clear() { tags.clear(); values.clear(); }
};

struct Cursor {
    uint32_t current, end;
    uint32_t next() {
        if (current == end) throw std::runtime_error("Too many direct observation fields");
        return current++;
    }
    void finish() const {
        if (current != end) throw std::runtime_error("Missing direct observation fields");
    }
};

template<class Json> class Writer {
    std::ostream &out;
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view value) const noexcept { return std::hash<std::string_view>{}(value); }
    };
    std::unordered_map<std::string, uint32_t, StringHash, std::equal_to<>> strings;
    std::unordered_map<uint64_t, std::vector<uint32_t>> layouts;
    std::vector<std::vector<uint32_t>> schemas;
    std::vector<uint32_t> keys;
    std::vector<uint8_t> raw, compressed;
    Buffer scratch;
    Json newStrings, newSchemas;

    uint32_t intern(std::string_view text) {
        auto found = strings.find(text);
        if (found != strings.end()) return found->second;
        const auto id = static_cast<uint32_t>(strings.size());
        strings.emplace(std::string(text), id);
        newStrings.push_back(std::string(text));
        return id;
    }
    uint32_t finishSchema() {
        uint64_t hash = 1469598103934665603ULL;
        for (const auto key : keys) hash = (hash ^ key) * 1099511628211ULL;
        auto &candidates = layouts[hash];
        for (const auto id : candidates) if (schemas[id] == keys) return id;
        const auto id = static_cast<uint32_t>(schemas.size());
        schemas.push_back(keys);
        candidates.push_back(id);
        newSchemas.push_back(keys);
        return id;
    }
    void encode(Buffer &buffer, const Json &value, uint32_t node) {
        uint8_t tag = 0;
        uint64_t bits = 0;
        if (value.is_boolean()) tag = value.template get<bool>() ? 2 : 1;
        else if (value.is_number_unsigned()) { tag = 4; bits = value.template get<uint64_t>(); }
        else if (value.is_number_integer()) { tag = 3; bits = static_cast<uint64_t>(value.template get<int64_t>()); }
        else if (value.is_number_float()) {
            const double number = value.template get<double>();
            // Match the JSON exporter's null representation for nonfinite values.
            if (std::isfinite(number)) { tag = 5; std::memcpy(&bits, &number, 8); }
        } else if (value.is_string()) {
            tag = 6; bits = intern(value.template get_ref<const typename Json::string_t &>());
        } else if (value.is_array() || value.is_object()) {
            uint32_t layout = static_cast<uint32_t>(value.size());
            if (value.is_object()) {
                keys.clear();
                for (auto it = value.begin(); it != value.end(); ++it) keys.push_back(intern(it.key()));
                layout = finishSchema();
            }
            const auto first = buffer.reserve(value.size());
            tag = value.is_object() ? 8 : 7;
            bits = (static_cast<uint64_t>(layout) << 32) | first;
            uint32_t child = first;
            for (const auto &item : value) encode(buffer, item, child++);
        } else if (!value.is_null()) throw std::runtime_error("Unsupported binary observation value");
        buffer.tags[node] = tag; buffer.values[node] = bits;
    }
    void u32(uint32_t value) {
        const char bytes[] = {static_cast<char>(value), static_cast<char>(value>>8),
                              static_cast<char>(value>>16), static_cast<char>(value>>24)};
        out.write(bytes, 4);
    }
public:
    explicit Writer(std::ostream &stream) : out(stream) {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "IEEE doubles required");
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
            throw std::runtime_error("Binary writer requires little-endian host");
        out.write("GOBS1\r\n\032", 8);
        newStrings = Json::array(); newSchemas = Json::array();
    }
    void begin(Buffer &buffer) { buffer.clear(); buffer.reserve(1); }
    Cursor object(Buffer &buffer, uint32_t node, const char *const *fields, size_t count) {
        keys.clear();
        for (size_t i = 0; i < count; ++i) keys.push_back(intern(fields[i]));
        const auto layout = finishSchema();
        const auto first = buffer.reserve(count);
        buffer.tags[node] = 8; buffer.values[node] = (static_cast<uint64_t>(layout)<<32)|first;
        return {first, first+static_cast<uint32_t>(count)};
    }
    Cursor object(Buffer &buffer, uint32_t node, std::initializer_list<const char *> fields) {
        return object(buffer, node, fields.begin(), fields.size());
    }
    Cursor array(Buffer &buffer, uint32_t node, size_t count) {
        const auto first = buffer.reserve(count);
        buffer.tags[node] = 7; buffer.values[node] = (static_cast<uint64_t>(count)<<32)|first;
        return {first, first+static_cast<uint32_t>(count)};
    }
    template<class Number> void number(Buffer &buffer, uint32_t node, Number value) {
        uint64_t bits = 0;
        uint8_t tag;
        if constexpr (std::is_same_v<Number, bool>) tag = value ? 2 : 1;
        else if constexpr (std::is_floating_point_v<Number>) {
            const double expanded = value;
            tag = std::isfinite(expanded) ? 5 : 0;
            if (tag) std::memcpy(&bits, &expanded, 8);
        } else {
            static_assert(std::is_integral_v<Number>, "Direct observation requires a numeric scalar");
            tag = std::is_unsigned_v<Number> ? 4 : 3;
            bits = static_cast<uint64_t>(value);
        }
        buffer.tags[node] = tag; buffer.values[node] = bits;
    }
    void string(Buffer &buffer, uint32_t node, std::string_view value) {
        const auto id = intern(value);
        buffer.tags[node] = 6; buffer.values[node] = id;
    }
    void null(Buffer &buffer, uint32_t node) { buffer.tags[node] = 0; buffer.values[node] = 0; }
    // For the existing sparse command receipts and optional reference path.
    void copyJson(Buffer &buffer, uint32_t node, const Json &value) { encode(buffer, value, node); }

    void write(const Json &record, const Buffer *prepared = nullptr) {
        Json envelope = Json::object(), metadata = Json::object();
        scratch.clear(); raw.clear();
        for (auto it = record.begin(); it != record.end(); ++it)
            if (it.key() != "observation") metadata[it.key()] = it.value();
        if (!prepared && record.contains("observation")) {
            scratch.reserve(1);
            encode(scratch, record.at("observation"), 0);
        }
        const Buffer &buffer = prepared ? *prepared : scratch;
        envelope["record"] = std::move(metadata);
        envelope["strings"] = std::move(newStrings);
        envelope["schemas"] = std::move(newSchemas);
        const auto text = envelope.dump(-1, ' ', false, Json::error_handler_t::replace);
        if (text.size() > 16*1024*1024) throw std::runtime_error("Binary metadata exceeds block budget");
        raw.resize(buffer.tags.size()*9);
        if (!buffer.tags.empty()) {
            std::memcpy(raw.data(), buffer.tags.data(), buffer.tags.size());
            std::memcpy(raw.data()+buffer.tags.size(), buffer.values.data(), buffer.values.size()*8);
        }
        uLongf size = raw.empty() ? 0 : compressBound(static_cast<uLong>(raw.size()));
        compressed.resize(size);
        if (!raw.empty() && compress2(compressed.data(), &size, raw.data(),
                                     static_cast<uLong>(raw.size()), 1) != Z_OK)
            throw std::runtime_error("Cannot compress binary observation");
        const bool useCompression = size < raw.size();
        const auto *body = useCompression ? compressed.data() : raw.data();
        const auto bodySize = useCompression ? static_cast<uint32_t>(size) : static_cast<uint32_t>(raw.size());
        auto crc = crc32(0, reinterpret_cast<const Bytef *>(text.data()), static_cast<uInt>(text.size()));
        if (!raw.empty()) crc = crc32(crc, raw.data(), static_cast<uInt>(raw.size()));
        out.write("BLK1", 4);
        u32(static_cast<uint32_t>(text.size())); u32(static_cast<uint32_t>(buffer.tags.size()));
        u32(static_cast<uint32_t>(raw.size())); u32(bodySize); u32(crc); u32(useCompression ? 1 : 0);
        out.write(text.data(), text.size());
        if (bodySize) out.write(reinterpret_cast<const char *>(body), bodySize);
        if (!out) throw std::runtime_error("Cannot write binary training export");
        newStrings = Json::array(); newSchemas = Json::array();
    }
};
} // namespace training_binary

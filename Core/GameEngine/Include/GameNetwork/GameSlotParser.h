/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <array>
#include <cctype>
#include <set>
#include <string>

// The wire format does not escape delimiters in human names. Recognize the
// fixed numeric suffix and the complete slot list rather than splitting names.
namespace GameSlotParser
{
struct Slot
{
    std::string text;
    std::size_t nameEnd = 0;
};

inline bool decimal(const std::string& text)
{
    std::size_t i = !text.empty() && text[0] == '-' ? 1 : 0;
    if (i == text.size()) return false;
    for (; i < text.size(); ++i)
        if (text[i] < '0' || text[i] > '9') return false;
    return true;
}

inline bool record(const std::string& text, Slot& result)
{
    if (text == "X" || text == "O")
    {
        result.text = text;
        return true;
    }
    if (text.empty() || (text[0] != 'H' && text[0] != 'C')) return false;
    const bool human = text[0] == 'H';
    const std::size_t suffix = human ? 8 : 4;
    std::array<std::string, 8> fields;
    std::size_t end = text.size();
    for (std::size_t n = suffix; n > 0; --n)
    {
        if (end == 0) return false;
        const std::size_t comma = text.rfind(',', end - 1);
        if (comma == std::string::npos) return false;
        fields[n - 1] = text.substr(comma + 1, end - comma - 1);
        end = comma;
    }
    if (human)
    {
        if (fields[0].empty() || fields[0].size() > 8) return false;
        for (unsigned char c : fields[0])
            if (!std::isxdigit(c)) return false;
        if (!decimal(fields[1]) || fields[1][0] == '-') return false;
        if (fields[2].size() != 2) return false;
        for (char c : fields[2])
            if (c != 'T' && c != 'F') return false;
        for (std::size_t n = 3; n < suffix; ++n)
            if (!decimal(fields[n])) return false;
    }
    else
    {
        if (end != 2 || (text[1] != 'E' && text[1] != 'M' && text[1] != 'H')) return false;
        for (std::size_t n = 0; n < suffix; ++n)
            if (!decimal(fields[n])) return false;
    }
    result.text = text;
    result.nameEnd = end;
    return true;
}

template<std::size_t N>
bool splitFrom(const std::string& text, std::array<Slot, N>& slots,
               std::array<std::set<std::size_t>, N>& failed,
               std::size_t cursor, std::size_t index)
{
    if (index == N) return cursor == text.size();
    if (cursor == text.size()) return false;
    if (failed[index].count(cursor)) return false;
    for (std::size_t end = text.find(':', cursor); end != std::string::npos;
         end = text.find(':', end + 1))
    {
        if (record(text.substr(cursor, end - cursor), slots[index])
            && splitFrom(text, slots, failed, end + 1, index + 1)) return true;
        // Only human names can contain a colon.
        if (text[cursor] != 'H') break;
    }
    failed[index].insert(cursor);
    return false;
}

template<std::size_t N>
bool split(const std::string& text, std::array<Slot, N>& slots)
{
    // Repeated slot-like text in a name must not cause exponential backtracking.
    std::array<std::set<std::size_t>, N> failed;
    return splitFrom(text, slots, failed, 0, 0);
}
}

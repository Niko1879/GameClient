#pragma once
#include <algorithm>

// Observer-only bookkeeping. Never consult hidden world state while pruning.
namespace training_sightings {
struct Flags {
    unsigned id = 0;
    bool visible = false;
    bool persistent = true;
};

template<class Cache>
auto find(Cache &cache, unsigned id)
{
    return std::lower_bound(cache.begin(), cache.end(), id,
        [](const auto &entry, unsigned key) { return entry.id < key; });
}

// The caller supplies visibility at the lifecycle event, before dead/destroyed
// flags or callbacks change it. Last observation's visible flag is insufficient.
template<class A, class B, class Visible>
bool retireObserved(A &first, B &second, unsigned id, Visible visible)
{
    const auto a = find(first, id);
    const auto b = find(second, id);
    const bool inA = a != first.end() && a->id == id;
    const bool inB = b != second.end() && b->id == id;
    if ((!inA && !inB) || !visible()) return false;
    if (inA) first.erase(a);
    if (inB) second.erase(b);
    return true;
}

template<class Cache>
auto forgetHiddenTemporary(Cache &cache)
{
    const auto before = cache.size();
    cache.erase(std::remove_if(cache.begin(), cache.end(),
        [](const auto &entry) { return !entry.visible && !entry.persistent; }), cache.end());
    return before-cache.size();
}
}

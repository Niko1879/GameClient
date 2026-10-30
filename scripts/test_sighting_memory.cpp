// Compile standalone: exercises the same observer cache helpers as the engine.
#undef NDEBUG
#include "../GeneralsMD/Code/GameEngine/Include/Common/TrainingSightingMemory.h"
#include <cassert>
#include <vector>

struct JsonEntry : training_sightings::Flags { int frozen = 17; };
struct DirectEntry : training_sightings::Flags { int frozen = 23; };

int main()
{
    std::vector<JsonEntry> json;
    std::vector<DirectEntry> direct;
    auto add = [&](unsigned id, bool visible, bool persistent) {
        JsonEntry a; DirectEntry b;
        a.id = b.id = id; a.visible = b.visible = visible;
        a.persistent = b.persistent = persistent;
        json.push_back(a); direct.push_back(b);
    };
    add(1, true, true); add(2, false, true); add(3, true, false); add(4, false, false);
    const auto before = json; // Already captured samples must remain immutable.
    int visibilityQueries = 0;
    auto visible = [&] { ++visibilityQueries; return true; };
    assert(!training_sightings::retireObserved(json, direct, 99, visible));
    assert(visibilityQueries == 0); // Unknown objects do not query hidden state.
    assert(!training_sightings::retireObserved(json, direct, 1, [] { return false; }));
    assert(json.size() == 4 && direct.size() == 4); // Previously visible, now hidden.
    assert(training_sightings::retireObserved(json, direct, 2, visible));
    assert(json.size() == 3 && direct.size() == 3); // Previously hidden, now witnessed.
    assert(!training_sightings::retireObserved(json, direct, 2, visible));
    assert(visibilityQueries == 1); // Repeated death/destroy is harmless.
    assert(before.size() == 4 && before[1].id == 2 && before[1].frozen == 17);
    training_sightings::forgetHiddenTemporary(json);
    training_sightings::forgetHiddenTemporary(direct);
    assert(json.size() == 2 && direct.size() == 2);
    assert(json[0].id == 1 && json[1].id == 3); // Visible power aircraft remains.
    json[0].visible = direct[0].visible = false;
    json[1].visible = direct[1].visible = false;
    training_sightings::forgetHiddenTemporary(json);
    training_sightings::forgetHiddenTemporary(direct);
    assert(json.size() == 1 && direct.size() == 1 && json[0].id == 1);
    assert(json[0].frozen == 17 && direct[0].frozen == 23); // Ordinary hidden state frozen.
    // A separate viewer that cannot see the death retains the same known ID.
    auto otherJson = json; auto otherDirect = direct;
    assert(training_sightings::retireObserved(json, direct, 1, visible));
    assert(!training_sightings::retireObserved(otherJson, otherDirect, 1, [] { return false; }));
    assert(json.empty() && direct.empty() && otherJson.size() == 1);
    // A returning temporary object may be observed afresh under the same ID.
    add(3, true, false);
    training_sightings::forgetHiddenTemporary(json);
    assert(json.size() == 1 && json[0].id == 3);
}

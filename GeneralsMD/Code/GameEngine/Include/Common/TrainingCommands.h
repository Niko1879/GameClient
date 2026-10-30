#pragma once
// Upgrades, abilities (special powers, weapon commands, toggles), garrison,
// guard, scatter, waypoints, force-fire, repair, cancelling and promotions, shared by replay export and the live self-play bridge.
// Every rule here mirrors the engine's own UI: a command is legal for the
// policy exactly when its command button would be usable by a human player.
#include "GameNetwork/GeneralsOnline/json.hpp"
#include <string>
#include <vector>

class AIGroup;
class CommandButton;
class CommandSet;
class GameMessage;
class Object;
class Player;

namespace training_commands {
using Json = nlohmann::ordered_json;

// Static catalogue: canonical commands, per-template command menus and the
// per-faction shortcut/promotion menus. Equivalent buttons (same message and
// arguments) share one canonical name, so labels never depend on UI layout.
Json catalog();

// Owner-only capabilities of one object at this instant.
struct ObjectCommands {
    std::vector<const char *> available;                      // canonical command names
    std::vector<std::pair<const char *, float>> charging;     // not ready, with percent ready
    std::vector<std::pair<const char *, std::vector<unsigned>>> targets;  // object-target commands
    std::vector<unsigned> enterTargets;                       // containers this object may enter now
    std::vector<const char *> upgrades;                       // completed object upgrades
    unsigned occupants = 0, capacity = 0;                     // as a container
    unsigned containerID = 0;                                 // 0 = not inside a retained container
    bool canGuard = false;                                    // dispatchable guard orders (any mode)
    std::vector<unsigned> repairTargets;                      // structures this dozer may repair now
    std::vector<unsigned> repairDepots;                       // facilities that may repair this object now
};
// The owned and currently visible objects of one observation (the only legal
// command targets), plus the pools individual checks require: containers and
// dock facilities. Pools keep `all` order, so a filtered scan lists exactly the
// targets, in the same order, as a scan of every candidate.
struct Candidates {
    std::vector<Object *> all, containers, depots;
    void clear() { all.clear(); containers.clear(); depots.clear(); }
    // `dock`: the object has a DockUpdateInterface (callers already scan for it).
    void add(Object *object, bool dock);
};
// `set` is the object's current command set (callers cache the lookup).
void objectCommands(Object *object, const CommandSet *set, const Candidates &candidates, ObjectCommands &result);

struct PlayerCommands {
    std::vector<const char *> sciences, purchasableSciences, upgrades;
};
void playerCommands(Player *player, PlayerCommands &result);

// Canonical action for a replay message that this module covers, using the
// dispatcher's real selection. Returns false (with a reason) for messages it
// covers but cannot represent: unavailable, ambiguous or unknown commands.
bool covers(const GameMessage *message);
bool resolve(const GameMessage *message, const AIGroup *selection, Json &action, std::string &reason);

// Live bridge: validate a canonical action for `player` (throws on illegal
// input), then issue exactly the messages the UI would have issued.
void validate(int player, const Json &action);
void apply(int player, const Json &action);
}

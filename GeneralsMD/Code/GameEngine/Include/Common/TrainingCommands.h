// GeneralsAI training export: canonical commands (C++98 port of the Generals Online
// exporter's command layer, replay export only). Upgrades, abilities (special
// powers, weapon commands, toggles), garrison, guard, scatter, waypoints, force-fire,
// repair, cancelling and promotions. A command is legal for the policy exactly when
// its command button would be usable by a human player.
#pragma once

#include "Common/TrainingJson.h"
#include <string>
#include <utility>
#include <vector>

class AIGroup;
class CommandSet;
class GameMessage;
class Object;
class Player;

namespace training_commands {

// Owner-only capabilities of one object at this instant.
struct ObjectCommands {
	std::vector<const char *> available;                      // canonical command names
	std::vector<std::pair<const char *, float> > charging;    // not ready, with percent ready
	std::vector<std::pair<const char *, std::vector<unsigned> > > targets;  // object-target commands
	std::vector<unsigned> enterTargets;                       // containers this object may enter now
	std::vector<const char *> upgrades;                       // completed object upgrades
	unsigned occupants, capacity;                             // as a container
	unsigned containerID;                                     // 0 = not inside a retained container
	bool canGuard;                                            // dispatchable guard orders (any mode)
	std::vector<unsigned> repairTargets;                      // structures this dozer may repair now
	std::vector<unsigned> repairDepots;                       // facilities that may repair this object now
	ObjectCommands() : occupants(0), capacity(0), containerID(0), canGuard(false) {}
};

// The owned and currently visible objects of one observation (the only legal
// command targets), plus the containers and dock facilities among them, in `all` order.
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
bool resolve(const GameMessage *message, const AIGroup *selection, TrainingJson &action, std::string &reason);

} // namespace training_commands

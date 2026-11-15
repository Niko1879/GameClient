// GeneralsAI training export for retail Zero Hour 1.04 replays. See TrainingExporter.h.
// C++98 port of the Generals Online exporter's binary ("direct capture") path.
// Records: header, map, one sample per China player per cadence tick or command
// tick, footer. Observations are written straight into binary columns.
#include "PreRTS.h"
#include "Common/TrainingExporter.h"
#include "Common/TrainingBinary.h"
#include "Common/TrainingCommands.h"
#include "Common/TrainingJson.h"
#include "Common/ActionManager.h"
#include "Common/Energy.h"
#include "Common/GlobalData.h"
#include "Common/MessageStream.h"
#include "Common/Money.h"
#include "Common/NameKeyGenerator.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/PlayerTemplate.h"
#include "Common/BuildAssistant.h"
#include "Common/Recorder.h"
#include "Common/ThingTemplate.h"
#include "Common/ThingFactory.h"
#include "Common/Upgrade.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/Weapon.h"
#include "GameLogic/WeaponSet.h"
#include "GameLogic/Armor.h"
#include "GameLogic/ArmorSet.h"
#include "GameLogic/PartitionManager.h"
#include "GameClient/ControlBar.h"
#include "GameLogic/TerrainLogic.h"
#include "GameLogic/Module/AIUpdate.h"
#include "GameLogic/AI.h"
#include "GameLogic/Module/DozerAIUpdate.h"
#include "GameLogic/Module/SupplyTruckAIUpdate.h"
#include "GameLogic/Module/SupplyWarehouseDockUpdate.h"
#include "GameLogic/Module/StealthUpdate.h"
#include "GameLogic/Module/BodyModule.h"
#include "GameLogic/Module/ProductionUpdate.h"
#include "GameLogic/Module/ContainModule.h"
#include "GameLogic/Module/SpawnBehavior.h"
#include "GameLogic/Module/LifetimeUpdate.h"
#include "GameLogic/Module/PhysicsUpdate.h"
#include "GameLogic/Module/PoisonedBehavior.h"
#include "GameLogic/Module/DumbProjectileBehavior.h"
#include "GameLogic/Module/SpecialPowerModule.h"
#include "Common/SpecialPower.h"
#include "Common/SpecialPowerType.h"
#include "Common/GameAudio.h"
#include "Common/AudioEventInfo.h"
#include "Common/AudioEventRTS.h"
#include "GameClient/InGameUI.h"
#include "gitinfo.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
typedef TrainingJson Json;
typedef training_binary::Buffer DirectBuffer;
typedef training_binary::Cursor DirectCursor;

#define FIELDS(names) names, sizeof(names) / sizeof(names[0])

const int gridSize = 128;
std::string outputDirectory, temporaryPath, finalPath;
FILE *output = nullptr;
training_binary::Writer *binaryWriter = nullptr;
DirectBuffer pendingDirectBuffers[MAX_PLAYER_COUNT];
int intervalMilliseconds = 100;
bool active = false, failed = false, mapWritten = false;
// Keep the pre-command sample until dispatch supplies the actual result.
Json pendingSamples[MAX_PLAYER_COUNT];
int pendingPlayers[MAX_PLAYER_COUNT];
size_t pendingCount = 0;
struct PendingAction { const GameMessage *message; int player; size_t index; };
std::vector<PendingAction> pendingBuildCommands, pendingResolvedCommands;
unsigned samples = 0, commands = 0, waitSamples = 0, crcMessages = 0;
unsigned observedSightingRemovals[MAX_PLAYER_COUNT], temporarySightingsForgotten[MAX_PLAYER_COUNT];
bool hasLastSample[MAX_PLAYER_COUNT], hasLastCommand[MAX_PLAYER_COUNT];
unsigned lastSample[MAX_PLAYER_COUNT], lastCommand[MAX_PLAYER_COUNT];
Json submissions[MAX_PLAYER_COUNT];
Json pendingHeader;
// Scheduled quiet capture: a preparation controller on stdin/stdout names the
// quiet cadence frames its event schedule can select. Other quiet rows keep their
// metadata and sighting-memory updates but omit the observation ("light").
// Command rows and each player's first row always carry an observation.
bool scheduledQuietCapture = false;
struct QuietDirective {
	bool current;                  // until a command row, or the last of `frames` has passed
	std::vector<unsigned> frames;  // quiet frames, from the request on, that need one
	unsigned earliest;             // no later schedule state can select a quiet frame before this
	QuietDirective() : current(false), earliest(0) {}
};
QuietDirective quietDirectives[MAX_PLAYER_COUNT];

// The game runs the x87 FPU at 24-bit precision for determinism, so on this
// compiler every double computation would round to float. Use full precision
// only for the exporter's own arithmetic and number formatting (never around
// engine calls), and restore the game's mode on every exit path.
class ExactDoubles {
	unsigned m_previous;
public:
	ExactDoubles() : m_previous(_controlfp(0, 0)) { _controlfp(_PC_53, _MCW_PC); }
	~ExactDoubles() { _controlfp(m_previous, _MCW_PC); }
};

double secondsFor(unsigned ticks)
{
	ExactDoubles exact;
	return static_cast<double>(ticks) / static_cast<int>(LOGICFRAMES_PER_SECOND);
}

Json U(unsigned __int64 value) { return Json::unsignedInteger(value); }
Json I(__int64 value) { return Json::integer(value); }
Json R(double value) { return Json::real(value); }
Json S(const char *value) { return Json::string(value); }
Json B(bool value) { return Json::boolean(value); }

Json position(const Coord3D *p)
{
	Json result = Json::array();
	result.push(R(p->x)); result.push(R(p->y)); result.push(R(p->z));
	return result;
}

void write(const Json &record, const DirectBuffer *observation)
{
	ExactDoubles exact;	// decimal formatting of doubles
	binaryWriter->write(record.dump(), observation);
}

void flushSamples()
{
	bool observedRow = false;
	for (size_t i = 0; i < pendingCount; ++i) {
		const int player = pendingPlayers[i];
		write(pendingSamples[player], &pendingDirectBuffers[player]);
		observedRow = observedRow || !pendingSamples[player].contains("light");
	}
	// Let the capture controller validate observed rows while the game runs on,
	// instead of only once the engine stops to ask it for a directive.
	if (scheduledQuietCapture && observedRow) {
		if (fflush(output) != 0) throw std::runtime_error("Cannot flush training export for the capture controller");
		printf("GAIX {\"event\":\"rows\"}\n");
		fflush(stdout);
	}
	pendingCount = 0;
	pendingBuildCommands.clear();
	pendingResolvedCommands.clear();
}

// Reads `"key": <unsigned>` from a controller reply.
unsigned replyNumber(const std::string &reply, const char *key)
{
	const std::string quoted = std::string("\"") + key + "\"";
	size_t at = reply.find(quoted);
	if (at == std::string::npos) throw std::runtime_error("Capture directive lacks a field");
	at = reply.find(':', at + quoted.size());
	if (at == std::string::npos) throw std::runtime_error("Malformed capture directive");
	char *end = nullptr;
	const unsigned long value = strtoul(reply.c_str() + at + 1, &end, 10);
	if (end == reply.c_str() + at + 1) throw std::runtime_error("Malformed capture directive");
	return static_cast<unsigned>(value);
}

// Ask the controller which quiet frames need an observation only when the answer
// can matter. A directive stays valid until a command row or its last frame has
// passed: quiet rows it named cannot change it. Commands only move the schedule
// later, so nothing before the first frame of the latest answer can be selected.
// Every earlier row of this player is already written (flushSamples runs first
// each tick).
bool quietObservationWanted(int player, unsigned frame)
{
	QuietDirective &directive = quietDirectives[player];
	if (directive.current && !directive.frames.empty() && frame > directive.frames.back())
		directive.current = false;
	if (!directive.current) {
		if (frame < directive.earliest) return false;
		if (fflush(output) != 0) throw std::runtime_error("Cannot flush training export for the capture controller");
		printf("GAIX {\"event\":\"capture\",\"player\":%d,\"frame\":%u}\n", player, frame);
		fflush(stdout);
		std::string reply;
		char chunk[1024];
		while (reply.empty() || reply[reply.size() - 1] != '\n') {
			if (!fgets(chunk, sizeof(chunk), stdin)) throw std::runtime_error("Capture controller closed");
			reply += chunk;
		}
		if (static_cast<int>(replyNumber(reply, "player")) != player || replyNumber(reply, "frame") != frame)
			throw std::runtime_error("Capture directive does not answer this request");
		size_t at = reply.find("\"frames\"");
		if (at == std::string::npos || (at = reply.find('[', at)) == std::string::npos)
			throw std::runtime_error("Capture directive lacks frames");
		directive.frames.clear();
		const char *p = reply.c_str() + at + 1;
		for (;;) {
			while (*p == ' ' || *p == ',') ++p;
			if (*p == ']') break;
			char *end = nullptr;
			const unsigned long wanted = strtoul(p, &end, 10);
			if (end == p) throw std::runtime_error("Malformed capture directive frames");
			if (wanted < frame) throw std::runtime_error("Capture directive names a past frame");
			directive.frames.push_back(static_cast<unsigned>(wanted));
			p = end;
		}
		// No frames: the episode is finished, so no quiet row can be selected again.
		directive.earliest = directive.frames.empty() ? 0xffffffffu : directive.frames.front();
		directive.current = true;
	}
	return std::find(directive.frames.begin(), directive.frames.end(), frame) != directive.frames.end();
}

bool hasAttackMoveCommand(const CommandSet *set)
{
	if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
		const CommandButton *button = set->getCommandButton(i);
		if (button && button->getCommandType() == GUI_COMMAND_ATTACK_MOVE) return true;
	}
	return false;
}

bool retainedEntityTemplate(const ThingTemplate *t)
{
	// Classification uses static template flags, not hidden object state.
	// Foliage is cosmetic observation clutter, including optimized trees.
	if (t->isKindOf(KINDOF_SHRUBBERY) || t->isKindOf(KINDOF_OPTIMIZED_TREE)) return false;
	// Short-lived weapons/FX must never enter persistent unit sighting memory.
	if (t->isKindOf(KINDOF_PROJECTILE) || t->isKindOf(KINDOF_SMALL_MISSILE)
		|| t->isKindOf(KINDOF_INERT) || t->isKindOf(KINDOF_CLEANUP_HAZARD)
		|| t->isKindOf(KINDOF_HULK) || t->isKindOf(KINDOF_PARACHUTE)
		|| t->isKindOf(KINDOF_DRAWABLE_ONLY) || t->isKindOf(KINDOF_WAVE_EFFECT)
		|| t->isKindOf(KINDOF_WAVEGUIDE) || t->isKindOf(KINDOF_BLAST_CRATER)) return false;
	// Fire walls, dying-infantry effects, debris and invisible power markers
	// are unattackable objects. Keep unattackable structures/bridges.
	return !t->isKindOf(KINDOF_UNATTACKABLE) || t->isKindOf(KINDOF_STRUCTURE)
		|| t->isKindOf(KINDOF_BRIDGE);
}

bool retainedNeutralTemplate(const ThingTemplate *t)
{
	// Objects of non-playable sides are kept only for gameplay roles: economy,
	// capture, garrison/transport, repair, crates, hazards, crossings and armed
	// faction actors. Static template data only.
	static const KindOfType roles[] = {KINDOF_SUPPLY_SOURCE, KINDOF_SUPPLY_SOURCE_ON_PREVIEW,
		KINDOF_TECH_BUILDING, KINDOF_CAPTURABLE, KINDOF_CASH_GENERATOR, KINDOF_REPAIR_PAD, KINDOF_HEAL_PAD,
		KINDOF_CRATE, KINDOF_MINE, KINDOF_BOOBY_TRAP, KINDOF_DEMOTRAP, KINDOF_BRIDGE, KINDOF_LANDMARK_BRIDGE,
		KINDOF_BRIDGE_TOWER, KINDOF_GARRISONABLE_UNTIL_DESTROYED};
	for (size_t r = 0; r < sizeof(roles) / sizeof(roles[0]); ++r)
		if (t->isKindOf(roles[r])) return true;
	// Civilian cars, boats, aircraft and trains carry CAN_ATTACK for ambient
	// behaviour; only faction-side armed objects are actors.
	if (t->isKindOf(KINDOF_CAN_ATTACK) && t->getDefaultOwningSide() != "Civilian") return true;
	const ModuleInfo &modules = t->getBehaviorModuleInfo();
	for (int i = 0; i < modules.getCount(); ++i)
		if (strstr(modules.getNthName(i).str(), "Contain")) return true;
	return false;
}

// Templates are static during a game, and each sample scans every object at least twice.
struct RetainedTemplate {
	const ThingTemplate *type;
	bool entity, neutral;
	int dock;
	RetainedTemplate() : type(nullptr), entity(false), neutral(false), dock(-1) {}
};
std::vector<RetainedTemplate> retainedTemplates(4096);

bool retainedEntity(const Object *o)
{
	// Ownership is observable; playable sides keep every retained template.
	const Player *owner = o->getControllingPlayer();
	const ThingTemplate *type = o->getTemplate();
	RetainedTemplate &known = retainedTemplates[(reinterpret_cast<size_t>(type) >> 4) & 4095];
	if (known.type != type) {
		known.type = type;
		known.entity = retainedEntityTemplate(type);
		known.neutral = known.entity && retainedNeutralTemplate(type);
		known.dock = -1;
	}
	return known.entity && ((owner && owner->isPlayableSide()) || known.neutral);
}

// An object's modules come from its template, so one object answers for all.
bool hasDockInterface(Object *o)
{
	const ThingTemplate *type = o->getTemplate();
	RetainedTemplate &known = retainedTemplates[(reinterpret_cast<size_t>(type) >> 4) & 4095];
	if (known.type != type || known.dock < 0) {
		bool dock = false;
		for (BehaviorModule **module = o->getBehaviorModules(); *module; ++module)
			if ((*module)->getDockUpdateInterface()) { dock = true; break; }
		if (known.type != type) return dock;
		known.dock = dock ? 1 : 0;
	}
	return known.dock != 0;
}

bool persistentSightingTemplate(const ThingTemplate *t)
{
	// Re-export fix 5: the observation holds only what the player's screen shows. Hidden mobile
	// units are not kept; the game draws fog ghosts only for immobile objects (KINDOF_IMMOBILE).
	if (!t->isKindOf(KINDOF_IMMOBILE)) return false;
	const ModuleInfo &modules = t->getBehaviorModuleInfo();
	for (int i = 0; i < modules.getCount(); ++i) {
		const AsciiString &name = modules.getNthName(i);
		// Delivery/strike aircraft and the circling gunship are useful while
		// visible, but their old positions do not belong in persistent memory.
		if (name == "DeliverPayloadAIUpdate" || name == "SpectreGunshipUpdate") return false;
		if (name == "LifetimeUpdate") {
			const LifetimeUpdateModuleData *lifetime = static_cast<const LifetimeUpdateModuleData *>(modules.getNthData(i));
			if (lifetime->m_minFrames || lifetime->m_maxFrames) return false;
		}
	}
	return true;
}

// ---- Observation v11 (re-export batch, user 15 Nov 2026) ----------------------------------------
// As the Generals Online exporter: everything here is what a player perceives right now; none of it
// enters sighting memory.

// Transient battlefield effects: projectiles and missiles in flight, and ground hazards (fire
// fields and walls, toxin/anthrax, radiation). retainedEntityTemplate excludes all of these.
enum { EFFECT_NONE = 0, EFFECT_PROJECTILE = 1, EFFECT_HAZARD = 2 };
int effectTemplateKind(const ThingTemplate *t)
{
	if (t->isKindOf(KINDOF_PROJECTILE) || t->isKindOf(KINDOF_SMALL_MISSILE)) return EFFECT_PROJECTILE;
	if (t->isKindOf(KINDOF_CLEANUP_HAZARD)) return EFFECT_HAZARD;
	if (!t->isKindOf(KINDOF_UNATTACKABLE) || t->isKindOf(KINDOF_STRUCTURE) || t->isKindOf(KINDOF_BRIDGE)) return EFFECT_NONE;
	// Unattackable damage areas (Dragon fire walls, napalm, firestorms) burn through a FireWeaponUpdate.
	const ModuleInfo &modules = t->getBehaviorModuleInfo();
	for (int i = 0; i < modules.getCount(); ++i)
		if (modules.getNthName(i) == "FireWeaponUpdate") return EFFECT_HAZARD;
	return EFFECT_NONE;
}
struct EffectTemplate {
	const ThingTemplate *type;
	int kind;
	EffectTemplate() : type(nullptr), kind(EFFECT_NONE) {}
};
std::vector<EffectTemplate> effectTemplates(4096);
int effectKind(const Object *o)
{
	const ThingTemplate *type = o->getTemplate();
	EffectTemplate &known = effectTemplates[(reinterpret_cast<size_t>(type) >> 4) & 4095];
	if (known.type != type) { known.type = type; known.kind = effectTemplateKind(type); }
	return known.kind;
}
const char *effectName(int kind) { return kind == EFFECT_PROJECTILE ? "projectile" : "hazard"; }

// World units per logic frame (defined below as objectVelocity).
Coord3D objectVelocity(const Object *object);
inline Coord3D objectVelocityOf(const Object *object) { return objectVelocity(object); }
// Ballistic shells move by position along a flight path (any physics they carry is not their motion).
Coord3D effectVelocity(const Object *o)
{
	static const NameKeyType key = NAMEKEY("DumbProjectileBehavior");
	const DumbProjectileBehavior *shell = static_cast<const DumbProjectileBehavior *>(o->findUpdateModule(key));
	if (shell) return shell->getFlightVelocity();
	return objectVelocityOf(o);
}

// Muzzle flashes and tracers: a shot within the last half second, by the object or by an occupant
// firing out of it (garrisons, fire ports). Hidden occupants stay hidden; only the flash shows.
bool firingVisibly(const Object *o, unsigned frame, bool occupants)
{
	if (o->testStatus(OBJECT_STATUS_IS_FIRING_WEAPON)) return true;
	const unsigned window = LOGICFRAMES_PER_SECOND / 2;
	for (int slot = PRIMARY_WEAPON; slot < WEAPONSLOT_COUNT; ++slot) {
		const Weapon *weapon = o->getWeaponInWeaponSlot(static_cast<WeaponSlotType>(slot));
		const UnsignedInt last = weapon ? weapon->getLastShotFrame() : 0;
		if (last && last <= frame && frame - last <= window) return true;
	}
	if (occupants) {
		const ContainModuleInterface *contain = o->getContain();
		const ContainedItemsList *items = contain ? contain->getContainedItemsList() : nullptr;
		if (items)
			for (ContainedItemsList::const_iterator it = items->begin(); it != items->end(); ++it)
				if (firingVisibly(*it, frame, false)) return true;
	}
	return false;
}
// The green poison tint and the flames on a burning unit.
bool poisonedVisibly(const Object *o)
{
	static const NameKeyType key = NAMEKEY("PoisonedBehavior");
	const PoisonedBehavior *poison = static_cast<const PoisonedBehavior *>(o->findUpdateModule(key));
	return poison && poison->isPoisoned();
}
bool burningVisibly(const Object *o) { return o->testStatus(OBJECT_STATUS_AFLAME) ? true : false; }

// Special-power launch sounds (SpecialPowerModule::aboutToDoSpecialPower): the sound at the
// launcher, the sound at the target, and EVA's superweapon announcement. Each is heard by the
// audience its AudioEventInfo names (AudioManager::shouldPlayLocally), evaluated per player. A cue
// lasts CUE_SECONDS; its position is given only where the listener has vision.
struct PowerCue {
	unsigned frame;
	int owner;
	std::string power;
	bool hasTarget, announced;
	Coord3D source, target;
	unsigned sourceAudience, targetAudience;
	PowerCue() : frame(0), owner(-1), hasTarget(false), announced(false), sourceAudience(0), targetAudience(0) {
		source.x = source.y = source.z = target.x = target.y = target.z = 0;
	}
};
std::vector<PowerCue> powerCues;
const unsigned CUE_SECONDS = 5;
unsigned soundAudience(const AudioEventRTS *sound)
{
	if (!sound || sound->getEventName().isEmpty() || !TheAudio) return 0;
	const AudioEventInfo *info = TheAudio->findAudioEventInfo(sound->getEventName());
	if (!info) return 0;
	// shouldPlayLocally treats a sound with no audience restriction as heard by everyone.
	return (info->m_type & (ST_PLAYER | ST_ALLIES | ST_ENEMIES | ST_EVERYONE)) ? info->m_type : ST_EVERYONE;
}
bool hears(unsigned audience, int ownerIndex, const Player *listener)
{
	const Player *owner = ThePlayerList->getNthPlayer(ownerIndex);
	if (!audience || !owner || !listener || !listener->getDefaultTeam()) return false;
	if (audience & ST_EVERYONE) return true;
	if (audience & ST_PLAYER) return owner == listener;
	if (audience & ST_ALLIES) return owner != listener && owner->getRelationship(listener->getDefaultTeam()) == ALLIES;
	if (audience & ST_ENEMIES) return owner->getRelationship(listener->getDefaultTeam()) == ENEMIES;
	return false;
}
bool cueCurrent(const PowerCue &cue, unsigned frame)
{
	return cue.frame <= frame && frame - cue.frame <= CUE_SECONDS * LOGICFRAMES_PER_SECOND;
}

void taskStatus(Json &result, const char *status)
{
	if (!result.isObject()) result = Json::object();
	result.erase("id");
	result.set("status", S(status));
}

void taskReference(Json &result, ObjectID id, Player *owner)
{
	if (id == INVALID_ID) { taskStatus(result, "none"); return; }
	Object *target = TheGameLogic->findObjectByID(id);
	// Missing, dead and hidden objects share one response. Never refresh a
	// hidden target's attributes through the owner's internal AI pointer.
	if (target && !target->isEffectivelyDead() && retainedEntity(target)
		&& (target->getControllingPlayer() == owner || TrainingObjectVisibleTo(target, owner->getPlayerIndex()))) {
		if (!result.isObject()) result = Json::object();
		result.set("status", S("current"));
		result.set("id", U(static_cast<unsigned>(id)));
		return;
	}
	taskStatus(result, "unavailable");
}

struct ObservationLookups {
	std::vector<std::pair<const AsciiString *, const CommandSet *> > commandSets;
	// Docks that pass TrainingCanDock's target test (alive, owned or visible).
	std::vector<Object *> docks;
	// Owned and currently visible objects: the only legal command targets.
	training_commands::Candidates commandTargets;
	training_commands::ObjectCommands commands;
	training_commands::PlayerCommands player;
} observationLookups;

const CommandSet *commandSetFor(Object *object, ObservationLookups &lookups)
{
	const AsciiString &name = object->getCommandSetString();
	for (size_t i = 0; i < lookups.commandSets.size(); ++i)
		if (*lookups.commandSets[i].first == name) return lookups.commandSets[i].second;
	const CommandSet *set = TheControlBar->findCommandSet(name);
	lookups.commandSets.push_back(std::make_pair(&name, set));
	return set;
}

// World units per logic frame; objects without physics do not move this way.
Coord3D objectVelocity(const Object *object)
{
	const PhysicsBehavior *physics = object->getPhysics();
	if (physics) return *physics->getVelocity();
	Coord3D zero;
	zero.x = zero.y = zero.z = 0;
	return zero;
}

// Boxes left in a supply dock, pile or warehouse; 0 for every other object.
int supplyBoxes(const Object *object)
{
	if (!object->isKindOf(KINDOF_SUPPLY_SOURCE)) return 0;
	static const NameKeyType key = NAMEKEY("SupplyWarehouseDockUpdate");
	const SupplyWarehouseDockUpdate *warehouse = static_cast<const SupplyWarehouseDockUpdate *>(object->findUpdateModule(key));
	return warehouse ? warehouse->getBoxesStored() : 0;
}

// Frames until the current weapon can fire again; 0 when ready or unarmed.
unsigned weaponCooldownTicks(const Object *object, unsigned frame)
{
	const Weapon *weapon = object->getCurrentWeapon();
	const UnsignedInt next = weapon ? weapon->getPossibleNextShotFrame() : 0;
	return next > frame ? next - frame : 0;
}

// Frames since this object last took damage; -1 if it never has.
int ticksSinceDamage(const BodyModuleInterface *body, unsigned frame)
{
	// Null when never damaged; retail-compatible builds return the raw sentinel instead.
	const UnsignedInt *timestamp = body ? body->getLastDamageTimestamp() : nullptr;
	const UnsignedInt last = timestamp ? *timestamp : 0;
	return last == 0 || last == 0xffffffff || last > frame ? -1 : static_cast<int>(frame - last);
}

// Weapon and armour identity (re-export fix 4; the same contract as the main engine's exporter,
// GameClient scripts/training_data_review.md). Templates get a per-game numeric id on first sight;
// each player's stream receives a template's stats once, in the observation where it first appears.
// Stats are read without logic random numbers (never WeaponTemplate::getDelayBetweenShots).
struct TemplateCatalog {
	std::vector<const WeaponTemplate *> weapons;
	std::vector<const ArmorTemplate *> armors;
	size_t weaponsSent[MAX_PLAYER_COUNT];
	size_t armorsSent[MAX_PLAYER_COUNT];
	TemplateCatalog() { reset(); }
	void reset() {
		weapons.clear(); armors.clear();
		for (int p = 0; p < MAX_PLAYER_COUNT; ++p) weaponsSent[p] = armorsSent[p] = 0;
	}
};
TemplateCatalog armsCatalog;

int weaponId(const WeaponTemplate *item)
{
	if (!item) return -1;
	std::vector<const WeaponTemplate *> &list = armsCatalog.weapons;
	for (size_t i = 0; i < list.size(); ++i) if (list[i] == item) return static_cast<int>(i);
	list.push_back(item);
	return static_cast<int>(list.size() - 1);
}

int armorId(const ArmorTemplate *item)
{
	if (!item) return -1;
	std::vector<const ArmorTemplate *> &list = armsCatalog.armors;
	for (size_t i = 0; i < list.size(); ++i) if (list[i] == item) return static_cast<int>(i);
	list.push_back(item);
	return static_cast<int>(list.size() - 1);
}

// An owned unit's current weapons, armour and fire bonus; any other unit its type's default
// weapon and armour sets with no bonus (its upgrades are not observable; veterancy is exported).
struct EntityArms {
	int weapons[WEAPONSLOT_COUNT];
	int armor;
	float damage, range, rate;
	EntityArms() : armor(-1), damage(1), range(1), rate(1) {
		for (int s = 0; s < WEAPONSLOT_COUNT; ++s) weapons[s] = -1;
	}
};

EntityArms entityArms(const Object *object, bool own)
{
	EntityArms arms;
	const ThingTemplate *type = object->getTemplate();
	int slot;
	if (own) {
		for (slot = 0; slot < WEAPONSLOT_COUNT; ++slot) {
			const Weapon *weapon = object->getWeaponInWeaponSlot(static_cast<WeaponSlotType>(slot));
			arms.weapons[slot] = weaponId(weapon ? weapon->getTemplate() : nullptr);
		}
		const Weapon *current = object->getCurrentWeapon();
		if (current) {
			WeaponBonus bonus;
			current->computeTrainingBonus(object, bonus);
			arms.damage = bonus.getField(WeaponBonus::DAMAGE);
			arms.range = bonus.getField(WeaponBonus::RANGE);
			arms.rate = bonus.getField(WeaponBonus::RATE_OF_FIRE);
		}
		ArmorSetFlags flags;
		BodyModuleInterface *body = object->getBodyModule();
		if (body)
			for (int set = 0; set < ARMORSET_COUNT; ++set)
				if (body->testArmorSetFlag(static_cast<ArmorSetType>(set))) flags.set(static_cast<ArmorSetType>(set), 1);
		const ArmorTemplateSet *armor = type->findArmorTemplateSet(flags);
		arms.armor = armorId(armor ? armor->getArmorTemplate() : nullptr);
	} else {
		const WeaponTemplateSet *weapons = type->findWeaponTemplateSet(WeaponSetFlags());
		for (slot = 0; slot < WEAPONSLOT_COUNT; ++slot)
			arms.weapons[slot] = weaponId(weapons ? weapons->getNth(static_cast<WeaponSlotType>(slot)) : nullptr);
		const ArmorTemplateSet *armor = type->findArmorTemplateSet(ArmorSetFlags());
		arms.armor = armorId(armor ? armor->getArmorTemplate() : nullptr);
	}
	return arms;
}

TrainingJson argument(const GameMessage *message, int index)
{
	const GameMessageArgumentType *a = message->getArgument(index);
	Json j = Json::object();
	j.set("type", I(static_cast<int>(message->getArgumentDataType(index))));
	switch (message->getArgumentDataType(index)) {
	case ARGUMENTDATATYPE_INTEGER: j.set("value", I(a->integer)); break;
	case ARGUMENTDATATYPE_REAL: j.set("value", R(a->real)); break;
	case ARGUMENTDATATYPE_BOOLEAN: j.set("value", B(a->boolean ? true : false)); break;
	case ARGUMENTDATATYPE_OBJECTID: j.set("value", U(static_cast<unsigned>(a->objectID))); break;
	case ARGUMENTDATATYPE_DRAWABLEID: j.set("value", U(static_cast<unsigned>(a->drawableID))); break;
	case ARGUMENTDATATYPE_TEAMID: j.set("value", U(a->teamID)); break;
	case ARGUMENTDATATYPE_LOCATION: j.set("value", position(&a->location)); break;
	case ARGUMENTDATATYPE_PIXEL: {
		Json pixel = Json::array();
		pixel.push(I(a->pixel.x)); pixel.push(I(a->pixel.y));
		j.set("value", pixel);
		break;
	}
	case ARGUMENTDATATYPE_PIXELREGION: {
		Json region = Json::array();
		region.push(I(a->pixelRegion.lo.x)); region.push(I(a->pixelRegion.lo.y));
		region.push(I(a->pixelRegion.hi.x)); region.push(I(a->pixelRegion.hi.y));
		j.set("value", region);
		break;
	}
	case ARGUMENTDATATYPE_TIMESTAMP: j.set("value", U(a->timestamp)); break;
	case ARGUMENTDATATYPE_WIDECHAR: j.set("value", U(static_cast<unsigned>(a->wChar))); break;
	default: throw std::runtime_error("Unknown command argument type");
	}
	return j;
}

Json actionsFor(int player)
{
	Json actions = Json::array();
	for (GameMessage *m = TheCommandList->getFirstMessage(); m; m = m->next()) {
		const GameMessage::Type type = m->getType();
		if (m->getPlayerIndex() != player || type <= GameMessage::MSG_BEGIN_NETWORK_MESSAGES
			|| type >= GameMessage::MSG_END_NETWORK_MESSAGES || type == GameMessage::MSG_LOGIC_CRC
			|| type == GameMessage::MSG_SET_REPLAY_CAMERA) continue;
		Json action = Json::object();
		action.set("type_id", I(static_cast<int>(type)));
		action.set("name", S(m->getCommandAsString()));
		Json arguments = Json::array();
		for (int i = 0; i < m->getArgumentCount(); ++i) arguments.push(argument(m, i));
		action.set("arguments", arguments);
		if (type == GameMessage::MSG_DOZER_CONSTRUCT) {
			Json result = Json::object();
			result.set("accepted", B(false));
			result.set("status", S("not_dispatched"));
			action.set("build_result", result);
			PendingAction pending = {m, player, actions.size()};
			pendingBuildCommands.push_back(pending);
		}
		if (training_commands::covers(m)) {
			Json result = Json::object();
			result.set("resolved", B(false));
			result.set("reason", S("not_dispatched"));
			action.set("command_result", result);
			PendingAction pending = {m, player, actions.size()};
			pendingResolvedCommands.push_back(pending);
		}
		actions.push(action);
	}
	return actions;
}

Region3D extent()
{
	Region3D region;
	TheTerrainLogic->getExtent(&region);
	return region;
}

Coord3D gridPoint(const Region3D &region, int x, int y)
{
	Coord3D p;
	p.x = region.lo.x + (x + 0.5f) * (region.hi.x - region.lo.x) / gridSize;
	p.y = region.lo.y + (y + 0.5f) * (region.hi.y - region.lo.y) / gridSize;
	p.z = 0;
	return p;
}

Json mapFeatures()
{
	const Region3D region = extent();
	Json heights = Json::array(), cliffs = Json::array();
	for (int y = 0; y < gridSize; ++y) for (int x = 0; x < gridSize; ++x) {
		const Coord3D p = gridPoint(region, x, y);
		heights.push(R(TheTerrainLogic->getGroundHeight(p.x, p.y)));
		cliffs.push(B(TheTerrainLogic->isCliffCell(p.x, p.y) ? true : false));
	}
	Json sources = Json::array();
	// Resource locations are static, public map information. Do not export
	// remaining supplies, player-owned sources, or any opponent objects here.
	for (Object *o = TheGameLogic->getFirstObject(); o; o = o->getNextObject())
		if (o->isKindOf(KINDOF_SUPPLY_SOURCE) && (!o->getControllingPlayer()
			|| o->getControllingPlayer()->getPlayerIndex() < 2)) {
			Json source = Json::object();
			source.set("id", U(static_cast<unsigned>(o->getID())));
			source.set("type", S(o->getTemplate()->getName().str()));
			source.set("position", position(o->getPosition()));
			sources.push(source);
		}
	Json result = Json::object();
	result.set("record", S("map"));
	result.set("grid_size", I(gridSize));
	result.set("supply_sources", sources);
	result.set("extent_low", position(&region.lo));
	result.set("extent_high", position(&region.hi));
	result.set("height", heights);
	result.set("cliff", cliffs);
	return result;
}

bool templateNameLess(const ThingTemplate *a, const ThingTemplate *b)
{
	return strcmp(a->getName().str(), b->getName().str()) < 0;
}

Json observationPolicy()
{
	std::vector<const ThingTemplate *> classes;
	for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate())
		classes.push_back(static_cast<const ThingTemplate *>(t->getFinalOverride()));
	std::stable_sort(classes.begin(), classes.end(), templateNameLess);
	Json excluded = Json::array(), visibleOnly = Json::array(), neutralExcluded = Json::array();
	Json effects = Json::object();
	for (size_t i = 0; i < classes.size(); ++i) {
		// Preserve the old map's sorted names and last-value-wins semantics.
		if (i + 1 < classes.size() && classes[i]->getName() == classes[i + 1]->getName()) continue;
		const int kind = effectTemplateKind(classes[i]);
		if (kind != EFFECT_NONE) effects.set(classes[i]->getName().str(), S(effectName(kind)));
		if (!retainedEntityTemplate(classes[i])) { excluded.push(S(classes[i]->getName().str())); continue; }
		if (!persistentSightingTemplate(classes[i])) visibleOnly.push(S(classes[i]->getName().str()));
		if (!retainedNeutralTemplate(classes[i])) neutralExcluded.push(S(classes[i]->getName().str()));
	}
	Json playable = Json::array();
	for (int p = 0; p < ThePlayerList->getPlayerCount(); ++p)
		if (ThePlayerList->getNthPlayer(p)->isPlayableSide()) playable.push(I(p));
	Json result = Json::object();
	result.set("version", I(11));
	result.set("excluded_templates", excluded);
	// v11: excluded entity templates that appear, visible-now only, in the observation's effects list.
	result.set("effect_templates", effects);
	result.set("current_perception", S("effects (projectiles, ground hazards) owned or visible now; entity firing/poisoned/burning on own and visible units, false on fog ghosts; public superweapon timers; special-power sound cues audible to the player for 5 s, positioned only where the player has vision"));
	result.set("visible_only_templates", visibleOnly);
	// These templates are omitted unless a playable side controls the object.
	result.set("neutral_excluded_templates", neutralExcluded);
	result.set("playable_players", playable);
	result.set("visibility", S("explicit-player shroud and engine stealth look; visible foundations retained; nonowned occupants and disguised identities omitted"));
	result.set("sighting_memory", S("witnessed removals retired; delivery/strike and finite-lifetime objects visible only; ordinary hidden sightings frozen"));
	result.set("neutral_entities", S("non-playable sides keep supply, tech/capturable, garrison/transport, repair, crate, hazard, bridge and armed faction-side objects only; civilian traffic is scenery"));
	return result;
}

Json templateCatalog()
{
	Json catalog = Json::object();
	char key[16];
	for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate()) {
		sprintf(key, "%u", static_cast<unsigned>(t->getTemplateID()));
		// Template IDs are unique, so no duplicate-key search is needed.
		catalog.append(key, S(t->getName().str()));
	}
	return catalog;
}

// The China factions whose players are exported (13 Nov 2026: vanilla, Tank and Nuke; Infantry is
// too different to share one policy). Players of any other faction are only observed. Same list and
// order as the Generals Online exporter.
static const char *const EXPORTED_FACTIONS[] = {"FactionChina", "FactionChinaTankGeneral", "FactionChinaNukeGeneral"};
static const int EXPORTED_FACTION_COUNT = sizeof(EXPORTED_FACTIONS)/sizeof(EXPORTED_FACTIONS[0]);

static Bool exportedFaction(Player *p)
{
	if (!p || !p->getPlayerTemplate()) return FALSE;
	const char *name = p->getPlayerTemplate()->getName().str();
	for (int f = 0; f < EXPORTED_FACTION_COUNT; ++f)
		if (!strcmp(name, EXPORTED_FACTIONS[f])) return TRUE;
	return FALSE;
}

void mapRecord()
{
	// Replay setup loads map.ini after Begin. Resolve its final template
	// overrides before declaring the exact classes that the encoder excludes.
	pendingHeader.set("observation_policy", observationPolicy());
	// The exported players are known now: name their factions, in EXPORTED_FACTIONS order. A replay
	// with only vanilla China players keeps exactly "FactionChina" (set() replaces in place).
	std::string factions;
	for (int f = 0; f < EXPORTED_FACTION_COUNT; ++f) {
		Bool present = FALSE;
		for (int i = 0; i < ThePlayerList->getPlayerCount() && !present; ++i) {
			Player *p = ThePlayerList->getNthPlayer(i);
			present = p && p->getPlayerTemplate() && !strcmp(p->getPlayerTemplate()->getName().str(), EXPORTED_FACTIONS[f]);
		}
		if (present) {
			if (!factions.empty()) factions += ",";
			factions += EXPORTED_FACTIONS[f];
		}
	}
	if (!factions.empty()) pendingHeader.set("controlled_faction", S(factions.c_str()));
	write(pendingHeader, nullptr);
	pendingHeader = Json();
	write(mapFeatures(), nullptr);
	mapWritten = true;
}

// ---- direct observation writing ---------------------------------------------

struct DirectOutput {
	training_binary::Writer &writer;
	DirectBuffer &buffer;
	DirectOutput(training_binary::Writer &w, DirectBuffer &b) : writer(w), buffer(b) {}
	DirectCursor object(DirectCursor &parent, const char *const *fields, size_t count) {
		return writer.object(buffer, parent.next(), fields, count);
	}
	DirectCursor array(DirectCursor &parent, size_t count) { return writer.array(buffer, parent.next(), count); }
	// Overloads keep the original exporter's signed/unsigned/float tags.
	void put(DirectCursor &parent, bool value) { writer.putBool(buffer, parent.next(), value); }
	void put(DirectCursor &parent, int value) { writer.putInt(buffer, parent.next(), value); }
	void put(DirectCursor &parent, long value) { writer.putInt(buffer, parent.next(), value); }
	void put(DirectCursor &parent, unsigned int value) { writer.putUInt(buffer, parent.next(), value); }
	void put(DirectCursor &parent, unsigned long value) { writer.putUInt(buffer, parent.next(), value); }
	void put(DirectCursor &parent, float value) { writer.putReal(buffer, parent.next(), value); }
	void put(DirectCursor &parent, double value) { writer.putReal(buffer, parent.next(), value); }
	void put(DirectCursor &parent, const char *value) { writer.putString(buffer, parent.next(), value); }
	void put(DirectCursor &parent, const std::string &value) { writer.putString(buffer, parent.next(), value.c_str()); }
	void null(DirectCursor &parent) { writer.putNull(buffer, parent.next()); }
	void point(DirectCursor &parent, const Coord3D *p) {
		DirectCursor xyz = array(parent, 3);
		put(xyz, p->x); put(xyz, p->y); put(xyz, p->z);
	}
	void ids(DirectCursor &parent, const std::vector<unsigned> &values) {
		DirectCursor list = array(parent, values.size());
		for (size_t i = 0; i < values.size(); ++i) put(list, values[i]);
	}
	void names(DirectCursor &parent, const std::vector<const char *> &values) {
		DirectCursor list = array(parent, values.size());
		for (size_t i = 0; i < values.size(); ++i) put(list, values[i]);
	}
};

struct DirectReference {
	const char *status;
	ObjectID id;
	Object *target;
	DirectReference() : status("none"), id(INVALID_ID), target(nullptr) {}
};

DirectReference directReference(ObjectID id, Player *owner)
{
	DirectReference result;
	if (id == INVALID_ID) return result;
	Object *target = TheGameLogic->findObjectByID(id);
	result.id = id;
	if (target && !target->isEffectivelyDead() && retainedEntity(target)
		&& (target->getControllingPlayer() == owner || TrainingObjectVisibleTo(target, owner->getPlayerIndex()))) {
		result.status = "current";
		result.target = target;
	} else result.status = "unavailable";
	return result;
}

void writeReference(DirectOutput &out, DirectCursor &parent, const DirectReference &reference)
{
	static const char *const statusOnly[] = {"status"};
	static const char *const statusAndID[] = {"status", "id"};
	DirectCursor fields = reference.target ? out.object(parent, FIELDS(statusAndID)) : out.object(parent, FIELDS(statusOnly));
	out.put(fields, reference.status);
	if (reference.target) out.put(fields, static_cast<unsigned>(reference.id));
	fields.finish();
}

void directTask(DirectOutput &out, DirectCursor &parent, Object *object)
{
	AIUpdateInterface *ai = object->getAIUpdateInterface();
	Player *owner = object->getControllingPlayer();
	const char *kind = "unknown";
	bool moving = false, attacking = false, waiting = false, busy = false;
	bool point = false, path = false, targetTask = false;
	DirectReference target, combat, dock;
	const Coord3D *destination = nullptr;
	Coord3D savedDestination;
	SupplyTruckAIInterface *supply = nullptr;
	bool ferrying = false;
	int boxes = 0;
	if (ai) {
		switch (ai->getAIStateType()) {
		case AI_IDLE: kind = "idle"; break;
		case AI_MOVE_TO: case AI_MOVE_AND_TIGHTEN: kind = "move"; point = true; targetTask = true; break;
		case AI_FOLLOW_PATH: kind = "move"; path = true; break;
		case AI_FOLLOW_EXITPRODUCTION_PATH:
		case AI_FOLLOW_WAYPOINT_PATH_AS_TEAM: case AI_FOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS:
		case AI_FOLLOW_WAYPOINT_PATH_AS_TEAM_EXACT: case AI_FOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS_EXACT:
			kind = "move"; break;
		case AI_ATTACK_MOVE_TO: kind = "attack_move"; point = true; break;
		case AI_ATTACKFOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS: case AI_ATTACKFOLLOW_WAYPOINT_PATH_AS_TEAM:
			kind = "attack_move"; break;
		case AI_ATTACK_OBJECT: case AI_FORCE_ATTACK_OBJECT: case AI_ATTACK_AND_FOLLOW_OBJECT:
			kind = "attack"; targetTask = true; break;
		case AI_ATTACK_POSITION: kind = "attack"; point = true; break;
		case AI_DOCK: kind = "dock"; targetTask = true; break;
		case AI_GUARD: case AI_GUARD_TUNNEL_NETWORK: case AI_GUARD_RETALIATE: kind = "guard"; break;
		default: kind = "other"; break;
		}
		ObjectID goal = targetTask ? ai->getGoalObjectID() : INVALID_ID;
		DozerAIInterface *dozer = ai->getDozerAIInterface();
		if (dozer && dozer->getCurrentTask() != DOZER_TASK_INVALID) {
			const DozerTask task = dozer->getCurrentTask();
			kind = task == DOZER_TASK_BUILD ? "construct" : task == DOZER_TASK_REPAIR ? "repair" : "other";
			goal = dozer->getTaskTarget(task);
			point = path = false;
		}
		target = directReference(goal, owner);
		if (goal != INVALID_ID) {
			if (target.target) destination = target.target->getPosition();
		} else if (point) destination = ai->getGoalPosition();
		else if (path && ai->getGoalPathEnd()) destination = ai->getGoalPathEnd();
		if (destination) { savedDestination = *destination; destination = &savedDestination; }
		moving = ai->isMoving() ? true : false;
		attacking = (ai->isAttacking() || object->testStatus(OBJECT_STATUS_IS_ATTACKING)) ? true : false;
		waiting = ai->isWaitingForPath() ? true : false;
		busy = ai->isBusy() ? true : false;
		combat = directReference(ai->getCurrentVictimID(), owner);
		supply = ai->getSupplyTruckAIInterface();
		if (supply) {
			ferrying = supply->isCurrentlyFerryingSupplies() ? true : false;
			boxes = supply->getNumberBoxes();
			dock = directReference(supply->getPreferredDockID(), owner);
			// The supply state machine delivers whenever it carries any boxes.
			if (ferrying) kind = boxes > 0 ? "return_supplies" : "collect";
		}
	} else kind = "idle";
	ProductionUpdateInterface *production = object->getProductionUpdateInterface();
	if (object->getConstructionPercent() != -1) kind = "construct";
	else if (production && production->firstProduction() && !strcmp(kind, "idle")) kind = "produce";
	static const char *const taskFields[] = {"kind", "available", "moving", "attacking", "waiting_for_path",
		"busy", "disabled", "destination", "target", "combat_target", "cargo", "preferred_dock"};
	DirectCursor fields = out.object(parent, FIELDS(taskFields));
	out.put(fields, kind); out.put(fields, true); out.put(fields, moving); out.put(fields, attacking);
	out.put(fields, waiting); out.put(fields, busy); out.put(fields, object->isDisabled() ? true : false);
	if (destination) out.point(fields, destination); else out.null(fields);
	writeReference(out, fields, target); writeReference(out, fields, combat);
	static const char *const cargoFull[] = {"available", "boxes", "capacity", "ferrying"};
	static const char *const cargoNone[] = {"available"};
	DirectCursor cargo = supply ? out.object(fields, FIELDS(cargoFull)) : out.object(fields, FIELDS(cargoNone));
	out.put(cargo, supply != nullptr);
	if (supply) { out.put(cargo, boxes); out.put(cargo, supply->getMaxBoxes()); out.put(cargo, ferrying); }
	cargo.finish();
	writeReference(out, fields, dock);
	fields.finish();
}

struct DirectSighting {
	unsigned id;
	bool visible, persistent;
	unsigned lastSeen;
	int owner;
	std::string type;
	Coord3D position;
	float health, maximumHealth;
	float velocityX, velocityY, facing;
	int veterancy;
	int supplyBoxes;
	EntityArms arms;
	bool firing, poisoned, burning;
	const ThingTemplate *typeTemplate;
	DirectSighting() : id(0), visible(false), persistent(true), lastSeen(0), owner(-1), health(0), maximumHealth(0),
		velocityX(0), velocityY(0), facing(0), veterancy(0), supplyBoxes(0), firing(false), poisoned(false), burning(false),
		typeTemplate(nullptr) {
		position.x = position.y = position.z = 0;
	}
};
typedef std::vector<DirectSighting> Sightings;
Sightings directSightings[MAX_PLAYER_COUNT];

// First index whose id is not below `id` (the memory is sorted by id).
size_t sightingIndex(const Sightings &memory, unsigned id)
{
	size_t low = 0, high = memory.size();
	while (low < high) {
		const size_t middle = low + (high - low) / 2;
		if (memory[middle].id < id) low = middle + 1; else high = middle;
	}
	return low;
}

struct DirectScratch {
	std::vector<unsigned> docks, resumes;
	std::vector<const ProductionEntry *> production;
	std::vector<const char *> buildable, trainable;
	std::vector<std::pair<int, int> > gridCells;
	std::vector<std::pair<Object *, int> > effects;
	std::vector<std::pair<Int, const SuperweaponInfo *> > superweapons;
} directScratch;

void refreshDirectSighting(DirectSighting &result, Object *object, unsigned frame)
{
	result.id = static_cast<unsigned>(object->getID());
	result.lastSeen = frame;
	Player *owner = object->getControllingPlayer();
	result.owner = owner ? static_cast<int>(owner->getPlayerIndex()) : -1;
	const char *name = object->getTemplate()->getName().str();
	if (result.type != name) result.type = name;
	if (result.typeTemplate != object->getTemplate()) {
		result.typeTemplate = object->getTemplate();
		result.persistent = persistentSightingTemplate(result.typeTemplate);
	}
	result.position = *object->getPosition();
	BodyModuleInterface *body = object->getBodyModule();
	result.health = body ? body->getHealth() : 0.0f;
	result.maximumHealth = body ? body->getMaxHealth() : 0.0f;
	const Coord3D velocity = objectVelocity(object);
	result.velocityX = velocity.x;
	result.velocityY = velocity.y;
	result.facing = object->getOrientation();
	result.veterancy = static_cast<int>(object->getVeterancyLevel());
	result.supplyBoxes = supplyBoxes(object);
	result.arms = entityArms(object, false);
	result.firing = firingVisibly(object, frame, true);
	result.poisoned = poisonedVisibly(object);
	result.burning = burningVisibly(object);
	result.visible = true;
}

void writeDirectSighting(DirectOutput &out, DirectCursor &parent, const DirectSighting &entity)
{
	static const char *const sightingFields[] = {"id", "type", "owner", "position", "health", "max_health",
		"last_seen_frame", "visible", "own", "velocity_x", "velocity_y", "facing", "veterancy", "supply_boxes",
		"weapons", "armor", "firing", "poisoned", "burning"};
	DirectCursor fields = out.object(parent, FIELDS(sightingFields));
	out.put(fields, entity.id); out.put(fields, entity.type); out.put(fields, entity.owner);
	out.point(fields, &entity.position); out.put(fields, entity.health); out.put(fields, entity.maximumHealth);
	out.put(fields, entity.lastSeen); out.put(fields, entity.visible); out.put(fields, false);
	out.put(fields, entity.velocityX); out.put(fields, entity.velocityY);
	out.put(fields, entity.facing); out.put(fields, entity.veterancy);
	out.put(fields, entity.supplyBoxes);
	DirectCursor weapons = out.array(fields, WEAPONSLOT_COUNT);
	for (int s = 0; s < WEAPONSLOT_COUNT; ++s) out.put(weapons, entity.arms.weapons[s]);
	out.put(fields, entity.arms.armor);
	// A fog ghost shows no current activity.
	out.put(fields, entity.visible && entity.firing); out.put(fields, entity.visible && entity.poisoned);
	out.put(fields, entity.visible && entity.burning);
	fields.finish();
}

Bool canDock(Object *actor, Object *target)
{
	if (!actor || !target || actor == target || actor->isEffectivelyDead() || target->isEffectivelyDead()
		|| !actor->getControllingPlayer() || !actor->getAIUpdateInterface()) return FALSE;
	const int player = actor->getControllingPlayer()->getPlayerIndex();
	if (target->getControllingPlayer() != actor->getControllingPlayer() && !TrainingObjectVisibleTo(target, player))
		return FALSE;
	return TheActionManager->canDockAt(actor, target, CMD_FROM_PLAYER);
}

Bool canResumeConstruction(Object *actor, Object *target)
{
	if (!actor || !target || actor == target || actor->isEffectivelyDead() || target->isEffectivelyDead()
		|| !actor->getControllingPlayer() || actor->getControllingPlayer() != target->getControllingPlayer()
		|| !actor->getAIUpdateInterface() || !actor->getAIUpdateInterface()->getDozerAIInterface()) return FALSE;
	// This is an owned foundation, so no hidden opponent state enters legality.
	return TheActionManager->canResumeConstructionOf(actor, target, CMD_FROM_PLAYER);
}

void directEntity(DirectOutput &out, DirectCursor &parent, Object *object, unsigned frame, ObservationLookups &lookups)
{
	Player *owner = object->getControllingPlayer();
	BodyModuleInterface *body = object->getBodyModule();
	Object *producer = TheGameLogic->findObjectByID(object->getProducerID());
	Object *builder = TheGameLogic->findObjectByID(object->getBuilderID());
	const bool hasProducer = producer && producer->getControllingPlayer() == owner;
	const bool hasBuilder = builder && builder->getControllingPlayer() == owner;
	const char *names[] = {"id", "type", "owner", "position", "health", "max_health", "last_seen_frame", "visible", "own",
		"velocity_x", "velocity_y", "facing", "veterancy", "supply_boxes", "weapons", "armor",
		"firing", "poisoned", "burning", "weapon_bonus",
		"task", "radius", "has_weapon", "weapon_cooldown_ticks", "ticks_since_damage",
		"can_attack", "has_atk_move_cmd", "can_attack_move", "dock_targets", "resume_targets",
		"construction_percent", "ai_state", "production_queue", "buildable_templates", "trainable_templates",
		"commands", "command_cooldowns", "command_targets", "enter_targets", "container_id", "occupants", "capacity",
		"upgrades", "can_guard", "repair_targets", "repair_depots", "builder_id", "builder_id"};
	names[46] = hasProducer ? "producer_id" : "builder_id";
	const size_t count = 46 + (hasProducer ? 1 : 0) + (hasBuilder ? 1 : 0);
	DirectCursor fields = out.writer.object(out.buffer, parent.next(), names, count);
	out.put(fields, static_cast<unsigned>(object->getID())); out.put(fields, object->getTemplate()->getName().str());
	out.put(fields, owner ? static_cast<int>(owner->getPlayerIndex()) : -1);
	out.point(fields, object->getPosition());
	out.put(fields, body ? body->getHealth() : 0.0f); out.put(fields, body ? body->getMaxHealth() : 0.0f);
	out.put(fields, frame); out.put(fields, true); out.put(fields, true);
	const Coord3D velocity = objectVelocity(object);
	out.put(fields, velocity.x); out.put(fields, velocity.y);
	out.put(fields, object->getOrientation()); out.put(fields, static_cast<int>(object->getVeterancyLevel()));
	out.put(fields, supplyBoxes(object));
	const EntityArms arms = entityArms(object, true);
	DirectCursor weaponList = out.array(fields, WEAPONSLOT_COUNT);
	for (int s = 0; s < WEAPONSLOT_COUNT; ++s) out.put(weaponList, arms.weapons[s]);
	out.put(fields, arms.armor);
	out.put(fields, firingVisibly(object, frame, true)); out.put(fields, poisonedVisibly(object));
	out.put(fields, burningVisibly(object));
	DirectCursor bonus = out.array(fields, 3);
	out.put(bonus, arms.damage); out.put(bonus, arms.range); out.put(bonus, arms.rate);
	const CommandSet *set = commandSetFor(object, lookups);
	directTask(out, fields, object);
	out.put(fields, object->getGeometryInfo().getBoundingCircleRadius());
	out.put(fields, object->hasAnyWeapon() ? true : false);
	out.put(fields, weaponCooldownTicks(object, frame));
	out.put(fields, ticksSinceDamage(body, frame));
	const bool canAttack = TrainingCanAttack(object) ? true : false;
	const bool attackMove = hasAttackMoveCommand(set);
	out.put(fields, canAttack); out.put(fields, attackMove);
	out.put(fields, canAttack && attackMove && !object->isKindOf(KINDOF_IMMOBILE) && object->getAIUpdateInterface() != nullptr);
	std::vector<unsigned> &docks = directScratch.docks;
	docks.clear();
	if (object->getAIUpdateInterface())
		for (size_t d = 0; d < lookups.docks.size(); ++d)
			if (canDock(object, lookups.docks[d])) docks.push_back(static_cast<unsigned>(lookups.docks[d]->getID()));
	out.ids(fields, docks);
	std::vector<unsigned> &resumes = directScratch.resumes;
	resumes.clear();
	if (object->getAIUpdateInterface() && object->getAIUpdateInterface()->getDozerAIInterface())
		for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject())
			if (canResumeConstruction(object, target)) resumes.push_back(static_cast<unsigned>(target->getID()));
	out.ids(fields, resumes);
	out.put(fields, object->getConstructionPercent());
	out.put(fields, object->getAIUpdateInterface() ? object->getAIUpdateInterface()->getCurrentStateName().str() : "");
	std::vector<const ProductionEntry *> &queue = directScratch.production;
	queue.clear();
	ProductionUpdateInterface *production = object->getProductionUpdateInterface();
	if (production) for (const ProductionEntry *p = production->firstProduction(); p; p = production->nextProduction(p)) queue.push_back(p);
	DirectCursor entries = out.array(fields, queue.size());
	for (size_t q = 0; q < queue.size(); ++q) {
		const ProductionEntry *p = queue[q];
		// Unit and upgrade pointers share a union: only the type discriminates.
		const bool unit = p->getProductionType() == PRODUCTION_UNIT && p->getProductionObject();
		const bool upgrade = p->getProductionType() == PRODUCTION_UPGRADE && p->getProductionUpgrade();
		const char *keys[] = {"production_id", "kind", "progress", "remaining_quantity", "upgrade", "upgrade"};
		if (unit) keys[4] = "unit_type";
		DirectCursor entry = out.writer.object(out.buffer, entries.next(), keys, 4 + (unit ? 1 : 0) + (upgrade ? 1 : 0));
		out.put(entry, static_cast<unsigned>(p->getProductionID())); out.put(entry, static_cast<int>(p->getProductionType()));
		out.put(entry, p->getPercentComplete()); out.put(entry, p->getProductionQuantityRemaining());
		if (unit) out.put(entry, p->getProductionObject()->getName().str());
		if (upgrade) out.put(entry, p->getProductionUpgrade()->getUpgradeName().str());
		entry.finish();
	}
	std::vector<const char *> &buildable = directScratch.buildable;
	std::vector<const char *> &trainable = directScratch.trainable;
	buildable.clear(); trainable.clear();
	if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
		const CommandButton *button = set->getCommandButton(i);
		if (!button || !button->getThingTemplate()) continue;
		const ThingTemplate *what = button->getThingTemplate();
		if (button->getCommandType() == GUI_COMMAND_DOZER_CONSTRUCT && TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK)
			buildable.push_back(what->getName().str());
		if (button->getCommandType() == GUI_COMMAND_UNIT_BUILD && production
			&& TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK && production->canQueueCreateUnit(what) == CANMAKE_OK)
			trainable.push_back(what->getName().str());
	}
	out.names(fields, buildable);
	out.names(fields, trainable);
	training_commands::ObjectCommands &state = lookups.commands;
	training_commands::objectCommands(object, set, lookups.commandTargets, state);
	out.names(fields, state.available);
	static const char *const cooldownFields[] = {"command", "ready_percent"};
	DirectCursor cooldowns = out.array(fields, state.charging.size());
	for (size_t c = 0; c < state.charging.size(); ++c) {
		DirectCursor entry = out.object(cooldowns, FIELDS(cooldownFields));
		out.put(entry, state.charging[c].first); out.put(entry, state.charging[c].second);
		entry.finish();
	}
	static const char *const targetFields[] = {"command", "targets"};
	DirectCursor targets = out.array(fields, state.targets.size());
	for (size_t t = 0; t < state.targets.size(); ++t) {
		DirectCursor entry = out.object(targets, FIELDS(targetFields));
		out.put(entry, state.targets[t].first);
		out.ids(entry, state.targets[t].second);
		entry.finish();
	}
	out.ids(fields, state.enterTargets);
	out.put(fields, state.containerID); out.put(fields, state.occupants); out.put(fields, state.capacity);
	out.names(fields, state.upgrades);
	out.put(fields, state.canGuard);
	out.ids(fields, state.repairTargets);
	out.ids(fields, state.repairDepots);
	if (hasProducer) out.put(fields, static_cast<unsigned>(producer->getID()));
	if (hasBuilder) out.put(fields, static_cast<unsigned>(builder->getID()));
	fields.finish();
}

// The last submitted command of this player (replays have no validations).
void directContext(DirectOutput &out, DirectCursor &parent, const Json &record, Player *owner, unsigned frame)
{
	static const char *const unavailable[] = {"available"};
	if (record.isNull()) {
		DirectCursor fields = out.object(parent, FIELDS(unavailable));
		out.put(fields, false);
		return;
	}
	std::vector<std::string> keys;
	for (size_t k = 0; k < record.size(); ++k) if (record.keyAt(k) != "age_seconds") keys.push_back(record.keyAt(k));
	keys.push_back("age_seconds");
	DirectCursor fields = out.writer.namedObject(out.buffer, parent.next(), keys);
	unsigned recordFrame = 0;
	for (size_t i = 0; i < record.size(); ++i) {
		const std::string &key = record.keyAt(i);
		if (key == "age_seconds") continue;
		if (key == "frame") recordFrame = record.at(i).asUnsigned();
		const Json *targetID = key == "target" ? record.at(i).find("id") : nullptr;
		// The target is re-resolved now: the owner's current view of it.
		if (targetID) writeReference(out, fields, directReference(static_cast<ObjectID>(targetID->asUnsigned()), owner));
		else record.at(i).encode(out.writer, out.buffer, fields.next());
	}
	out.put(fields, secondsFor(frame - recordFrame));
	fields.finish();
}

// One sample's pass over the world: owned objects leave sighting memory (a full
// observation also writes them), visible others refresh theirs, and hidden
// temporary objects are forgotten.
void observeObjects(Player *player, unsigned frame, DirectOutput *out, DirectCursor *own, ObservationLookups &lookups)
{
	const int index = static_cast<int>(player->getPlayerIndex());
	Sightings &memory = directSightings[index];
	for (size_t m = 0; m < memory.size(); ++m) memory[m].visible = false;
	for (Object *object = TheGameLogic->getFirstObject(); object; object = object->getNextObject()) {
		if (!retainedEntity(object) || object->isEffectivelyDead() || object->isDestroyed()) continue;
		const unsigned id = static_cast<unsigned>(object->getID());
		const size_t found = sightingIndex(memory, id);
		if (object->getControllingPlayer() == player) {
			if (out) directEntity(*out, *own, object, frame, lookups);
			if (found < memory.size() && memory[found].id == id) memory.erase(memory.begin() + found);
		} else if (TrainingObjectVisibleTo(object, index)) {
			if (found == memory.size() || memory[found].id != id) memory.insert(memory.begin() + found, DirectSighting());
			refreshDirectSighting(memory[found], object, frame);
		}
	}
	const size_t before = memory.size();
	Sightings kept;
	kept.reserve(memory.size());
	// Hidden temporary objects are forgotten, and so is a remembered immobile object whose spot
	// is revealed but which is not seen there (the game's ghost rule; observer shroud only).
	for (size_t k = 0; k < memory.size(); ++k)
		if (memory[k].visible || (memory[k].persistent
			&& ThePartitionManager->getShroudStatusForPlayer(index, &memory[k].position) != CELLSHROUD_CLEAR))
			kept.push_back(memory[k]);
	memory.swap(kept);
	temporarySightingsForgotten[index] += static_cast<unsigned>(before - memory.size());
}

// Stats of the templates this player's stream has not received yet; the shard keeps one table per game.
void directTemplateStats(DirectOutput &out, DirectCursor &parent, int player)
{
	static const char *const statsFields[] = {"weapons", "armors"};
	static const char *const weaponFields[] = {"id", "name", "range", "min_range", "damage", "radius",
		"min_delay_frames", "max_delay_frames", "clip_size", "clip_reload_frames", "anti_mask", "damage_type"};
	static const char *const armorFields[] = {"id", "damage_multipliers"};
	TemplateCatalog &catalog = armsCatalog;
	const WeaponBonus none;
	DirectCursor stats = out.object(parent, FIELDS(statsFields));
	DirectCursor weapons = out.array(stats, catalog.weapons.size() - catalog.weaponsSent[player]);
	size_t i;
	for (i = catalog.weaponsSent[player]; i < catalog.weapons.size(); ++i) {
		const WeaponTemplate *t = catalog.weapons[i];
		DirectCursor w = out.object(weapons, FIELDS(weaponFields));
		out.put(w, static_cast<unsigned int>(i)); out.put(w, t->getName().str());
		out.put(w, t->getAttackRange(none)); out.put(w, t->getMinimumAttackRange());
		out.put(w, t->getPrimaryDamage(none)); out.put(w, t->getPrimaryDamageRadius(none));
		out.put(w, static_cast<int>(t->getMinDelayBetweenShots())); out.put(w, static_cast<int>(t->getMaxDelayBetweenShots()));
		out.put(w, static_cast<int>(t->getClipSize())); out.put(w, static_cast<int>(t->getClipReloadTime(none)));
		out.put(w, static_cast<int>(t->getAntiMask())); out.put(w, static_cast<int>(t->getDamageType()));
		w.finish();
	}
	weapons.finish();
	catalog.weaponsSent[player] = catalog.weapons.size();
	DirectCursor armors = out.array(stats, catalog.armors.size() - catalog.armorsSent[player]);
	for (i = catalog.armorsSent[player]; i < catalog.armors.size(); ++i) {
		DirectCursor a = out.object(armors, FIELDS(armorFields));
		out.put(a, static_cast<unsigned int>(i));
		DirectCursor multipliers = out.array(a, DAMAGE_NUM_TYPES);
		for (int damage = 0; damage < DAMAGE_NUM_TYPES; ++damage)
			out.put(multipliers, catalog.armors[i]->adjustDamage(static_cast<DamageType>(damage), 1.0f));
		a.finish();
	}
	armors.finish();
	catalog.armorsSent[player] = catalog.armors.size();
	stats.finish();
}

// v11 current perception, as the Generals Online exporter writes it.
void directPerception(DirectOutput &out, DirectCursor &root, Player *player, unsigned frame)
{
	const int index = static_cast<int>(player->getPlayerIndex());
	static const char *const effectFields[] = {"id", "type", "owner", "kind", "position", "velocity", "radius"};
	std::vector<std::pair<Object *, int> > &visible = directScratch.effects;
	visible.clear();
	for (Object *o = TheGameLogic->getFirstObject(); o; o = o->getNextObject()) {
		const int kind = effectKind(o);
		if (kind == EFFECT_NONE || o->isEffectivelyDead() || o->isDestroyed()) continue;
		if (o->getControllingPlayer() != player && !TrainingObjectVisibleTo(o, index)) continue;
		visible.push_back(std::make_pair(o, kind));
	}
	DirectCursor effects = out.array(root, visible.size());
	for (size_t e = 0; e < visible.size(); ++e) {
		const Object *o = visible[e].first;
		const Player *owner = o->getControllingPlayer();
		const Coord3D velocity = effectVelocity(o);
		DirectCursor fields = out.object(effects, FIELDS(effectFields));
		out.put(fields, static_cast<unsigned>(o->getID())); out.put(fields, o->getTemplate()->getName().str());
		out.put(fields, owner ? static_cast<int>(owner->getPlayerIndex()) : -1);
		out.put(fields, effectName(visible[e].second)); out.point(fields, o->getPosition());
		out.point(fields, &velocity);
		out.put(fields, o->getGeometryInfo().getBoundingCircleRadius());
		fields.finish();
	}
	effects.finish();
	// Public superweapon countdowns: exactly the timers InGameUI draws, with its rules.
	static const char *const timerFields[] = {"owner", "power", "ready_seconds", "ready"};
	std::vector<std::pair<Int, const SuperweaponInfo *> > &shown = directScratch.superweapons;
	shown.clear();
	if (TheInGameUI) TheInGameUI->getDisplayedSuperweapons(shown);
	std::vector<size_t> kept;
	for (size_t s = 0; s < shown.size(); ++s) {
		Object *owner = TheGameLogic->findObjectByID(shown[s].second->m_id);
		if (owner && ThePlayerList->getNthPlayer(shown[s].first) && !owner->testStatus(OBJECT_STATUS_UNDER_CONSTRUCTION)
			&& owner->getSpecialPowerModule(shown[s].second->getSpecialPowerTemplate()))
			kept.push_back(s);
	}
	const UnsignedInt now = TheGameLogic->getFrame();
	DirectCursor superweapons = out.array(root, kept.size());
	for (size_t k = 0; k < kept.size(); ++k) {
		const SuperweaponInfo *info = shown[kept[k]].second;
		SpecialPowerModuleInterface *power = TheGameLogic->findObjectByID(info->m_id)->getSpecialPowerModule(info->getSpecialPowerTemplate());
		const UnsignedInt ready = power->getReadyFrame();
		DirectCursor fields = out.object(superweapons, FIELDS(timerFields));
		out.put(fields, static_cast<int>(shown[kept[k]].first)); out.put(fields, info->getSpecialPowerTemplate()->getName().str());
		out.put(fields, ready > now ? static_cast<int>((ready - now) / LOGICFRAMES_PER_SECOND) : 0);
		out.put(fields, power->isReady() ? true : false);
		fields.finish();
	}
	superweapons.finish();
	// Special-power launch sounds this player hears.
	static const char *const cueFields[] = {"power", "owner", "ticks_since", "position"};
	std::vector<std::pair<const PowerCue *, const Coord3D *> > heard;
	for (size_t c = 0; c < powerCues.size(); ++c) {
		const PowerCue &cue = powerCues[c];
		if (!cueCurrent(cue, frame)) continue;
		const bool atSource = hears(cue.sourceAudience, cue.owner, player);
		const bool atTarget = cue.hasTarget && hears(cue.targetAudience, cue.owner, player);
		if (!atSource && !atTarget && !cue.announced) continue;
		const Coord3D *where = nullptr;
		if (atTarget && ThePartitionManager->getShroudStatusForPlayer(index, &cue.target) == CELLSHROUD_CLEAR) where = &cue.target;
		else if (atSource && ThePartitionManager->getShroudStatusForPlayer(index, &cue.source) == CELLSHROUD_CLEAR) where = &cue.source;
		heard.push_back(std::make_pair(&cue, where));
	}
	DirectCursor cues = out.array(root, heard.size());
	for (size_t h = 0; h < heard.size(); ++h) {
		DirectCursor fields = out.object(cues, FIELDS(cueFields));
		out.put(fields, heard[h].first->power); out.put(fields, heard[h].first->owner);
		out.put(fields, static_cast<unsigned>(frame - heard[h].first->frame));
		if (heard[h].second) out.point(fields, heard[h].second); else out.null(fields);
		fields.finish();
	}
	cues.finish();
}

void directObservation(Player *player, unsigned frame, DirectBuffer &buffer)
{
	DirectOutput out(*binaryWriter, buffer);
	out.writer.begin(buffer);
	static const char *const rootFields[] = {"player", "own_entities", "known_nonowned_entities", "visibility_grid",
		"observation_policy_version", "last_submission", "last_validation", "template_stats",
		"effects", "superweapons", "audio_cues"};
	DirectCursor root = out.writer.object(buffer, 0, FIELDS(rootFields));
	const int index = static_cast<int>(player->getPlayerIndex());
	static const char *const playerFields[] = {"index", "faction", "cash", "power_production", "power_consumption", "radar", "rank",
		"skill_points", "science_points", "dead", "sciences", "purchasable_sciences", "upgrades"};
	DirectCursor state = out.object(root, FIELDS(playerFields));
	out.put(state, index); out.put(state, player->getPlayerTemplate()->getName().str());
	out.put(state, player->getMoney()->countMoney()); out.put(state, player->getEnergy()->getProduction());
	out.put(state, player->getEnergy()->getConsumption()); out.put(state, player->hasRadar() ? true : false);
	out.put(state, player->getRankLevel()); out.put(state, player->getSkillPoints());
	out.put(state, player->getSciencePurchasePoints()); out.put(state, player->isPlayerDead() ? true : false);
	ObservationLookups &lookups = observationLookups;
	training_commands::playerCommands(player, lookups.player);
	out.names(state, lookups.player.sciences);
	out.names(state, lookups.player.purchasableSciences);
	out.names(state, lookups.player.upgrades);
	state.finish();
	lookups.commandSets.clear(); lookups.docks.clear(); lookups.commandTargets.clear();
	size_t owned = 0;
	for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject()) {
		if (!retainedEntity(target)) continue;
		if (!target->isEffectivelyDead() && !target->isDestroyed() && target->getControllingPlayer() == player) ++owned;
		const bool dock = hasDockInterface(target);
		// canDock also requires a live, owned or visible target.
		const bool seen = !target->isEffectivelyDead()
			&& (target->getControllingPlayer() == player || TrainingObjectVisibleTo(target, index));
		if (dock && seen) lookups.docks.push_back(target);
		if (seen && !target->isDestroyed()) lookups.commandTargets.add(target, dock);
	}
	DirectCursor own = out.array(root, owned);
	observeObjects(player, frame, &out, &own, lookups);
	own.finish();
	const Sightings &memory = directSightings[index];
	DirectCursor known = out.array(root, memory.size());
	for (size_t m = 0; m < memory.size(); ++m) writeDirectSighting(out, known, memory[m]);
	DirectCursor visibility = out.array(root, gridSize * gridSize);
	std::vector<std::pair<int, int> > &cells = directScratch.gridCells;
	if (cells.empty()) { // the map does not change during a game
		const Region3D region = extent();
		cells.resize(gridSize * gridSize);
		for (int y = 0; y < gridSize; ++y) for (int x = 0; x < gridSize; ++x) {
			const Coord3D p = gridPoint(region, x, y);
			Int cellX, cellY;
			ThePartitionManager->worldToCell(p.x, p.y, &cellX, &cellY);
			cells[y * gridSize + x] = std::make_pair(static_cast<int>(cellX), static_cast<int>(cellY));
		}
	}
	for (size_t c = 0; c < cells.size(); ++c)
		out.put(visibility, static_cast<int>(ThePartitionManager->getShroudStatusForPlayer(index, cells[c].first, cells[c].second)));
	out.put(root, 11);
	directContext(out, root, submissions[index], player, frame);
	directContext(out, root, Json(), player, frame);
	// After every entity this observation wrote: their template ids are now in the catalogue.
	directTemplateStats(out, root, index);
	directPerception(out, root, player, frame);
	root.finish();
}

void resetObservations()
{
	for (int p = 0; p < MAX_PLAYER_COUNT; ++p) {
		directSightings[p].clear();
		submissions[p] = Json();
		observedSightingRemovals[p] = temporarySightingsForgotten[p] = 0;
		hasLastSample[p] = hasLastCommand[p] = false;
		lastSample[p] = lastCommand[p] = 0;
		pendingSamples[p] = Json();
		pendingDirectBuffers[p].clear();
		quietDirectives[p] = QuietDirective();
	}
	observationLookups.commandSets.clear(); observationLookups.docks.clear();
	armsCatalog.reset();
	std::fill(retainedTemplates.begin(), retainedTemplates.end(), RetainedTemplate());
	std::fill(effectTemplates.begin(), effectTemplates.end(), EffectTemplate());
	powerCues.clear();
	directScratch.gridCells.clear();
}

void createDirectories(const std::string &path)
{
	for (size_t i = 0; i <= path.size(); ++i)
		if (i == path.size() || path[i] == '\\' || path[i] == '/') {
			const std::string prefix = path.substr(0, i);
			if (!prefix.empty() && prefix[prefix.size() - 1] != ':') CreateDirectoryA(prefix.c_str(), nullptr);
		}
	const DWORD attributes = GetFileAttributesA(path.c_str());
	if (attributes == static_cast<DWORD>(-1) || !(attributes & FILE_ATTRIBUTE_DIRECTORY))
		throw std::runtime_error("Cannot create training output directory");
}

Json counts(const unsigned *values)
{
	Json result = Json::array();
	for (int p = 0; p < MAX_PLAYER_COUNT; ++p) result.push(U(values[p]));
	return result;
}

void closeOutput()
{
	delete binaryWriter;
	binaryWriter = nullptr;
	if (output) fclose(output);
	output = nullptr;
}
} // namespace

Bool TrainingObjectVisibleTo(const Object *object, Int playerIndex)
{
	const Player *viewer = ThePlayerList->getNthPlayer(playerIndex);
	if (!object || !viewer || object->isEffectivelyDead() || object->isDestroyed()) return FALSE;
	if (object->getControllingPlayer() == viewer) return TRUE;
	// Contained objects have no partition data and report CLEAR as a fallback.
	// Visibility of the transport/building does not reveal its occupants.
	if (object->getContainedBy()) return FALSE;
	const ObjectShroudStatus shroud = object->getShroudedStatus(playerIndex);
	if (shroud != OBJECTSHROUD_CLEAR && shroud != OBJECTSHROUD_PARTIAL_CLEAR) return FALSE;
	StealthUpdate *stealth = object->getStealth();
	if (stealth) {
		// The engine makes detection visible globally, but each viewer still
		// needs current shroud visibility. Never query the observed/local player.
		const StealthLookType look = stealth->getStealthLookForPlayer(viewer);
		// Disguises need an apparent-identity contract; do not disclose real IDs.
		return look != STEALTHLOOK_INVISIBLE && look != STEALTHLOOK_DISGUISED_ENEMY;
	}
	return !object->testStatus(OBJECT_STATUS_STEALTHED) || object->testStatus(OBJECT_STATUS_DETECTED)
		|| viewer->getRelationship(object->getTeam()) == ALLIES;
}

Bool TrainingCanAttack(Object *object)
{
	if (!object || object->isEffectivelyDead() || !object->isAbleToAttack()) return FALSE;
	for (Int i = 0; i < WEAPONSLOT_COUNT; ++i) {
		const WeaponSlotType slot = static_cast<WeaponSlotType>(i);
		const Weapon *weapon = object->getWeaponInWeaponSlot(slot);
		// Mine-clearing and hacking weapons do not make a worker a general
		// combat actor. The normal dispatcher still resolves the target.
		if (weapon && object->getWeaponInWeaponSlotCommandSourceMask(slot)
			&& weapon->getDamageType() != DAMAGE_DISARM && weapon->getDamageType() != DAMAGE_HACK)
			return TRUE;
	}
	const ContainModuleInterface *contain = object->getContain();
	if (contain) {
		const ContainedItemsList *passengers = contain->getContainedItemsList();
		if (passengers)
			for (ContainedItemsList::const_iterator it = passengers->begin(); it != passengers->end(); ++it)
				if (contain->isPassengerAllowedToFire((*it)->getID()) && TrainingCanAttack(*it))
					return TRUE;
	}
	SpawnBehaviorInterface *spawn = object->getSpawnBehaviorInterface();
	return spawn && spawn->canAnySlavesAttack();
}

void TrainingObserveObjectRemoval(const Object *object)
{
	if (!active || !ThePlayerList || !object) return;
	const unsigned id = static_cast<unsigned>(object->getID());
	for (int player = 0; player < MAX_PLAYER_COUNT; ++player) {
		Sightings &memory = directSightings[player];
		const size_t found = sightingIndex(memory, id);
		// Only a witnessed removal retires a sighting; visibility is checked now,
		// before dead/destroyed flags change it.
		if (found < memory.size() && memory[found].id == id && TrainingObjectVisibleTo(object, player)) {
			memory.erase(memory.begin() + found);
			++observedSightingRemovals[player];
		}
	}
}

void TrainingObserveSpecialPower(const Object *source, const SpecialPowerTemplate *power, const Coord3D *target)
{
	if (!active || !ThePlayerList || !source || !power) return;
	const Player *owner = source->getControllingPlayer();
	if (!owner) return;
	const unsigned frame = TheGameLogic->getFrame();
	std::vector<PowerCue> current;
	for (size_t c = 0; c < powerCues.size(); ++c)
		if (cueCurrent(powerCues[c], frame)) current.push_back(powerCues[c]);
	powerCues.swap(current);
	PowerCue cue;
	cue.frame = frame;
	cue.owner = static_cast<int>(owner->getPlayerIndex());
	cue.power = power->getName().str();
	cue.source = *source->getPosition();
	cue.sourceAudience = soundAudience(power->getInitiateSound());
	if (target) {
		cue.hasTarget = true;
		cue.target = *target;
		cue.targetAudience = soundAudience(power->getInitiateAtTargetSound());
	}
	// EVA announces superweapon launches to every player (SpecialPowerModule::aboutToDoSpecialPower).
	switch (power->getSpecialPowerType()) {
	case SPECIAL_PARTICLE_UPLINK_CANNON: case SUPW_SPECIAL_PARTICLE_UPLINK_CANNON: case LAZR_SPECIAL_PARTICLE_UPLINK_CANNON:
	case SPECIAL_NEUTRON_MISSILE: case NUKE_SPECIAL_NEUTRON_MISSILE: case SUPW_SPECIAL_NEUTRON_MISSILE:
	case SPECIAL_SCUD_STORM:
		cue.announced = true; break;
	default: break;
	}
	powerCues.push_back(cue);
}

void TrainingObserveCommand(const GameMessage *message, const AIGroup *selection)
{
	if (!active || !TheGlobalData->m_headless) return;
	const GameMessage::Type type = message->getType();
	// The contiguous gameplay region excludes all selection/hotkey bookkeeping.
	if (type < GameMessage::MSG_DO_ATTACKSQUAD || type > GameMessage::MSG_ENABLE_RETALIATION_MODE) return;
	if (type == GameMessage::MSG_AREA_SELECTION_DEPRECATED || type == GameMessage::MSG_SET_REPLAY_CAMERA
		|| type == GameMessage::MSG_LOGIC_CRC || type == GameMessage::MSG_CLEAR_INGAME_POPUP_MESSAGE
		|| type == GameMessage::MSG_PLACE_BEACON || type == GameMessage::MSG_REMOVE_BEACON
		|| type == GameMessage::MSG_SET_BEACON_TEXT) return;
	Player *owner = ThePlayerList->getNthPlayer(message->getPlayerIndex());
	if (!owner) return;
	// Upgrades, abilities, garrison and promotions: one canonical action from
	// the dispatcher's real selection, before the message executes.
	Json resolved;
	if (training_commands::covers(message)) {
		std::string reason;
		Json action;
		if (training_commands::resolve(message, selection, action, reason)) resolved = action;
		else if (reason.empty()) reason = "unresolved";
		if (!failed) {
			for (size_t i = 0; i < pendingResolvedCommands.size(); ++i) {
				if (pendingResolvedCommands[i].message != message) continue;
				Json result = Json::object();
				if (resolved.isNull()) { result.set("resolved", B(false)); result.set("reason", S(reason.c_str())); }
				else { result.set("resolved", B(true)); result.set("action", resolved); }
				Json *actions = pendingSamples[pendingResolvedCommands[i].player].find("actions");
				if (actions && pendingResolvedCommands[i].index < actions->size())
					actions->at(pendingResolvedCommands[i].index).set("command_result", result);
				break;
			}
		}
	}
	Json &j = submissions[owner->getPlayerIndex()];
	if (!j.isObject()) j = Json::object();
	j.set("available", B(true));
	j.set("frame", U(TheGameLogic->getFrame()));
	j.set("kind", S("other"));
	Json actors = Json::array();
	if (!j.contains("actors")) j.set("actors", actors);
	if (!j.contains("target")) j.set("target", Json::object());
	taskStatus(*j.find("target"), "none");
	j.set("result", S("submitted"));
	if (selection) {
		const std::vector<ObjectID> &ids = selection->getAllIDs();
		for (size_t i = 0; i < ids.size(); ++i) actors.push(U(static_cast<unsigned>(ids[i])));
	}
	int point = -1, target = -1, what = -1;
	switch (type) {
	case GameMessage::MSG_DO_MOVETO: j.set("kind", S("move")); point = 0; break;
	case GameMessage::MSG_DO_ATTACKMOVETO: j.set("kind", S("attack_move")); point = 0; break;
	case GameMessage::MSG_DO_ATTACK_OBJECT: j.set("kind", S("attack")); target = 0; break;
	case GameMessage::MSG_DOCK: j.set("kind", S("dock")); target = 0; break;
	case GameMessage::MSG_RESUME_CONSTRUCTION: j.set("kind", S("resume_construction")); target = 0; break;
	case GameMessage::MSG_DO_STOP: j.set("kind", S("stop")); break;
	case GameMessage::MSG_SELL: j.set("kind", S("sell")); break;
	case GameMessage::MSG_DOZER_CONSTRUCT: j.set("kind", S("build")); point = 1; what = 0; break;
	case GameMessage::MSG_QUEUE_UNIT_CREATE: j.set("kind", S("train")); what = 0; break;
	case GameMessage::MSG_SET_RALLY_POINT:
		j.set("kind", S("rally")); point = 1;
		actors = Json::array();
		actors.push(U(static_cast<unsigned>(message->getArgument(0)->objectID)));
		break;
	default: break;
	}
	if (resolved.isObject()) {
		// Canonical actors, target and position; the raw message layouts vary.
		j.set("kind", *resolved.find("kind"));
		actors = Json::array();
		const Json *units = resolved.find("units");
		if (units) for (size_t u = 0; u < units->size(); ++u) actors.push(units->at(u));
		const Json *producer = resolved.find("producer");
		if (producer) actors.push(*producer);
		j.set("actors", actors);
		const Json *resolvedTarget = resolved.find("target");
		if (resolvedTarget) taskReference(*j.find("target"), static_cast<ObjectID>(resolvedTarget->asUnsigned()), owner);
		const Json *resolvedPosition = resolved.find("position");
		if (resolvedPosition) j.set("position", *resolvedPosition);
		else j.erase("position");
		j.erase("template");
		return;
	}
	j.set("actors", actors);
	if (point >= 0 && message->getArgumentCount() > point && message->getArgumentDataType(point) == ARGUMENTDATATYPE_LOCATION)
		j.set("position", position(&message->getArgument(point)->location));
	else j.erase("position");
	if (target >= 0 && message->getArgumentCount() > target && message->getArgumentDataType(target) == ARGUMENTDATATYPE_OBJECTID)
		taskReference(*j.find("target"), message->getArgument(target)->objectID, owner);
	if (what >= 0 && message->getArgumentCount() > what && message->getArgumentDataType(what) == ARGUMENTDATATYPE_INTEGER) {
		const ThingTemplate *t = TheThingFactory->findByTemplateID(message->getArgument(what)->integer);
		if (t) j.set("template", S(t->getName().str()));
		else j.erase("template");
	} else j.erase("template");
}

void TrainingExporterConfigure(const char *directory) { outputDirectory = directory; }
void TrainingExporterSetInterval(Int milliseconds) { intervalMilliseconds = milliseconds; }
void TrainingExporterSetScheduledQuietCapture(Bool scheduled) { scheduledQuietCapture = scheduled ? true : false; }
Bool TrainingExporterIsEnabled() { return !outputDirectory.empty(); }

void TrainingExporterBuildResult(const GameMessage *message, const Object *building)
{
	if (!active || failed) return;
	for (size_t i = 0; i < pendingBuildCommands.size(); ++i) {
		if (pendingBuildCommands[i].message != message) continue;
		Json result = Json::object();
		result.set("accepted", B(building != nullptr));
		result.set("status", S(building ? "created" : "rejected"));
		if (building) result.set("building_id", U(static_cast<unsigned>(building->getID())));
		Json *actions = pendingSamples[pendingBuildCommands[i].player].find("actions");
		if (actions && pendingBuildCommands[i].index < actions->size())
			actions->at(pendingBuildCommands[i].index).set("build_result", result);
		return;
	}
}

Bool TrainingExporterBegin(const AsciiString &replay)
{
	if (!TrainingExporterIsEnabled()) return TRUE;
	try {
		if (!TheGlobalData->m_headless) throw std::runtime_error("Training export requires -headless");
		createDirectories(outputDirectory);
		std::string name = replay.str();
		const size_t slash = name.find_last_of("/\\");
		if (slash != std::string::npos) name = name.substr(slash + 1);
		finalPath = outputDirectory;
		if (finalPath[finalPath.size() - 1] != '\\' && finalPath[finalPath.size() - 1] != '/') finalPath += '\\';
		finalPath += name + ".training.bin";
		temporaryPath = finalPath + ".part";
		closeOutput();
		output = fopen(temporaryPath.c_str(), "wb");
		if (!output) throw std::runtime_error("Cannot open training output");
		binaryWriter = new training_binary::Writer(output);
		active = true;
		failed = mapWritten = false;
		samples = commands = waitSamples = crcMessages = 0;
		pendingCount = 0; pendingBuildCommands.clear(); pendingResolvedCommands.clear();
		resetObservations();
		pendingHeader = Json::object();
		pendingHeader.set("record", S("header"));
		pendingHeader.set("schema_version", I(1));
		pendingHeader.set("replay", S(replay.str()));
		pendingHeader.set("engine_git_sha", S(GitSHA1));
		pendingHeader.set("engine_git_dirty", B(GitUncommittedChanges ? true : false));
		pendingHeader.set("engine_exe_crc", U(TheGlobalData->m_exeCRC));
		pendingHeader.set("engine_ini_crc", U(TheGlobalData->m_iniCRC));
		pendingHeader.set("ticks_per_second", I(static_cast<int>(LOGICFRAMES_PER_SECOND)));
		pendingHeader.set("interval_ms", I(intervalMilliseconds));
		pendingHeader.set("quiet_capture", S(scheduledQuietCapture ? "scheduled" : "all"));
		pendingHeader.set("replay_logic", S("current"));
		pendingHeader.set("engine_build", S("thesuperhackers-retail-vc6"));
		pendingHeader.set("controlled_faction", S("FactionChina"));
		pendingHeader.set("placement_mode", S("fast"));
		pendingHeader.set("template_catalog", templateCatalog());
		pendingHeader.set("observation_boundary", S("after recorder update, before this tick's command batch"));
		pendingHeader.set("action_format", S("ordered engine messages; selection messages retained; camera/CRC omitted"));
		pendingHeader.set("visibility_policy", S("per-player shroud and engine stealth look; no hidden target refresh"));
		return TRUE;
	} catch (const std::exception &e) {
		printf("Training export: %s\n", e.what());
		closeOutput();
		active = false;
		return FALSE;
	}
}

void TrainingExporterCapturePreCommands()
{
	if (!active || failed || !TheGameLogic->isInGame() || !ThePartitionManager) return;
	try {
		flushSamples();
		if (!mapWritten) mapRecord();
		const unsigned frame = TheGameLogic->getFrame();
		// Recorded CRCs have a false origin flag; locally generated replay
		// CRCs have true. Count only the former as replay coverage evidence.
		for (GameMessage *m = TheCommandList->getFirstMessage(); m; m = m->next())
			if (m->getType() == GameMessage::MSG_LOGIC_CRC && m->getArgumentCount() > 1
				&& m->getArgumentDataType(1) == ARGUMENTDATATYPE_BOOLEAN
				&& !m->getArgument(1)->boolean) ++crcMessages;
		int cadence = (intervalMilliseconds * LOGICFRAMES_PER_SECOND + 999) / 1000;
		if (cadence < 1) cadence = 1;
		for (int i = 0; i < ThePlayerList->getPlayerCount(); ++i) {
			Player *p = ThePlayerList->getNthPlayer(i);
			if (!exportedFaction(p)) continue;
			Json actions = actionsFor(i);
			if (actions.size() == 0 && frame % cadence != 0) continue;
			if (hasLastSample[i] && lastSample[i] == frame) continue;
			const bool waiting = actions.size() == 0;
			const unsigned actionCount = static_cast<unsigned>(actions.size());
			Json sample = Json::object();
			sample.set("record", S("sample"));
			sample.set("frame", U(frame));
			sample.set("player_index", I(i));
			sample.set("delta_ticks", U(hasLastSample[i] ? frame - lastSample[i] : 0));
			sample.set("time_seconds", R(secondsFor(frame)));
			const bool observed = !scheduledQuietCapture || !waiting || !hasLastSample[i]
				|| quietObservationWanted(i, frame);
			if (!waiting) quietDirectives[i].current = false;
			if (!observed) sample.set("light", B(true));
			sample.set("decision", S(waiting ? "wait" : "commands"));
			sample.set("actions", actions);
			if (observed) directObservation(p, frame, pendingDirectBuffers[i]);
			else {
				// Sighting memory still follows every row, as with a full observation.
				observeObjects(p, frame, nullptr, nullptr, observationLookups);
				pendingDirectBuffers[i].clear();
			}
			if (hasLastCommand[i]) sample.set("ticks_since_previous_command", U(frame - lastCommand[i]));
			else sample.set("ticks_since_previous_command", Json());
			pendingSamples[i] = sample;
			pendingPlayers[pendingCount++] = i;
			hasLastSample[i] = true;
			lastSample[i] = frame;
			++samples;
			if (waiting) ++waitSamples;
			else { hasLastCommand[i] = true; lastCommand[i] = frame; commands += actionCount; }
		}
	} catch (const std::exception &e) {
		failed = true;
		printf("Training export failed: %s\n", e.what());
	}
}

Bool TrainingExporterFinish(Bool playbackSucceeded, Bool crcMismatch)
{
	if (!TrainingExporterIsEnabled()) return TRUE;
	bool valid = active && !failed && playbackSucceeded && !crcMismatch && crcMessages > 0
		&& TheRecorder->getRecordedCRCComparisons() > 0 && samples > 0 && commands > 0;
	try {
		if (active) {
			flushSamples();
			Json footer = Json::object();
			footer.set("record", S("footer"));
			footer.set("valid", B(valid));
			footer.set("playback_completed", B(playbackSucceeded ? true : false));
			footer.set("crc_mismatch", B(crcMismatch ? true : false));
			footer.set("recorded_crc_messages_seen", U(crcMessages));
			footer.set("recorded_crc_messages_compared", U(TheRecorder->getRecordedCRCComparisons()));
			footer.set("replay_stream_eof", B(TheRecorder->reachedPlaybackEOF() ? true : false));
			footer.set("last_recorded_command_frame", U(TheRecorder->getLastPlaybackCommandFrame()));
			footer.set("samples", U(samples));
			footer.set("commands", U(commands));
			footer.set("wait_samples", U(waitSamples));
			footer.set("final_frame", U(TheGameLogic->getFrame()));
			footer.set("observed_sighting_removals", counts(observedSightingRemovals));
			footer.set("temporary_sightings_forgotten", counts(temporarySightingsForgotten));
			footer.set("result", S(valid ? "recorded simulation CRC checks passed" : "not training eligible"));
			write(footer, nullptr);
			if (fflush(output) != 0) throw std::runtime_error("Cannot write binary training export");
			closeOutput();
			std::string destination = finalPath;
			if (!valid) destination += ".failed";
			if (!MoveFileExA(temporaryPath.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
				throw std::runtime_error("Cannot finalize training output");
			printf("Training export: %u samples, %u commands, %u waits -> %s\n",
				samples, commands, waitSamples, destination.c_str());
		}
	} catch (const std::exception &e) {
		printf("Training export finish failed: %s\n", e.what());
		valid = false;
	}
	closeOutput();
	for (int p = 0; p < MAX_PLAYER_COUNT; ++p) pendingSamples[p] = Json();
	pendingCount = 0;
	pendingBuildCommands.clear();
	pendingResolvedCommands.clear();
	active = false;
	return valid;
}

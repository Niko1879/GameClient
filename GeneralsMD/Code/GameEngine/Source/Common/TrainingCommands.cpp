// GeneralsAI training export: canonical commands. See TrainingCommands.h.
// C++98 port of the Generals Online exporter's command layer (replay export only).
#include "PreRTS.h"
#include "Common/TrainingCommands.h"
#include "Common/TrainingExporter.h"
#include "Common/ActionManager.h"
#include "Common/MessageStream.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/PlayerTemplate.h"
#include "Common/Science.h"
#include "Common/SpecialPower.h"
#include "Common/ThingFactory.h"
#include "Common/ThingTemplate.h"
#include "Common/Upgrade.h"
#include "GameClient/ControlBar.h"
#include "GameLogic/AI.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/Weapon.h"
#include "GameLogic/Module/AIUpdate.h"
#include "GameLogic/Module/ContainModule.h"
#include "GameLogic/Module/ProductionUpdate.h"
#include "GameLogic/Module/SpecialAbilityUpdate.h"
#include "GameLogic/Module/SpecialPowerModule.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <map>

namespace training_commands {
namespace {
enum Family { FAMILY_UPGRADE, FAMILY_POWER, FAMILY_CONSTRUCT, FAMILY_WEAPON, FAMILY_SWITCH_WEAPON,
	FAMILY_HACK, FAMILY_OVERCHARGE, FAMILY_EVACUATE };
enum Target { TARGET_NONE, TARGET_POSITION, TARGET_OBJECT };
const char *const familyNames[] = {"upgrade", "power", "construct", "weapon", "switch_weapon",
	"hack_internet", "overcharge", "evacuate"};

struct Command {
	std::string name;
	const CommandButton *button;   // representative: every alias has identical message arguments
	Family family;
	Target target;
};

struct Catalogue {
	std::vector<Command> commands;
	std::map<const CommandButton *, size_t> canonical;
	bool built;
	Catalogue() : built(false) {}
} table;

typedef std::vector<const CommandButton *> Buttons;

bool familyOf(const CommandButton *button, Family &family)
{
	if (BitIsSet(button->getOptions(), SCRIPT_ONLY)) return false;
	switch (button->getCommandType()) {
	case GUI_COMMAND_PLAYER_UPGRADE: case GUI_COMMAND_OBJECT_UPGRADE:
		family = FAMILY_UPGRADE; return button->getUpgradeTemplate() != nullptr;
	case GUI_COMMAND_SPECIAL_POWER: case GUI_COMMAND_SPECIAL_POWER_FROM_SHORTCUT:
		family = FAMILY_POWER; return button->getSpecialPowerTemplate() != nullptr;
	case GUI_COMMAND_SPECIAL_POWER_CONSTRUCT: case GUI_COMMAND_SPECIAL_POWER_CONSTRUCT_FROM_SHORTCUT:
		family = FAMILY_CONSTRUCT; return button->getSpecialPowerTemplate() != nullptr;
	case GUI_COMMAND_FIRE_WEAPON: family = FAMILY_WEAPON; return true;
	case GUI_COMMAND_SWITCH_WEAPON: family = FAMILY_SWITCH_WEAPON; return true;
	case GUI_COMMAND_HACK_INTERNET: family = FAMILY_HACK; return true;
	case GUI_COMMAND_TOGGLE_OVERCHARGE: family = FAMILY_OVERCHARGE; return true;
	case GUI_COMMAND_EVACUATE: family = FAMILY_EVACUATE; return !BitIsSet(button->getOptions(), NEED_TARGET_POS);
	default: return false;
	}
}

Target targetOf(const CommandButton *button, Family family)
{
	const UnsignedInt options = button->getOptions();
	if (family == FAMILY_POWER || family == FAMILY_WEAPON) {
		if (BitIsSet(options, COMMAND_OPTION_NEED_OBJECT_TARGET)) return TARGET_OBJECT;
		if (BitIsSet(options, NEED_TARGET_POS)) return TARGET_POSITION;
	}
	return family == FAMILY_CONSTRUCT ? TARGET_POSITION : TARGET_NONE;
}

// Identical messages, arguments and availability rules: one canonical label.
std::string signature(const CommandButton *button, Family family)
{
	char number[32];
	std::string key = familyNames[family];
	const GUICommandType type = button->getCommandType();
	if (family == FAMILY_UPGRADE) key += type == GUI_COMMAND_PLAYER_UPGRADE ? "/player/" : "/object/";
	sprintf(number, "|%u", static_cast<unsigned>(button->getOptions()));
	key += number;
	if (button->getSpecialPowerTemplate()) key += "|sp=" + std::string(button->getSpecialPowerTemplate()->getName().str());
	if (button->getUpgradeTemplate()) key += "|up=" + std::string(button->getUpgradeTemplate()->getUpgradeName().str());
	if (family == FAMILY_WEAPON || family == FAMILY_SWITCH_WEAPON) {
		sprintf(number, "|slot=%d|shots=%d", static_cast<int>(button->getWeaponSlot()), static_cast<int>(button->getMaxShotsToFire()));
		key += number;
	}
	const ScienceVec &sciences = button->getScienceVec();
	for (size_t i = 0; i < sciences.size(); ++i)
		key += "|sci=" + std::string(TheScienceStore->getInternalNameForScience(sciences[i]).str());
	return key;
}

bool buttonNameLess(const CommandButton *a, const CommandButton *b)
{
	return strcmp(a->getName().str(), b->getName().str()) < 0;
}

bool classNameLess(const Buttons &a, const Buttons &b)
{
	return strcmp(a.front()->getName().str(), b.front()->getName().str()) < 0;
}

void build()
{
	if (table.built) return;
	std::map<std::string, Buttons> groups;
	for (const CommandButton *button = TheControlBar->getCommandButtons(); button; button = button->getNext()) {
		Family family;
		if (familyOf(button, family)) groups[signature(button, family)].push_back(button);
	}
	// The canonical name is the alphabetically first equivalent button.
	std::vector<Buttons> classes;
	for (std::map<std::string, Buttons>::iterator group = groups.begin(); group != groups.end(); ++group) {
		std::sort(group->second.begin(), group->second.end(), buttonNameLess);
		classes.push_back(group->second);
	}
	std::sort(classes.begin(), classes.end(), classNameLess);
	for (size_t c = 0; c < classes.size(); ++c) {
		const Buttons &buttons = classes[c];
		Family family;
		familyOf(buttons.front(), family);
		Command command;
		command.name = buttons.front()->getName().str();
		command.button = buttons.front();
		command.family = family;
		command.target = targetOf(buttons.front(), family);
		const size_t index = table.commands.size();
		table.commands.push_back(command);
		for (size_t b = 0; b < buttons.size(); ++b) table.canonical[buttons[b]] = index;
	}
	table.built = true;
}

const Command *canonicalFor(const CommandButton *button)
{
	build();
	std::map<const CommandButton *, size_t>::const_iterator found = table.canonical.find(button);
	return found == table.canonical.end() ? nullptr : &table.commands[found->second];
}

void setButtons(const AsciiString &name, Buttons &out)
{
	out.clear();
	if (name.isEmpty()) return;
	const CommandSet *set = TheControlBar->findCommandSet(name);
	if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
		const CommandButton *button = set->getCommandButton(i);
		if (button) out.push_back(button);
	}
}

// The UI evaluates shortcut powers on the local player's most-ready source and
// weapon switching on the local selection. Training always names the actor.
CommandAvailability availability(const CommandButton *button, Object *object)
{
	const GUICommandType type = button->getCommandType();
	if (type == GUI_COMMAND_SPECIAL_POWER_FROM_SHORTCUT || type == GUI_COMMAND_SPECIAL_POWER_CONSTRUCT_FROM_SHORTCUT) {
		if (object->testScriptStatusBit(OBJECT_STATUS_SCRIPT_DISABLED) || object->testScriptStatusBit(OBJECT_STATUS_SCRIPT_UNPOWERED)
			|| object->isDisabledByType(DISABLED_UNMANNED)) return COMMAND_HIDDEN;
		if (object->hasSingleUseCommandBeenUsed()) return COMMAND_RESTRICTED;
		bool disabled = object->isDisabled();
		if (disabled && BitIsSet(button->getOptions(), IGNORES_UNDERPOWERED)
			&& object->getDisabledFlags().test(DISABLED_UNDERPOWERED) && object->getDisabledFlags().count() == 1)
			disabled = false;
		if (disabled) return COMMAND_RESTRICTED;
		SpecialPowerModuleInterface *module = object->getSpecialPowerModule(button->getSpecialPowerTemplate());
		if (!module) return COMMAND_HIDDEN;
		if (!module->isReady()) return COMMAND_NOT_READY;
		SpecialAbilityUpdate *ability = object->findSpecialAbilityUpdate(button->getSpecialPowerTemplate()->getSpecialPowerType());
		if (ability && ability->isPowerCurrentlyInUse(button)) return COMMAND_RESTRICTED;
		return COMMAND_AVAILABLE;
	}
	if (type == GUI_COMMAND_SWITCH_WEAPON) {
		const CommandAvailability base = TheControlBar->evaluateCommandAvailability(button, object);
		if (base != COMMAND_AVAILABLE && base != COMMAND_ACTIVE) return base;
		WeaponSlotType current;
		const Weapon *weapon = object->getCurrentWeapon(&current);
		return weapon && current == button->getWeaponSlot() ? COMMAND_ACTIVE : COMMAND_AVAILABLE;
	}
	return TheControlBar->evaluateCommandAvailability(button, object);
}

bool scienceAllows(const CommandButton *button, Object *object)
{
	const SpecialPowerTemplate *power = button->getSpecialPowerTemplate();
	return !power || power->getRequiredScience() == SCIENCE_INVALID
		|| object->getControllingPlayer()->hasScience(power->getRequiredScience());
}

bool objectTargetLegal(const Command &command, Object *actor, Object *target)
{
	if (!target || target == actor || target->isEffectivelyDead()) return false;
	const CommandButton *button = command.button;
	if (!button->isValidObjectTarget(actor, target)) return false;
	if (command.family == FAMILY_POWER)
		return TheActionManager->canDoSpecialPowerAtObject(actor, target, CMD_FROM_PLAYER,
			button->getSpecialPowerTemplate(), button->getOptions()) ? true : false;
	return TheActionManager->canFireWeaponAtObject(actor, target, CMD_FROM_PLAYER, button->getWeaponSlot()) ? true : false;
}

// Usable now, up to the target: the button's UI state plus the dispatcher's checks.
CommandAvailability usable(const Command &command, const CommandButton *button, Object *actor)
{
	if (actor->isEffectivelyDead() || actor->getConstructionPercent() != -1) return COMMAND_HIDDEN;
	if (!scienceAllows(button, actor)) return COMMAND_HIDDEN;
	if ((command.family == FAMILY_POWER || command.family == FAMILY_CONSTRUCT)
		&& !actor->getSpecialPowerModule(button->getSpecialPowerTemplate())) return COMMAND_HIDDEN;
	CommandAvailability state = availability(button, actor);
	if (state == COMMAND_ACTIVE && command.family == FAMILY_OVERCHARGE) state = COMMAND_AVAILABLE;
	if (state != COMMAND_AVAILABLE) return state;
	if (command.family == FAMILY_POWER && command.target == TARGET_NONE
		&& !TheActionManager->canDoSpecialPower(actor, button->getSpecialPowerTemplate(), CMD_FROM_PLAYER, button->getOptions()))
		return COMMAND_RESTRICTED;
	if (command.family == FAMILY_WEAPON && command.target == TARGET_NONE
		&& !TheActionManager->canFireWeapon(actor, button->getWeaponSlot(), CMD_FROM_PLAYER))
		return COMMAND_RESTRICTED;
	return COMMAND_AVAILABLE;
}

float percentReady(const Command &command, const CommandButton *button, Object *actor)
{
	if (command.family == FAMILY_POWER || command.family == FAMILY_CONSTRUCT) {
		SpecialPowerModuleInterface *module = actor->getSpecialPowerModule(button->getSpecialPowerTemplate());
		return module ? module->getPercentReady() * 100.0f : 0.0f;
	}
	if (command.family == FAMILY_WEAPON) {
		const Weapon *weapon = actor->getWeaponInWeaponSlot(button->getWeaponSlot());
		return weapon ? weapon->getPercentReadyToFire() * 100.0f : 0.0f;
	}
	return 0.0f;
}

// Buttons the owner can press for this object: its command set, plus the
// faction's general's-power shortcuts for which this object is a source.
void objectButtons(Object *object, const CommandSet *set, Buttons &out)
{
	out.clear();
	if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
		const CommandButton *button = set->getCommandButton(i);
		if (button) out.push_back(button);
	}
	// ControlBar grafts a structure inventory (Command_Evacuate) onto garrisonable
	// buildings without a command set, such as civilian structures.
	const ContainModuleInterface *contain = object->getContain();
	if (contain && contain->isGarrisonable() && object->getCommandSetString().isEmpty()) {
		static const CommandButton *evacuate = TheControlBar->findCommandButton("Command_Evacuate");
		if (evacuate) out.push_back(evacuate);
	}
	const Player *owner = object->getControllingPlayer();
	const PlayerTemplate *faction = owner ? owner->getPlayerTemplate() : nullptr;
	if (!faction) return;
	// Command sets are static after INI load; avoid a name search per object.
	static std::map<const PlayerTemplate *, const CommandSet *> shortcuts;
	std::map<const PlayerTemplate *, const CommandSet *>::iterator found = shortcuts.find(faction);
	if (found == shortcuts.end()) {
		const AsciiString name = faction->getSpecialPowerShortcutCommandSet();
		const CommandSet *shortcutSet = name.isEmpty() ? nullptr : TheControlBar->findCommandSet(name);
		found = shortcuts.insert(std::make_pair(faction, shortcutSet)).first;
	}
	if (found->second) for (int k = 0; k < MAX_COMMANDS_PER_SET; ++k) {
		const CommandButton *button = found->second->getCommandButton(k);
		if (button && button->getSpecialPowerTemplate() && object->getSpecialPowerModule(button->getSpecialPowerTemplate()))
			out.push_back(button);
	}
}

TrainingJson point(const Coord3D &p)
{
	TrainingJson result = TrainingJson::array();
	result.push(TrainingJson::real(p.x));
	result.push(TrainingJson::real(p.y));
	result.push(TrainingJson::real(p.z));
	return result;
}

TrainingJson idOf(Object *object)
{
	return TrainingJson::unsignedInteger(static_cast<unsigned>(object->getID()));
}

std::vector<Object *> selected(const AIGroup *selection)
{
	std::vector<Object *> result;
	if (selection) {
		const std::vector<ObjectID> &ids = selection->getAllIDs();
		for (size_t i = 0; i < ids.size(); ++i) {
			Object *object = TheGameLogic->findObjectByID(ids[i]);
			if (object) result.push_back(object);
		}
	}
	return result;
}

TrainingJson ids(const std::vector<Object *> &objects)
{
	TrainingJson result = TrainingJson::array();
	for (size_t i = 0; i < objects.size(); ++i) result.push(idOf(objects[i]));
	return result;
}

bool canEnter(Object *actor, Object *target)
{
	return actor && target && target->getContain() && actor != target && !target->isEffectivelyDead()
		&& actor->getAIUpdateInterface() && !actor->getContainedBy()
		&& TheActionManager->canEnterObject(actor, target, CMD_FROM_PLAYER, CHECK_CAPACITY);
}

bool canEvacuate(Object *container)
{
	Buttons buttons;
	objectButtons(container, TheControlBar->findCommandSet(container->getCommandSetString()), buttons);
	for (size_t i = 0; i < buttons.size(); ++i) {
		const Command *command = canonicalFor(buttons[i]);
		if (command && command->family == FAMILY_EVACUATE && usable(*command, buttons[i], container) == COMMAND_AVAILABLE)
			return true;
	}
	return false;
}

// Guard orders need no command button (double-click guards), and the group
// dispatch reaches every member with an AI. Immobile objects do not guard.
bool canGuard(Object *object)
{
	return object && !object->isEffectivelyDead() && object->getAIUpdateInterface()
		&& !object->isKindOf(KINDOF_IMMOBILE) && object->getConstructionPercent() == -1;
}

// The player UI issues normal and air guard; no-pursuit guard is script-only.
const char *guardKind(int mode)
{
	return mode == GUARDMODE_NORMAL ? "guard" : mode == GUARDMODE_GUARD_FLYING_UNITS_ONLY ? "guard_air" : nullptr;
}

bool visibleTarget(int player, Object *target)
{
	return target && (target->getControllingPlayer() == ThePlayerList->getNthPlayer(player) || TrainingObjectVisibleTo(target, player));
}

bool canForceAttack(Object *object)
{
	return object && !object->isEffectivelyDead() && object->getConstructionPercent() == -1 && TrainingCanAttack(object);
}

bool canRepair(Object *dozer, Object *target)
{
	return dozer && target && dozer != target && !target->isEffectivelyDead() && dozer->getAIUpdateInterface()
		&& dozer->getAIUpdateInterface()->getDozerAIInterface()
		&& TheActionManager->canRepairObject(dozer, target, CMD_FROM_PLAYER);
}

bool canGetRepaired(Object *unit, Object *depot)
{
	return unit && depot && unit != depot && !depot->isEffectivelyDead() && depot->getDockUpdateInterface()
		&& unit->getAIUpdateInterface() && TheActionManager->canGetRepairedAt(unit, depot, CMD_FROM_PLAYER);
}

bool underConstruction(Object *object)
{
	return object && !object->isEffectivelyDead() && object->testStatus(OBJECT_STATUS_UNDER_CONSTRUCTION);
}

Object *singleSelected(const std::vector<Object *> &group)
{
	return group.size() == 1 ? group[0] : nullptr;
}

Object *argumentObject(const GameMessage *message, int index)
{
	if (message->getArgumentCount() <= index || message->getArgumentDataType(index) != ARGUMENTDATATYPE_OBJECTID) return nullptr;
	return TheGameLogic->findObjectByID(message->getArgument(index)->objectID);
}

bool fail(std::string &reason, const char *why)
{
	reason = why;
	return false;
}

// Does this button match the pressed message (resolve's button search)?
bool matches(GameMessage::Type type, const CommandButton *button, const UpgradeTemplate *upgrade,
	const SpecialPowerTemplate *power, int options, int slot, int shots)
{
	const GUICommandType t = button->getCommandType();
	switch (type) {
	case GameMessage::MSG_QUEUE_UPGRADE:
		return (t == GUI_COMMAND_PLAYER_UPGRADE || t == GUI_COMMAND_OBJECT_UPGRADE) && button->getUpgradeTemplate() == upgrade;
	case GameMessage::MSG_DO_SPECIAL_POWER: case GameMessage::MSG_DO_SPECIAL_POWER_AT_LOCATION:
	case GameMessage::MSG_DO_SPECIAL_POWER_AT_OBJECT:
		return button->getSpecialPowerTemplate() == power && static_cast<int>(button->getOptions()) == options
			&& t != GUI_COMMAND_FIRE_WEAPON;
	case GameMessage::MSG_DO_WEAPON: case GameMessage::MSG_DO_WEAPON_AT_LOCATION: case GameMessage::MSG_DO_WEAPON_AT_OBJECT:
		return t == GUI_COMMAND_FIRE_WEAPON && static_cast<int>(button->getWeaponSlot()) == slot && button->getMaxShotsToFire() == shots;
	case GameMessage::MSG_SWITCH_WEAPONS: return t == GUI_COMMAND_SWITCH_WEAPON && static_cast<int>(button->getWeaponSlot()) == slot;
	case GameMessage::MSG_INTERNET_HACK: return t == GUI_COMMAND_HACK_INTERNET;
	case GameMessage::MSG_TOGGLE_OVERCHARGE: return t == GUI_COMMAND_TOGGLE_OVERCHARGE;
	default: return false;
	}
}

bool containsName(const std::vector<const char *> &names, const char *name)
{
	for (size_t i = 0; i < names.size(); ++i) if (names[i] == name) return true;
	return false;
}
} // namespace

void Candidates::add(Object *object, bool dock)
{
	all.push_back(object);
	if (object->getContain()) containers.push_back(object);
	if (dock) depots.push_back(object);
}

void objectCommands(Object *object, const CommandSet *set, const Candidates &candidates, ObjectCommands &result)
{
	build();
	result.available.clear(); result.charging.clear(); result.targets.clear();
	result.enterTargets.clear(); result.upgrades.clear(); result.repairTargets.clear(); result.repairDepots.clear();
	result.occupants = result.capacity = result.containerID = 0;
	result.canGuard = canGuard(object);
	static Buttons buttons;
	objectButtons(object, set, buttons);
	for (size_t b = 0; b < buttons.size(); ++b) {
		const CommandButton *button = buttons[b];
		const Command *command = canonicalFor(button);
		if (!command) continue;
		const char *name = command->name.c_str();
		if (containsName(result.available, name)) continue;
		const CommandAvailability state = usable(*command, button, object);
		if (state == COMMAND_NOT_READY) {
			bool charging = false;
			for (size_t c = 0; c < result.charging.size(); ++c) if (result.charging[c].first == name) charging = true;
			if (!charging) result.charging.push_back(std::make_pair(name, percentReady(*command, button, object)));
			continue;
		}
		if (state != COMMAND_AVAILABLE) continue;
		if (command->target == TARGET_OBJECT) {
			std::vector<unsigned> legal;
			for (size_t t = 0; t < candidates.all.size(); ++t)
				if (objectTargetLegal(*command, object, candidates.all[t]))
					legal.push_back(static_cast<unsigned>(candidates.all[t]->getID()));
			if (legal.empty()) continue;
			result.targets.push_back(std::make_pair(name, legal));
		}
		result.available.push_back(name);
	}
	// canEnter requires a contain module and canGetRepaired a dock interface.
	if (object->getAIUpdateInterface() && !object->isKindOf(KINDOF_STRUCTURE))
		for (size_t c = 0; c < candidates.containers.size(); ++c)
			if (canEnter(object, candidates.containers[c]))
				result.enterTargets.push_back(static_cast<unsigned>(candidates.containers[c]->getID()));
	const bool dozer = object->getAIUpdateInterface() && object->getAIUpdateInterface()->getDozerAIInterface();
	if (dozer)
		for (size_t a = 0; a < candidates.all.size(); ++a)
			if (candidates.all[a]->getControllingPlayer() == object->getControllingPlayer() && canRepair(object, candidates.all[a]))
				result.repairTargets.push_back(static_cast<unsigned>(candidates.all[a]->getID()));
	if (result.canGuard)
		for (size_t d = 0; d < candidates.depots.size(); ++d)
			if (canGetRepaired(object, candidates.depots[d]))
				result.repairDepots.push_back(static_cast<unsigned>(candidates.depots[d]->getID()));
	if (object->getObjectCompletedUpgradeMask().any())
		for (const UpgradeTemplate *t = TheUpgradeCenter->firstUpgradeTemplate(); t; t = t->friend_getNext())
			if (t->getUpgradeType() == UPGRADE_TYPE_OBJECT && object->hasUpgrade(t)) result.upgrades.push_back(t->getUpgradeName().str());
	const ContainModuleInterface *contain = object->getContain();
	if (contain) {
		result.occupants = contain->getContainCount();
		const Int maximum = contain->getContainMax();
		result.capacity = maximum > 0 ? static_cast<unsigned>(maximum) : 0;
	}
	const Object *container = object->getContainedBy();
	if (container && visibleTarget(object->getControllingPlayer()->getPlayerIndex(), const_cast<Object *>(container)))
		result.containerID = static_cast<unsigned>(container->getID());
}

void playerCommands(Player *player, PlayerCommands &result)
{
	build();
	result.sciences.clear(); result.purchasableSciences.clear(); result.upgrades.clear();
	static std::vector<std::string> names;
	static std::vector<ScienceType> sciences;
	if (names.empty()) {
		const std::vector<AsciiString> all = TheScienceStore->friend_getScienceNames();
		for (size_t i = 0; i < all.size(); ++i) {
			names.push_back(all[i].str());
			sciences.push_back(TheScienceStore->getScienceFromInternalName(all[i]));
		}
	}
	for (size_t s = 0; s < names.size(); ++s)
		if (player->hasScience(sciences[s])) result.sciences.push_back(names[s].c_str());
	const PlayerTemplate *faction = player->getPlayerTemplate();
	if (faction) {
		AsciiString sets[3];
		sets[0] = faction->getPurchaseScienceCommandSetRank1();
		sets[1] = faction->getPurchaseScienceCommandSetRank3();
		sets[2] = faction->getPurchaseScienceCommandSetRank8();
		Buttons buttons;
		for (int r = 0; r < 3; ++r) {
			setButtons(sets[r], buttons);
			for (size_t b = 0; b < buttons.size(); ++b) {
				const CommandButton *button = buttons[b];
				if (button->getCommandType() != GUI_COMMAND_PURCHASE_SCIENCE) continue;
				const ScienceVec &offered = button->getScienceVec();
				for (size_t o = 0; o < offered.size(); ++o) {
					const ScienceType science = offered[o];
					if (player->hasScience(science) || !TheScienceStore->playerHasPrereqsForScience(player, science)
						|| TheScienceStore->getSciencePurchaseCost(science) > player->getSciencePurchasePoints()) continue;
					const std::string name = TheScienceStore->getInternalNameForScience(science).str();
					const std::vector<std::string>::const_iterator found = std::find(names.begin(), names.end(), name);
					if (found != names.end() && !containsName(result.purchasableSciences, found->c_str()))
						result.purchasableSciences.push_back(found->c_str());
				}
			}
		}
	}
	for (const UpgradeTemplate *t = TheUpgradeCenter->firstUpgradeTemplate(); t; t = t->friend_getNext())
		if (t->getUpgradeType() == UPGRADE_TYPE_PLAYER && player->hasUpgradeComplete(t)) result.upgrades.push_back(t->getUpgradeName().str());
}

bool covers(const GameMessage *message)
{
	switch (message->getType()) {
	case GameMessage::MSG_QUEUE_UPGRADE: case GameMessage::MSG_PURCHASE_SCIENCE:
	case GameMessage::MSG_DO_SPECIAL_POWER: case GameMessage::MSG_DO_SPECIAL_POWER_AT_LOCATION:
	case GameMessage::MSG_DO_SPECIAL_POWER_AT_OBJECT: case GameMessage::MSG_DO_WEAPON:
	case GameMessage::MSG_DO_WEAPON_AT_LOCATION: case GameMessage::MSG_DO_WEAPON_AT_OBJECT:
	case GameMessage::MSG_SWITCH_WEAPONS: case GameMessage::MSG_INTERNET_HACK: case GameMessage::MSG_TOGGLE_OVERCHARGE:
	case GameMessage::MSG_ENTER: case GameMessage::MSG_EXIT: case GameMessage::MSG_EVACUATE:
	case GameMessage::MSG_DO_GUARD_POSITION: case GameMessage::MSG_CANCEL_UNIT_CREATE: case GameMessage::MSG_CANCEL_UPGRADE:
	case GameMessage::MSG_DOZER_CANCEL_CONSTRUCT: case GameMessage::MSG_DO_SCATTER: case GameMessage::MSG_ADD_WAYPOINT:
	case GameMessage::MSG_DO_FORCE_ATTACK_GROUND: case GameMessage::MSG_DO_FORCE_ATTACK_OBJECT:
	case GameMessage::MSG_DO_REPAIR: case GameMessage::MSG_GET_REPAIRED:
		return true;
	default: return false;
	}
}

bool resolve(const GameMessage *message, const AIGroup *selection, TrainingJson &action, std::string &reason)
{
	build();
	Player *owner = ThePlayerList->getNthPlayer(message->getPlayerIndex());
	const int player = owner ? owner->getPlayerIndex() : -1;
	std::vector<Object *> group;
	{
		const std::vector<Object *> all = selected(selection);
		for (size_t i = 0; i < all.size(); ++i)
			if (!all[i]->isEffectivelyDead() && all[i]->getControllingPlayer() == owner) group.push_back(all[i]);
	}
	const GameMessage::Type type = message->getType();
	action = TrainingJson::object();
	if (type == GameMessage::MSG_PURCHASE_SCIENCE) {
		const ScienceType science = static_cast<ScienceType>(message->getArgument(0)->integer);
		PlayerCommands state;
		playerCommands(owner, state);
		const std::string name = TheScienceStore->getInternalNameForScience(science).str();
		bool purchasable = false;
		for (size_t i = 0; i < state.purchasableSciences.size(); ++i) if (name == state.purchasableSciences[i]) purchasable = true;
		if (!purchasable) return fail(reason, "Science is not purchasable now");
		action.set("kind", TrainingJson::string("science"));
		action.set("science", TrainingJson::string(name));
		return true;
	}
	if (type == GameMessage::MSG_ENTER) {
		Object *target = argumentObject(message, 1);
		if (group.empty() || !target) return fail(reason, "Enter needs a selection and container");
		if (!visibleTarget(player, target)) return fail(reason, "Container not currently observable");
		// The dispatcher sends every selected member; keep only eligible entrants.
		std::vector<Object *> entrants;
		for (size_t i = 0; i < group.size(); ++i) if (canEnter(group[i], target)) entrants.push_back(group[i]);
		if (entrants.size() != group.size()) return fail(reason, "Not every selected unit can enter");
		action.set("kind", TrainingJson::string("enter"));
		action.set("units", ids(entrants));
		action.set("target", idOf(target));
		return true;
	}
	if (type == GameMessage::MSG_DO_GUARD_POSITION) {
		const char *kind = guardKind(message->getArgument(1)->integer);
		if (!kind) return fail(reason, "Guard mode is not a player command");
		// Members without an AI are ignored by the dispatcher; keep the guards.
		std::vector<Object *> guards;
		for (size_t i = 0; i < group.size(); ++i) if (canGuard(group[i])) guards.push_back(group[i]);
		if (guards.empty()) return fail(reason, "No selected unit can guard");
		action.set("kind", TrainingJson::string(kind));
		action.set("units", ids(guards));
		action.set("position", point(message->getArgument(0)->location));
		return true;
	}
	if (type == GameMessage::MSG_CANCEL_UNIT_CREATE || type == GameMessage::MSG_CANCEL_UPGRADE) {
		Object *producer = singleSelected(group);
		if (!producer) return fail(reason, "Cancel needs one selected producer");
		std::vector<const ProductionEntry *> entries;
		ProductionUpdateInterface *production = producer->getProductionUpdateInterface();
		if (production)
			for (const ProductionEntry *p = production->firstProduction(); p; p = production->nextProduction(p)) entries.push_back(p);
		const UpgradeTemplate *upgrade = type == GameMessage::MSG_CANCEL_UPGRADE
			? TheUpgradeCenter->findUpgradeByKey(static_cast<NameKeyType>(message->getArgument(0)->integer)) : nullptr;
		for (size_t slot = 0; slot < entries.size(); ++slot) {
			const bool match = type == GameMessage::MSG_CANCEL_UNIT_CREATE
				? entries[slot]->getProductionType() == PRODUCTION_UNIT
					&& entries[slot]->getProductionID() == static_cast<ProductionID>(message->getArgument(0)->integer)
				: upgrade && entries[slot]->getProductionType() == PRODUCTION_UPGRADE
					&& entries[slot]->getProductionUpgrade() == upgrade;
			if (match) {
				action.set("kind", TrainingJson::string("cancel_production"));
				action.set("producer", idOf(producer));
				action.set("slot", TrainingJson::unsignedInteger(slot));
				return true;
			}
		}
		return fail(reason, "Cancelled entry is not in the producer queue");
	}
	if (type == GameMessage::MSG_DOZER_CANCEL_CONSTRUCT) {
		Object *building = singleSelected(group);
		if (!underConstruction(building)) return fail(reason, "Cancel needs one selected foundation");
		TrainingJson units = TrainingJson::array();
		units.push(idOf(building));
		action.set("kind", TrainingJson::string("cancel_construction"));
		action.set("units", units);
		return true;
	}
	if (type == GameMessage::MSG_DO_SCATTER || type == GameMessage::MSG_ADD_WAYPOINT
		|| type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND || type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT
		|| type == GameMessage::MSG_DO_REPAIR || type == GameMessage::MSG_GET_REPAIRED) {
		const bool targeted = type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT || type == GameMessage::MSG_DO_REPAIR
			|| type == GameMessage::MSG_GET_REPAIRED;
		Object *target = targeted ? argumentObject(message, 0) : nullptr;
		if (targeted && !target) return fail(reason, "Command target is missing");
		if (target && !visibleTarget(player, target)) return fail(reason, "Command target not currently observable");
		// The group dispatch reaches every eligible member; others are ignored.
		std::vector<Object *> users;
		for (size_t i = 0; i < group.size(); ++i) {
			Object *o = group[i];
			const bool eligible = type == GameMessage::MSG_DO_SCATTER || type == GameMessage::MSG_ADD_WAYPOINT ? canGuard(o)
				: type == GameMessage::MSG_DO_REPAIR ? canRepair(o, target)
				: type == GameMessage::MSG_GET_REPAIRED ? canGetRepaired(o, target) : canForceAttack(o) && o != target;
			if (eligible) users.push_back(o);
		}
		if (users.empty()) return fail(reason, "No selected unit can carry out the command");
		const char *kind = type == GameMessage::MSG_DO_SCATTER ? "scatter" : type == GameMessage::MSG_ADD_WAYPOINT ? "add_waypoint"
			: type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND ? "force_attack_ground"
			: type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT ? "force_attack"
			: type == GameMessage::MSG_DO_REPAIR ? "repair" : "get_repaired";
		action.set("kind", TrainingJson::string(kind));
		action.set("units", ids(users));
		if (target) action.set("target", idOf(target));
		if (type == GameMessage::MSG_ADD_WAYPOINT || type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND)
			action.set("position", point(message->getArgument(0)->location));
		return true;
	}
	if (type == GameMessage::MSG_EXIT) {
		Object *occupant = argumentObject(message, 0);
		if (!occupant || occupant->getControllingPlayer() != owner || !occupant->getContainedBy())
			return fail(reason, "Exit needs an owned contained unit");
		TrainingJson units = TrainingJson::array();
		units.push(idOf(occupant));
		action.set("kind", TrainingJson::string("exit"));
		action.set("units", units);
		return true;
	}
	if (type == GameMessage::MSG_EVACUATE) {
		if (group.empty()) return fail(reason, "Evacuate needs a selection");
		for (size_t i = 0; i < group.size(); ++i) if (!canEvacuate(group[i])) return fail(reason, "Selected object cannot evacuate");
		action.set("kind", TrainingJson::string("evacuate"));
		action.set("units", ids(group));
		return true;
	}
	// Remaining messages name a command button: find the one pressed.
	Object *source = nullptr;
	int powerID = -1, slot = -1, shots = -1, options = -1;
	const UpgradeTemplate *upgrade = nullptr;
	Object *target = nullptr;
	Coord3D position;
	position.x = position.y = position.z = 0;
	bool positioned = false;
	switch (type) {
	case GameMessage::MSG_QUEUE_UPGRADE:
		upgrade = TheUpgradeCenter->findUpgradeByKey(static_cast<NameKeyType>(message->getArgument(1)->integer)); break;
	case GameMessage::MSG_DO_SPECIAL_POWER:
		powerID = message->getArgument(0)->integer; options = message->getArgument(1)->integer; source = argumentObject(message, 2); break;
	case GameMessage::MSG_DO_SPECIAL_POWER_AT_LOCATION: {
		const bool angle = message->getArgumentCount() >= 6;
		powerID = message->getArgument(0)->integer; position = message->getArgument(1)->location; positioned = true;
		options = message->getArgument(angle ? 4 : 3)->integer; source = argumentObject(message, angle ? 5 : 4); break;
	}
	case GameMessage::MSG_DO_SPECIAL_POWER_AT_OBJECT:
		powerID = message->getArgument(0)->integer; target = argumentObject(message, 1);
		options = message->getArgument(2)->integer; source = argumentObject(message, 3); break;
	case GameMessage::MSG_DO_WEAPON:
		slot = message->getArgument(0)->integer; shots = message->getArgument(1)->integer; break;
	case GameMessage::MSG_DO_WEAPON_AT_LOCATION:
		slot = message->getArgument(0)->integer; position = message->getArgument(1)->location; positioned = true;
		shots = message->getArgument(2)->integer; target = argumentObject(message, 3); break;
	case GameMessage::MSG_DO_WEAPON_AT_OBJECT:
		slot = message->getArgument(0)->integer; target = argumentObject(message, 1); shots = message->getArgument(2)->integer; break;
	case GameMessage::MSG_SWITCH_WEAPONS: slot = message->getArgument(0)->integer; break;
	default: break;
	}
	std::vector<Object *> users;
	if (source) users.push_back(source); else users = group;
	if (users.empty()) return fail(reason, "Command needs a selected or explicit source");
	if (source && source->getControllingPlayer() != owner) return fail(reason, "Command source is not owned");
	const SpecialPowerTemplate *power = powerID >= 0 ? TheSpecialPowerStore->findSpecialPowerTemplateByID(powerID) : nullptr;
	if (powerID >= 0 && !power) return fail(reason, "Unknown special power");
	if (type == GameMessage::MSG_QUEUE_UPGRADE && !upgrade) return fail(reason, "Unknown upgrade");
	const Command *resolved = nullptr;
	Buttons buttons;
	for (size_t u = 0; u < users.size(); ++u) {
		Object *user = users[u];
		const Command *found = nullptr;
		const CommandButton *pressed = nullptr;
		objectButtons(user, TheControlBar->findCommandSet(user->getCommandSetString()), buttons);
		for (size_t b = 0; b < buttons.size() && !found; ++b)
			if (matches(type, buttons[b], upgrade, power, options, slot, shots) && canonicalFor(buttons[b])) {
				found = canonicalFor(buttons[b]);
				pressed = buttons[b];
			}
		if (!found) return fail(reason, "No matching command button for a selected object");
		if (resolved && resolved != found) return fail(reason, "Selected objects disagree on the command");
		if (usable(*found, pressed, user) != COMMAND_AVAILABLE) return fail(reason, "Command is not available to the actor");
		resolved = found;
	}
	// A position weapon fired at an object (ATTACK_OBJECTS_POSITION) is an object-target command.
	if (resolved->target == TARGET_OBJECT) {
		if (!target || !visibleTarget(player, target)) return fail(reason, "Command target not currently observable");
		for (size_t u = 0; u < users.size(); ++u)
			if (!objectTargetLegal(*resolved, users[u], target)) return fail(reason, "Illegal command target");
	} else if (resolved->target == TARGET_POSITION) {
		if (!positioned) return fail(reason, "Position command without a position");
		for (size_t u = 0; u < users.size(); ++u)
			if (resolved->family == FAMILY_POWER && !TheActionManager->canDoSpecialPowerAtLocation(users[u], &position, CMD_FROM_PLAYER,
					resolved->button->getSpecialPowerTemplate(), nullptr, resolved->button->getOptions()))
				return fail(reason, "Illegal command location");
	}
	if (resolved->family == FAMILY_UPGRADE) {
		if (users.size() != 1) return fail(reason, "Upgrade requires a single producer");
		action.set("kind", TrainingJson::string("upgrade"));
		action.set("producer", idOf(users[0]));
		action.set("command", TrainingJson::string(resolved->name));
		return true;
	}
	action.set("kind", TrainingJson::string("ability"));
	action.set("units", ids(users));
	action.set("command", TrainingJson::string(resolved->name));
	if (resolved->target == TARGET_OBJECT) action.set("target", idOf(target));
	if (resolved->target == TARGET_POSITION) action.set("position", point(position));
	return true;
}

} // namespace training_commands

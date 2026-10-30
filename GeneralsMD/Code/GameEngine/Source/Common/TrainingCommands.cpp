#include "PreRTS.h"
#define DEFINE_GUI_COMMAND_NAMES
#include "Common/TrainingCommands.h"
#include "Common/ActionManager.h"
#include "Common/MessageStream.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/PlayerTemplate.h"
#include "Common/Science.h"
#include "Common/SpecialPower.h"
#include "Common/ThingFactory.h"
#include "Common/ThingTemplate.h"
#include "Common/TrainingExporter.h"
#include "Common/Upgrade.h"
#include "GameClient/ControlBar.h"
#include "GameLogic/AI.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/TerrainLogic.h"
#include "GameLogic/Weapon.h"
#include "GameLogic/Module/AIUpdate.h"
#include "GameLogic/Module/ContainModule.h"
#include "GameLogic/Module/ProductionUpdate.h"
#include "GameLogic/Module/SpecialAbilityUpdate.h"
#include "GameLogic/Module/SpecialPowerModule.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace training_commands {
namespace {
enum class Family { Upgrade, Power, Construct, Weapon, SwitchWeapon, Hack, Overcharge, Evacuate };
enum class Target { None, Position, Object };
const char *const familyNames[] = {"upgrade", "power", "construct", "weapon", "switch_weapon",
                                   "hack_internet", "overcharge", "evacuate"};
const char *const targetNames[] = {"none", "position", "object"};

struct Command {
    std::string name;
    const CommandButton *button;   // representative: every alias has identical message arguments
    Family family;
    Target target;
};

struct Catalogue {
    std::vector<Command> commands;
    std::unordered_map<const CommandButton *, size_t> canonical;
    std::map<std::string, size_t> byName;
    bool built = false;
} table;

bool familyOf(const CommandButton *button, Family &family)
{
    if (BitIsSet(button->getOptions(), SCRIPT_ONLY)) return false;
    switch (button->getCommandType()) {
    case GUI_COMMAND_PLAYER_UPGRADE: case GUI_COMMAND_OBJECT_UPGRADE:
        family = Family::Upgrade; return button->getUpgradeTemplate() != nullptr;
    case GUI_COMMAND_SPECIAL_POWER: case GUI_COMMAND_SPECIAL_POWER_FROM_SHORTCUT:
        family = Family::Power; return button->getSpecialPowerTemplate() != nullptr;
    case GUI_COMMAND_SPECIAL_POWER_CONSTRUCT: case GUI_COMMAND_SPECIAL_POWER_CONSTRUCT_FROM_SHORTCUT:
        family = Family::Construct; return button->getSpecialPowerTemplate() != nullptr;
    case GUI_COMMAND_FIRE_WEAPON: family = Family::Weapon; return true;
    case GUI_COMMAND_SWITCH_WEAPON: family = Family::SwitchWeapon; return true;
    case GUI_COMMAND_HACK_INTERNET: family = Family::Hack; return true;
    case GUI_COMMAND_TOGGLE_OVERCHARGE: family = Family::Overcharge; return true;
    case GUI_COMMAND_EVACUATE: family = Family::Evacuate; return !BitIsSet(button->getOptions(), NEED_TARGET_POS);
    default: return false;
    }
}

Target targetOf(const CommandButton *button, Family family)
{
    const UnsignedInt options = button->getOptions();
    if (family == Family::Power || family == Family::Weapon) {
        if (BitIsSet(options, COMMAND_OPTION_NEED_OBJECT_TARGET)) return Target::Object;
        if (BitIsSet(options, NEED_TARGET_POS)) return Target::Position;
    }
    return family == Family::Construct ? Target::Position : Target::None;
}

// Identical messages, arguments and availability rules: one canonical label.
std::string signature(const CommandButton *button, Family family)
{
    std::string key = familyNames[static_cast<int>(family)];
    const GUICommandType type = button->getCommandType();
    if (family == Family::Upgrade) key += type == GUI_COMMAND_PLAYER_UPGRADE ? "/player/" : "/object/";
    key += "|" + std::to_string(button->getOptions());
    if (button->getSpecialPowerTemplate()) key += "|sp=" + std::string(button->getSpecialPowerTemplate()->getName().str());
    if (button->getUpgradeTemplate()) key += "|up=" + std::string(button->getUpgradeTemplate()->getUpgradeName().str());
    if (family == Family::Weapon || family == Family::SwitchWeapon)
        key += "|slot=" + std::to_string(button->getWeaponSlot()) + "|shots=" + std::to_string(button->getMaxShotsToFire());
    for (ScienceType science : button->getScienceVec())
        key += "|sci=" + std::string(TheScienceStore->getInternalNameForScience(science).str());
    return key;
}

void build()
{
    if (table.built) return;
    std::map<std::string, std::vector<const CommandButton *>> groups;
    for (const CommandButton *button = TheControlBar->getCommandButtons(); button; button = button->getNext()) {
        Family family;
        if (familyOf(button, family)) groups[signature(button, family)].push_back(button);
    }
    // The canonical name is the alphabetically first equivalent button.
    std::vector<std::vector<const CommandButton *>> classes;
    for (auto &group : groups) {
        auto &buttons = group.second;
        std::sort(buttons.begin(), buttons.end(), [](const CommandButton *a, const CommandButton *b) {
            return std::strcmp(a->getName().str(), b->getName().str()) < 0;
        });
        classes.push_back(buttons);
    }
    std::sort(classes.begin(), classes.end(), [](const auto &a, const auto &b) {
        return std::strcmp(a.front()->getName().str(), b.front()->getName().str()) < 0;
    });
    for (const auto &buttons : classes) {
        Family family;
        familyOf(buttons.front(), family);
        const size_t index = table.commands.size();
        table.commands.push_back({buttons.front()->getName().str(), buttons.front(), family, targetOf(buttons.front(), family)});
        for (const CommandButton *button : buttons) table.canonical[button] = index;
        table.byName[table.commands.back().name] = index;
    }
    table.built = true;
}

const Command *canonicalFor(const CommandButton *button)
{
    build();
    const auto found = table.canonical.find(button);
    return found == table.canonical.end() ? nullptr : &table.commands[found->second];
}

const Command &commandNamed(const std::string &name)
{
    build();
    const auto found = table.byName.find(name);
    if (found == table.byName.end()) throw std::runtime_error("Unknown command");
    return table.commands[found->second];
}

const PlayerTemplate *templateNamed(const AsciiString &name)
{
    for (int i = 0; i < ThePlayerTemplateStore->getPlayerTemplateCount(); ++i) {
        const PlayerTemplate *t = ThePlayerTemplateStore->getNthPlayerTemplate(i);
        if (t && t->getName() == name) return t;
    }
    return nullptr;
}

template<class Visit> void visitSet(const AsciiString &name, Visit visit)
{
    if (name.isEmpty()) return;
    const CommandSet *set = TheControlBar->findCommandSet(name);
    if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i)
        if (const CommandButton *button = set->getCommandButton(i)) visit(button);
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
        if (SpecialAbilityUpdate *ability = object->findSpecialAbilityUpdate(button->getSpecialPowerTemplate()->getSpecialPowerType()))
            if (ability->isPowerCurrentlyInUse(button)) return COMMAND_RESTRICTED;
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
    if (command.family == Family::Power)
        return TheActionManager->canDoSpecialPowerAtObject(actor, target, CMD_FROM_PLAYER,
            button->getSpecialPowerTemplate(), button->getOptions());
    return TheActionManager->canFireWeaponAtObject(actor, target, CMD_FROM_PLAYER, button->getWeaponSlot());
}

// Usable now, up to the target: the button's UI state plus the dispatcher's checks.
CommandAvailability usable(const Command &command, const CommandButton *button, Object *actor)
{
    if (actor->isEffectivelyDead() || actor->getConstructionPercent() != -1) return COMMAND_HIDDEN;
    if (!scienceAllows(button, actor)) return COMMAND_HIDDEN;
    if ((command.family == Family::Power || command.family == Family::Construct)
        && !actor->getSpecialPowerModule(button->getSpecialPowerTemplate())) return COMMAND_HIDDEN;
    CommandAvailability state = availability(button, actor);
    if (state == COMMAND_ACTIVE && command.family == Family::Overcharge) state = COMMAND_AVAILABLE;
    if (state != COMMAND_AVAILABLE) return state;
    if (command.family == Family::Power && command.target == Target::None
        && !TheActionManager->canDoSpecialPower(actor, button->getSpecialPowerTemplate(), CMD_FROM_PLAYER, button->getOptions()))
        return COMMAND_RESTRICTED;
    if (command.family == Family::Weapon && command.target == Target::None
        && !TheActionManager->canFireWeapon(actor, button->getWeaponSlot(), CMD_FROM_PLAYER))
        return COMMAND_RESTRICTED;
    return COMMAND_AVAILABLE;
}

float percentReady(const Command &command, const CommandButton *button, Object *actor)
{
    if (command.family == Family::Power || command.family == Family::Construct) {
        SpecialPowerModuleInterface *module = actor->getSpecialPowerModule(button->getSpecialPowerTemplate());
        return module ? module->getPercentReady() * 100.0f : 0.0f;
    }
    if (command.family == Family::Weapon) {
        const Weapon *weapon = actor->getWeaponInWeaponSlot(button->getWeaponSlot());
        return weapon ? weapon->getPercentReadyToFire() * 100.0f : 0.0f;
    }
    return 0.0f;
}

// Buttons the owner can press for this object: its command set, plus the
// faction's general's-power shortcuts for which this object is a source.
template<class Visit> void visitButtons(Object *object, const CommandSet *set, Visit visit)
{
    if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i)
        if (const CommandButton *button = set->getCommandButton(i)) visit(button);
    // ControlBar grafts a structure inventory (Command_Evacuate) onto garrisonable
    // buildings without a command set, such as civilian structures.
    const ContainModuleInterface *contain = object->getContain();
    if (contain && contain->isGarrisonable() && object->getCommandSetString().isEmpty()) {
        static const CommandButton *evacuate = TheControlBar->findCommandButton("Command_Evacuate");
        if (evacuate) visit(evacuate);
    }
    const Player *owner = object->getControllingPlayer();
    const PlayerTemplate *faction = owner ? owner->getPlayerTemplate() : nullptr;
    if (!faction) return;
    // Command sets are static after INI load; avoid a name search per object.
    static std::unordered_map<const PlayerTemplate *, const CommandSet *> shortcuts;
    auto found = shortcuts.find(faction);
    if (found == shortcuts.end()) {
        const AsciiString name = faction->getSpecialPowerShortcutCommandSet();
        found = shortcuts.emplace(faction, name.isEmpty() ? nullptr : TheControlBar->findCommandSet(name)).first;
    }
    if (found->second) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i)
        if (const CommandButton *button = found->second->getCommandButton(i))
            if (button->getSpecialPowerTemplate() && object->getSpecialPowerModule(button->getSpecialPowerTemplate())) visit(button);
}

const CommandButton *pressable(Object *object, const Command &command)
{
    const CommandButton *found = nullptr;
    visitButtons(object, TheControlBar->findCommandSet(object->getCommandSetString()), [&](const CommandButton *button) {
        if (!found && canonicalFor(button) == &command) found = button;
    });
    return found;
}

Object *owned(int player, const Json &id)
{
    Object *object = TheGameLogic->findObjectByID(static_cast<ObjectID>(id.get<unsigned>()));
    if (!object || object->isEffectivelyDead() || !object->getControllingPlayer()
        || object->getControllingPlayer()->getPlayerIndex() != player)
        throw std::runtime_error("Unit is missing or not owned by this player");
    return object;
}

Coord3D location(const Json &value)
{
    if (!value.is_array() || (value.size() != 2 && value.size() != 3)) throw std::runtime_error("Expected position [x,y,z]");
    Coord3D p = {value[0].get<float>(), value[1].get<float>(), 0};
    Region3D extent;
    TheTerrainLogic->getExtent(&extent);
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < extent.lo.x || p.x > extent.hi.x
        || p.y < extent.lo.y || p.y > extent.hi.y) throw std::runtime_error("Position outside map");
    p.z = TheTerrainLogic->getGroundHeight(p.x, p.y);
    return p;
}

Json point(const Coord3D &p) { return Json::array({p.x, p.y, p.z}); }

GameMessage *message(GameMessage::Type type, int player)
{
    GameMessage *m = newInstance(GameMessage)(type);
    m->friend_setPlayerIndex(player);
    TheCommandList->appendMessage(m);
    return m;
}

void select(int player, const std::vector<Object *> &objects)
{
    message(GameMessage::MSG_DESTROY_SELECTED_GROUP, player);
    GameMessage *m = message(GameMessage::MSG_CREATE_SELECTED_GROUP, player);
    m->appendBooleanArgument(TRUE);
    for (Object *object : objects) m->appendObjectIDArgument(object->getID());
}

std::vector<Object *> actors(int player, const Json &units, size_t limit)
{
    if (!units.is_array() || units.empty() || units.size() > limit) throw std::runtime_error("Invalid actor list");
    std::vector<Object *> result;
    for (const Json &id : units) {
        Object *object = owned(player, id);
        if (std::find(result.begin(), result.end(), object) != result.end()) throw std::runtime_error("Duplicate actor");
        result.push_back(object);
    }
    return result;
}

std::vector<Object *> selected(const AIGroup *selection)
{
    std::vector<Object *> result;
    if (selection) for (ObjectID id : selection->getAllIDs())
        if (Object *object = TheGameLogic->findObjectByID(id)) result.push_back(object);
    return result;
}

Json ids(const std::vector<Object *> &objects)
{
    Json result = Json::array();
    for (Object *object : objects) result.push_back(static_cast<unsigned>(object->getID()));
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
    bool offered = false;
    visitButtons(container, TheControlBar->findCommandSet(container->getCommandSetString()), [&](const CommandButton *button) {
        const Command *command = canonicalFor(button);
        if (!offered && command && command->family == Family::Evacuate && usable(*command, button, container) == COMMAND_AVAILABLE)
            offered = true;
    });
    return offered;
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

int guardMode(const std::string &kind)
{
    return kind == "guard" ? GUARDMODE_NORMAL : kind == "guard_air" ? GUARDMODE_GUARD_FLYING_UNITS_ONLY : -1;
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

// Queue slots in the exported production_queue order (the UI's build queue).
std::vector<const ProductionEntry *> productionQueue(Object *producer)
{
    std::vector<const ProductionEntry *> entries;
    if (ProductionUpdateInterface *production = producer->getProductionUpdateInterface())
        for (const ProductionEntry *p = production->firstProduction(); p; p = production->nextProduction(p)) entries.push_back(p);
    return entries;
}

Object *singleSelected(const std::vector<Object *> &group)
{
    return group.size() == 1 ? group[0] : nullptr;
}
} // namespace

Json catalog()
{
    build();
    Json result = Json::object();
    Json commands = Json::object(), aliases = Json::object();
    for (const Command &command : table.commands) {
        const CommandButton *button = command.button;
        Json entry = {{"family", familyNames[static_cast<int>(command.family)]},
                      {"kind", command.family == Family::Upgrade ? "upgrade" : command.family == Family::Evacuate ? "evacuate" : "ability"},
                      {"target", targetNames[static_cast<int>(command.target)]},
                      {"button_type", TheGuiCommandNames[button->getCommandType()]},
                      {"options", button->getOptions()}};
        if (button->getSpecialPowerTemplate()) entry["special_power"] = button->getSpecialPowerTemplate()->getName().str();
        if (button->getUpgradeTemplate()) entry["upgrade"] = button->getUpgradeTemplate()->getUpgradeName().str();
        if (command.family == Family::Weapon || command.family == Family::SwitchWeapon) {
            entry["weapon_slot"] = static_cast<int>(button->getWeaponSlot());
            entry["max_shots"] = button->getMaxShotsToFire();
        }
        commands[command.name] = std::move(entry);
    }
    for (const auto &entry : table.canonical)
        if (entry.first->getName().str() != table.commands[entry.second].name)
            aliases[entry.first->getName().str()] = table.commands[entry.second].name;
    Json menus = Json::object();
    for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate()) {
        Json names = Json::array();
        visitSet(t->friend_getCommandSetString(), [&](const CommandButton *button) {
            if (const Command *command = canonicalFor(button)) names.push_back(command->name);
        });
        if (!names.empty()) menus[t->getName().str()] = std::move(names);
    }
    Json shortcuts = Json::object(), promotions = Json::object();
    for (int i = 0; i < ThePlayerTemplateStore->getPlayerTemplateCount(); ++i) {
        const PlayerTemplate *faction = ThePlayerTemplateStore->getNthPlayerTemplate(i);
        if (!faction || !faction->isPlayableSide()) continue;
        Json names = Json::array(), sciences = Json::array();
        visitSet(faction->getSpecialPowerShortcutCommandSet(), [&](const CommandButton *button) {
            if (const Command *command = canonicalFor(button)) names.push_back(command->name);
        });
        for (const AsciiString &set : {faction->getPurchaseScienceCommandSetRank1(), faction->getPurchaseScienceCommandSetRank3(),
                                        faction->getPurchaseScienceCommandSetRank8()})
            visitSet(set, [&](const CommandButton *button) {
                if (button->getCommandType() == GUI_COMMAND_PURCHASE_SCIENCE)
                    for (ScienceType science : button->getScienceVec())
                        sciences.push_back(TheScienceStore->getInternalNameForScience(science).str());
            });
        shortcuts[faction->getName().str()] = std::move(names);
        promotions[faction->getName().str()] = std::move(sciences);
    }
    Json upgrades = Json::array();
    for (const UpgradeTemplate *t = TheUpgradeCenter->firstUpgradeTemplate(); t; t = t->friend_getNext())
        upgrades.push_back(t->getUpgradeName().str());
    Json sciences = Json::array();
    for (const AsciiString &name : TheScienceStore->friend_getScienceNames()) sciences.push_back(name.str());
    result["commands"] = std::move(commands);
    result["aliases"] = std::move(aliases);
    result["menus"] = std::move(menus);
    result["shortcuts"] = std::move(shortcuts);
    result["promotions"] = std::move(promotions);
    result["upgrades"] = std::move(upgrades);
    result["sciences"] = std::move(sciences);
    return result;
}

// Opt-in export profiling; outside observations and game state.
class PartTimer {
    TrainingCommandsProfilePart part;
    bool enabled;
    std::chrono::steady_clock::time_point started;
public:
    explicit PartTimer(TrainingCommandsProfilePart value) : part(value), enabled(TrainingExportProfilingEnabled()) {
        if (enabled) started = std::chrono::steady_clock::now();
    }
    ~PartTimer() {
        if (enabled) TrainingExportProfileCommands(part, std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
    }
};

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
    {
        PartTimer buttons(TC_BUTTONS);
        visitButtons(object, set, [&](const CommandButton *button) {
            const Command *command = canonicalFor(button);
            if (!command) return;
            const char *name = command->name.c_str();
            if (std::find(result.available.begin(), result.available.end(), name) != result.available.end()) return;
            const CommandAvailability state = usable(*command, button, object);
            if (state == COMMAND_NOT_READY) {
                if (std::none_of(result.charging.begin(), result.charging.end(), [name](const auto &item) { return item.first == name; }))
                    result.charging.emplace_back(name, percentReady(*command, button, object));
                return;
            }
            if (state != COMMAND_AVAILABLE) return;
            if (command->target == Target::Object) {
                PartTimer timer(TC_OBJECT_TARGETS);
                std::vector<unsigned> legal;
                for (Object *target : candidates.all)
                    if (objectTargetLegal(*command, object, target))
                        legal.push_back(static_cast<unsigned>(target->getID()));
                if (legal.empty()) return;
                result.targets.emplace_back(name, std::move(legal));
            }
            result.available.push_back(name);
        });
    }
    PartTimer rest(TC_TARGETS);
    // canEnter requires a contain module and canGetRepaired a dock interface.
    if (object->getAIUpdateInterface() && !object->isKindOf(KINDOF_STRUCTURE))
        for (Object *target : candidates.containers)
            if (canEnter(object, target))
                result.enterTargets.push_back(static_cast<unsigned>(target->getID()));
    const bool dozer = object->getAIUpdateInterface() && object->getAIUpdateInterface()->getDozerAIInterface();
    if (dozer)
        for (Object *target : candidates.all)
            if (target->getControllingPlayer() == object->getControllingPlayer() && canRepair(object, target))
                result.repairTargets.push_back(static_cast<unsigned>(target->getID()));
    if (result.canGuard)
        for (Object *target : candidates.depots)
            if (canGetRepaired(object, target))
                result.repairDepots.push_back(static_cast<unsigned>(target->getID()));
    if (object->getObjectCompletedUpgradeMask().any())
        for (const UpgradeTemplate *t = TheUpgradeCenter->firstUpgradeTemplate(); t; t = t->friend_getNext())
            if (t->getUpgradeType() == UPGRADE_TYPE_OBJECT && object->hasUpgrade(t)) result.upgrades.push_back(t->getUpgradeName().str());
    if (const ContainModuleInterface *contain = object->getContain()) {
        result.occupants = contain->getContainCount();
        result.capacity = static_cast<unsigned>(std::max(0, contain->getContainMax()));
    }
    if (const Object *container = object->getContainedBy())
        if (visibleTarget(object->getControllingPlayer()->getPlayerIndex(), const_cast<Object *>(container))) result.containerID = static_cast<unsigned>(container->getID());
}

void playerCommands(Player *player, PlayerCommands &result)
{
    build();
    result.sciences.clear(); result.purchasableSciences.clear(); result.upgrades.clear();
    static std::vector<std::string> names;
    if (names.empty()) for (const AsciiString &name : TheScienceStore->friend_getScienceNames()) names.push_back(name.str());
    for (const std::string &name : names)
        if (player->hasScience(TheScienceStore->getScienceFromInternalName(name.c_str()))) result.sciences.push_back(name.c_str());
    const PlayerTemplate *faction = player->getPlayerTemplate();
    if (faction) for (const AsciiString &set : {faction->getPurchaseScienceCommandSetRank1(), faction->getPurchaseScienceCommandSetRank3(),
                                                 faction->getPurchaseScienceCommandSetRank8()})
        visitSet(set, [&](const CommandButton *button) {
            if (button->getCommandType() != GUI_COMMAND_PURCHASE_SCIENCE) return;
            for (ScienceType science : button->getScienceVec()) {
                if (player->hasScience(science) || !TheScienceStore->playerHasPrereqsForScience(player, science)
                    || TheScienceStore->getSciencePurchaseCost(science) > player->getSciencePurchasePoints()) continue;
                const std::string name = TheScienceStore->getInternalNameForScience(science).str();
                const auto found = std::find(names.begin(), names.end(), name);
                if (found != names.end() && std::find(result.purchasableSciences.begin(), result.purchasableSciences.end(),
                        found->c_str()) == result.purchasableSciences.end())
                    result.purchasableSciences.push_back(found->c_str());
            }
        });
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

bool resolve(const GameMessage *message, const AIGroup *selection, Json &action, std::string &reason)
{
    build();
    Player *owner = ThePlayerList->getNthPlayer(message->getPlayerIndex());
    const int player = owner ? owner->getPlayerIndex() : -1;
    std::vector<Object *> group = selected(selection);
    group.erase(std::remove_if(group.begin(), group.end(), [owner](Object *o) {
        return o->isEffectivelyDead() || o->getControllingPlayer() != owner; }), group.end());
    auto fail = [&](const char *why) { reason = why; return false; };
    auto argumentObject = [message](int index) -> Object * {
        if (message->getArgumentCount() <= index || message->getArgumentDataType(index) != ARGUMENTDATATYPE_OBJECTID) return nullptr;
        return TheGameLogic->findObjectByID(message->getArgument(index)->objectID);
    };
    const GameMessage::Type type = message->getType();
    if (type == GameMessage::MSG_PURCHASE_SCIENCE) {
        const ScienceType science = static_cast<ScienceType>(message->getArgument(0)->integer);
        PlayerCommands state;
        playerCommands(owner, state);
        const std::string name = TheScienceStore->getInternalNameForScience(science).str();
        if (std::none_of(state.purchasableSciences.begin(), state.purchasableSciences.end(),
                         [&](const char *s) { return name == s; })) return fail("Science is not purchasable now");
        action = {{"kind", "science"}, {"science", name}};
        return true;
    }
    if (type == GameMessage::MSG_ENTER) {
        Object *target = argumentObject(1);
        if (group.empty() || !target) return fail("Enter needs a selection and container");
        if (!visibleTarget(player, target)) return fail("Container not currently observable");
        // The dispatcher sends every selected member; keep only eligible entrants.
        std::vector<Object *> entrants;
        for (Object *o : group) if (canEnter(o, target)) entrants.push_back(o);
        if (entrants.size() != group.size()) return fail("Not every selected unit can enter");
        action = {{"kind", "enter"}, {"units", ids(entrants)}, {"target", static_cast<unsigned>(target->getID())}};
        return true;
    }
    if (type == GameMessage::MSG_DO_GUARD_POSITION) {
        const char *kind = guardKind(message->getArgument(1)->integer);
        if (!kind) return fail("Guard mode is not a player command");
        // Members without an AI are ignored by the dispatcher; keep the guards.
        std::vector<Object *> guards;
        for (Object *o : group) if (canGuard(o)) guards.push_back(o);
        if (guards.empty()) return fail("No selected unit can guard");
        action = {{"kind", kind}, {"units", ids(guards)}, {"position", point(message->getArgument(0)->location)}};
        return true;
    }
    if (type == GameMessage::MSG_CANCEL_UNIT_CREATE || type == GameMessage::MSG_CANCEL_UPGRADE) {
        Object *producer = singleSelected(group);
        if (!producer) return fail("Cancel needs one selected producer");
        const std::vector<const ProductionEntry *> entries = productionQueue(producer);
        const UpgradeTemplate *upgrade = type == GameMessage::MSG_CANCEL_UPGRADE
            ? TheUpgradeCenter->findUpgradeByKey(static_cast<NameKeyType>(message->getArgument(0)->integer)) : nullptr;
        for (size_t slot = 0; slot < entries.size(); ++slot) {
            const bool match = type == GameMessage::MSG_CANCEL_UNIT_CREATE
                ? entries[slot]->getProductionType() == PRODUCTION_UNIT
                    && entries[slot]->getProductionID() == static_cast<ProductionID>(message->getArgument(0)->integer)
                : upgrade && entries[slot]->getProductionType() == PRODUCTION_UPGRADE
                    && entries[slot]->getProductionUpgrade() == upgrade;
            if (match) {
                action = {{"kind", "cancel_production"}, {"producer", static_cast<unsigned>(producer->getID())}, {"slot", slot}};
                return true;
            }
        }
        return fail("Cancelled entry is not in the producer queue");
    }
    if (type == GameMessage::MSG_DOZER_CANCEL_CONSTRUCT) {
        Object *building = singleSelected(group);
        if (!underConstruction(building)) return fail("Cancel needs one selected foundation");
        action = {{"kind", "cancel_construction"}, {"units", Json::array({static_cast<unsigned>(building->getID())})}};
        return true;
    }
    if (type == GameMessage::MSG_DO_SCATTER || type == GameMessage::MSG_ADD_WAYPOINT
        || type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND || type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT
        || type == GameMessage::MSG_DO_REPAIR || type == GameMessage::MSG_GET_REPAIRED) {
        const bool targeted = type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT || type == GameMessage::MSG_DO_REPAIR
            || type == GameMessage::MSG_GET_REPAIRED;
        Object *target = targeted ? argumentObject(0) : nullptr;
        if (targeted && !target) return fail("Command target is missing");
        if (target && !visibleTarget(player, target)) return fail("Command target not currently observable");
        // The group dispatch reaches every eligible member; others are ignored.
        std::vector<Object *> users;
        for (Object *o : group) {
            const bool eligible = type == GameMessage::MSG_DO_SCATTER || type == GameMessage::MSG_ADD_WAYPOINT ? canGuard(o)
                : type == GameMessage::MSG_DO_REPAIR ? canRepair(o, target)
                : type == GameMessage::MSG_GET_REPAIRED ? canGetRepaired(o, target) : canForceAttack(o) && o != target;
            if (eligible) users.push_back(o);
        }
        if (users.empty()) return fail("No selected unit can carry out the command");
        const char *kind = type == GameMessage::MSG_DO_SCATTER ? "scatter" : type == GameMessage::MSG_ADD_WAYPOINT ? "add_waypoint"
            : type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND ? "force_attack_ground"
            : type == GameMessage::MSG_DO_FORCE_ATTACK_OBJECT ? "force_attack"
            : type == GameMessage::MSG_DO_REPAIR ? "repair" : "get_repaired";
        action = {{"kind", kind}, {"units", ids(users)}};
        if (target) action["target"] = static_cast<unsigned>(target->getID());
        if (type == GameMessage::MSG_ADD_WAYPOINT || type == GameMessage::MSG_DO_FORCE_ATTACK_GROUND)
            action["position"] = point(message->getArgument(0)->location);
        return true;
    }
    if (type == GameMessage::MSG_EXIT) {
        Object *occupant = argumentObject(0);
        if (!occupant || occupant->getControllingPlayer() != owner || !occupant->getContainedBy())
            return fail("Exit needs an owned contained unit");
        action = {{"kind", "exit"}, {"units", Json::array({static_cast<unsigned>(occupant->getID())})}};
        return true;
    }
    if (type == GameMessage::MSG_EVACUATE) {
        if (group.empty()) return fail("Evacuate needs a selection");
        for (Object *o : group) if (!canEvacuate(o)) return fail("Selected object cannot evacuate");
        action = {{"kind", "evacuate"}, {"units", ids(group)}};
        return true;
    }
    // Remaining messages name a command button: find the one pressed.
    Object *source = nullptr;
    int powerID = -1, slot = -1, shots = -1, options = -1;
    const UpgradeTemplate *upgrade = nullptr;
    Object *target = nullptr;
    Coord3D position{};
    bool positioned = false;
    switch (type) {
    case GameMessage::MSG_QUEUE_UPGRADE:
        upgrade = TheUpgradeCenter->findUpgradeByKey(static_cast<NameKeyType>(message->getArgument(1)->integer)); break;
    case GameMessage::MSG_DO_SPECIAL_POWER:
        powerID = message->getArgument(0)->integer; options = message->getArgument(1)->integer; source = argumentObject(2); break;
    case GameMessage::MSG_DO_SPECIAL_POWER_AT_LOCATION: {
        const bool angle = message->getArgumentCount() >= 6;
        powerID = message->getArgument(0)->integer; position = message->getArgument(1)->location; positioned = true;
        options = message->getArgument(angle ? 4 : 3)->integer; source = argumentObject(angle ? 5 : 4); break;
    }
    case GameMessage::MSG_DO_SPECIAL_POWER_AT_OBJECT:
        powerID = message->getArgument(0)->integer; target = argumentObject(1);
        options = message->getArgument(2)->integer; source = argumentObject(3); break;
    case GameMessage::MSG_DO_WEAPON:
        slot = message->getArgument(0)->integer; shots = message->getArgument(1)->integer; break;
    case GameMessage::MSG_DO_WEAPON_AT_LOCATION:
        slot = message->getArgument(0)->integer; position = message->getArgument(1)->location; positioned = true;
        shots = message->getArgument(2)->integer; target = argumentObject(3); break;
    case GameMessage::MSG_DO_WEAPON_AT_OBJECT:
        slot = message->getArgument(0)->integer; target = argumentObject(1); shots = message->getArgument(2)->integer; break;
    case GameMessage::MSG_SWITCH_WEAPONS: slot = message->getArgument(0)->integer; break;
    default: break;
    }
    std::vector<Object *> users = source ? std::vector<Object *>{source} : group;
    if (users.empty()) return fail("Command needs a selected or explicit source");
    if (source && source->getControllingPlayer() != owner) return fail("Command source is not owned");
    const SpecialPowerTemplate *power = powerID >= 0 ? TheSpecialPowerStore->findSpecialPowerTemplateByID(powerID) : nullptr;
    if (powerID >= 0 && !power) return fail("Unknown special power");
    if (type == GameMessage::MSG_QUEUE_UPGRADE && !upgrade) return fail("Unknown upgrade");
    auto matches = [&](const CommandButton *button) {
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
    };
    const Command *resolved = nullptr;
    for (Object *user : users) {
        const Command *found = nullptr;
        const CommandButton *pressed = nullptr;
        visitButtons(user, TheControlBar->findCommandSet(user->getCommandSetString()), [&](const CommandButton *button) {
            if (!found && matches(button) && canonicalFor(button)) { found = canonicalFor(button); pressed = button; }
        });
        if (!found) return fail("No matching command button for a selected object");
        if (resolved && resolved != found) return fail("Selected objects disagree on the command");
        if (usable(*found, pressed, user) != COMMAND_AVAILABLE) return fail("Command is not available to the actor");
        resolved = found;
    }
    // A position weapon fired at an object (ATTACK_OBJECTS_POSITION) is an object-target command.
    if (resolved->target == Target::Object) {
        if (!target || !visibleTarget(player, target)) return fail("Command target not currently observable");
        for (Object *user : users) if (!objectTargetLegal(*resolved, user, target)) return fail("Illegal command target");
    } else if (resolved->target == Target::Position) {
        if (!positioned) return fail("Position command without a position");
        for (Object *user : users)
            if (resolved->family == Family::Power && !TheActionManager->canDoSpecialPowerAtLocation(user, &position, CMD_FROM_PLAYER,
                    resolved->button->getSpecialPowerTemplate(), nullptr, resolved->button->getOptions()))
                return fail("Illegal command location");
    }
    if (resolved->family == Family::Upgrade) {
        if (users.size() != 1) return fail("Upgrade requires a single producer");
        action = {{"kind", "upgrade"}, {"producer", static_cast<unsigned>(users[0]->getID())}, {"command", resolved->name}};
        return true;
    }
    action = {{"kind", "ability"}, {"units", ids(users)}, {"command", resolved->name}};
    if (resolved->target == Target::Object) action["target"] = static_cast<unsigned>(target->getID());
    if (resolved->target == Target::Position) action["position"] = point(position);
    return true;
}

void validate(int player, const Json &action)
{
    build();
    const std::string kind = action.at("kind").get<std::string>();
    Player *owner = ThePlayerList->getNthPlayer(player);
    if (kind == "science") {
        PlayerCommands state;
        playerCommands(owner, state);
        const std::string name = action.at("science").get<std::string>();
        if (std::none_of(state.purchasableSciences.begin(), state.purchasableSciences.end(), [&](const char *s) { return name == s; }))
            throw std::runtime_error("Science is not purchasable now");
        return;
    }
    if (kind == "enter") {
        const std::vector<Object *> units = actors(player, action.at("units"), 64);
        Object *target = TheGameLogic->findObjectByID(static_cast<ObjectID>(action.at("target").get<unsigned>()));
        // Identical rejection for hidden, dead or nonexistent containers.
        if (!visibleTarget(player, target)) throw std::runtime_error("Container is not currently observable");
        for (Object *unit : units) if (!canEnter(unit, target)) throw std::runtime_error("Unit cannot enter this container");
        return;
    }
    if (guardMode(kind) >= 0) {
        for (Object *unit : actors(player, action.at("units"), 64))
            if (!canGuard(unit)) throw std::runtime_error("Unit cannot guard");
        location(action.at("position"));
        return;
    }
    if (kind == "cancel_production") {
        Object *producer = actors(player, Json::array({action.at("producer")}), 1)[0];
        if (action.at("slot").get<size_t>() >= productionQueue(producer).size())
            throw std::runtime_error("No production entry in that queue slot");
        return;
    }
    if (kind == "cancel_construction") {
        if (!underConstruction(actors(player, action.at("units"), 1)[0])) throw std::runtime_error("Not an unfinished foundation");
        return;
    }
    if (kind == "scatter" || kind == "add_waypoint" || kind == "force_attack" || kind == "force_attack_ground"
        || kind == "repair" || kind == "get_repaired") {
        Object *target = nullptr;
        if (kind == "force_attack" || kind == "repair" || kind == "get_repaired") {
            target = TheGameLogic->findObjectByID(static_cast<ObjectID>(action.at("target").get<unsigned>()));
            // Identical rejection for hidden, dead or nonexistent targets.
            if (!target || target->isEffectivelyDead() || !visibleTarget(player, target))
                throw std::runtime_error("Command target is not currently observable");
        }
        for (Object *unit : actors(player, action.at("units"), 64)) {
            const bool eligible = kind == "scatter" || kind == "add_waypoint" ? canGuard(unit)
                : kind == "repair" ? canRepair(unit, target) : kind == "get_repaired" ? canGetRepaired(unit, target)
                : canForceAttack(unit) && unit != target;
            if (!eligible) throw std::runtime_error("Unit cannot carry out this command");
        }
        if (kind == "add_waypoint" || kind == "force_attack_ground") location(action.at("position"));
        return;
    }
    if (kind == "exit") {
        const std::vector<Object *> units = actors(player, action.at("units"), 1);
        if (!units[0]->getContainedBy() || units[0]->getContainedBy()->isDisabledByType(DISABLED_SUBDUED))
            throw std::runtime_error("Unit cannot exit now");
        return;
    }
    if (kind == "evacuate") {
        for (Object *unit : actors(player, action.at("units"), 64))
            if (!canEvacuate(unit)) throw std::runtime_error("Object cannot evacuate");
        return;
    }
    if (kind != "upgrade" && kind != "ability") throw std::runtime_error("Unsupported command kind");
    const Command &command = commandNamed(action.at("command").get<std::string>());
    if ((kind == "upgrade") != (command.family == Family::Upgrade) || command.family == Family::Evacuate)
        throw std::runtime_error("Command does not match action kind");
    const std::vector<Object *> users = kind == "upgrade" ? actors(player, Json::array({action.at("producer")}), 1)
                                                          : actors(player, action.at("units"), 64);
    for (Object *user : users) {
        const CommandButton *button = pressable(user, command);
        if (!button || usable(command, button, user) != COMMAND_AVAILABLE) throw std::runtime_error("Command is not available");
    }
    if (command.target == Target::Object) {
        Object *target = TheGameLogic->findObjectByID(static_cast<ObjectID>(action.at("target").get<unsigned>()));
        if (!visibleTarget(player, target)) throw std::runtime_error("Command target is not currently observable");
        for (Object *user : users) if (!objectTargetLegal(command, user, target)) throw std::runtime_error("Illegal command target");
    } else if (command.target == Target::Position) {
        const Coord3D p = location(action.at("position"));
        for (Object *user : users)
            if (command.family == Family::Power && !TheActionManager->canDoSpecialPowerAtLocation(user, &p, CMD_FROM_PLAYER,
                    command.button->getSpecialPowerTemplate(), nullptr, command.button->getOptions()))
                throw std::runtime_error("Illegal command location");
    }
}

void apply(int player, const Json &action)
{
    build();
    const std::string kind = action.at("kind").get<std::string>();
    if (kind == "science") {
        message(GameMessage::MSG_PURCHASE_SCIENCE, player)->appendIntegerArgument(
            TheScienceStore->getScienceFromInternalName(action.at("science").get<std::string>().c_str()));
        return;
    }
    if (kind == "enter") {
        select(player, actors(player, action.at("units"), 64));
        GameMessage *m = message(GameMessage::MSG_ENTER, player);
        m->appendObjectIDArgument(INVALID_ID);
        m->appendObjectIDArgument(static_cast<ObjectID>(action.at("target").get<unsigned>()));
        return;
    }
    if (kind == "cancel_production") {
        Object *producer = actors(player, Json::array({action.at("producer")}), 1)[0];
        const ProductionEntry *entry = productionQueue(producer).at(action.at("slot").get<size_t>());
        select(player, {producer});
        // The unit and upgrade pointers share a union; the type is authoritative.
        if (entry->getProductionType() == PRODUCTION_UPGRADE)
            message(GameMessage::MSG_CANCEL_UPGRADE, player)->appendIntegerArgument(entry->getProductionUpgrade()->getUpgradeNameKey());
        else message(GameMessage::MSG_CANCEL_UNIT_CREATE, player)->appendIntegerArgument(entry->getProductionID());
        return;
    }
    if (kind == "cancel_construction") {
        select(player, actors(player, action.at("units"), 1));
        message(GameMessage::MSG_DOZER_CANCEL_CONSTRUCT, player);
        return;
    }
    if (kind == "scatter" || kind == "add_waypoint" || kind == "force_attack" || kind == "force_attack_ground"
        || kind == "repair" || kind == "get_repaired") {
        select(player, actors(player, action.at("units"), 64));
        const GameMessage::Type type = kind == "scatter" ? GameMessage::MSG_DO_SCATTER : kind == "add_waypoint"
            ? GameMessage::MSG_ADD_WAYPOINT : kind == "force_attack" ? GameMessage::MSG_DO_FORCE_ATTACK_OBJECT
            : kind == "force_attack_ground" ? GameMessage::MSG_DO_FORCE_ATTACK_GROUND : kind == "repair"
            ? GameMessage::MSG_DO_REPAIR : GameMessage::MSG_GET_REPAIRED;
        GameMessage *m = message(type, player);
        if (kind == "add_waypoint" || kind == "force_attack_ground") m->appendLocationArgument(location(action.at("position")));
        if (action.contains("target")) m->appendObjectIDArgument(static_cast<ObjectID>(action.at("target").get<unsigned>()));
        return;
    }
    if (guardMode(kind) >= 0) {
        select(player, actors(player, action.at("units"), 64));
        GameMessage *m = message(GameMessage::MSG_DO_GUARD_POSITION, player);
        m->appendLocationArgument(location(action.at("position")));
        m->appendIntegerArgument(guardMode(kind));
        return;
    }
    if (kind == "exit") {
        Object *occupant = actors(player, action.at("units"), 1)[0];
        select(player, {occupant->getContainedBy()});
        message(GameMessage::MSG_EXIT, player)->appendObjectIDArgument(occupant->getID());
        return;
    }
    if (kind == "evacuate") {
        select(player, actors(player, action.at("units"), 64));
        message(GameMessage::MSG_EVACUATE, player);
        return;
    }
    const Command &command = commandNamed(action.at("command").get<std::string>());
    const CommandButton *button = command.button;
    const std::vector<Object *> users = kind == "upgrade" ? actors(player, Json::array({action.at("producer")}), 1)
                                                          : actors(player, action.at("units"), 64);
    select(player, users);
    // A single source is named explicitly, as the shortcut bar does; groups use the selection.
    const ObjectID source = users.size() == 1 ? users[0]->getID() : INVALID_ID;
    Object *target = command.target == Target::Object
        ? TheGameLogic->findObjectByID(static_cast<ObjectID>(action.at("target").get<unsigned>())) : nullptr;
    switch (command.family) {
    case Family::Upgrade: {
        GameMessage *m = message(GameMessage::MSG_QUEUE_UPGRADE, player);
        m->appendObjectIDArgument(users[0]->getID());
        m->appendIntegerArgument(button->getUpgradeTemplate()->getUpgradeNameKey());
        break;
    }
    case Family::Power: case Family::Construct: {
        const UnsignedInt id = button->getSpecialPowerTemplate()->getID();
        if (command.target == Target::Object) {
            GameMessage *m = message(GameMessage::MSG_DO_SPECIAL_POWER_AT_OBJECT, player);
            m->appendIntegerArgument(id); m->appendObjectIDArgument(target->getID());
            m->appendIntegerArgument(button->getOptions()); m->appendObjectIDArgument(source);
        } else if (command.target == Target::Position) {
            GameMessage *m = message(GameMessage::MSG_DO_SPECIAL_POWER_AT_LOCATION, player);
            m->appendIntegerArgument(id); m->appendLocationArgument(location(action.at("position")));
            m->appendRealArgument(command.family == Family::Construct ? 0.0f : INVALID_ANGLE);
            m->appendObjectIDArgument(INVALID_ID); m->appendIntegerArgument(button->getOptions());
            m->appendObjectIDArgument(source);
        } else {
            GameMessage *m = message(GameMessage::MSG_DO_SPECIAL_POWER, player);
            m->appendIntegerArgument(id); m->appendIntegerArgument(button->getOptions()); m->appendObjectIDArgument(source);
        }
        break;
    }
    case Family::Weapon:
        // The UI arms the mine-clearing weapon set before such an order.
        if (BitIsSet(button->getOptions(), USES_MINE_CLEARING_WEAPONSET))
            message(GameMessage::MSG_SET_MINE_CLEARING_DETAIL, player);
        if (command.target == Target::Object && BitIsSet(button->getOptions(), ATTACK_OBJECTS_POSITION)) {
            GameMessage *m = message(GameMessage::MSG_DO_WEAPON_AT_LOCATION, player);
            m->appendIntegerArgument(button->getWeaponSlot()); m->appendLocationArgument(*target->getPosition());
            m->appendIntegerArgument(button->getMaxShotsToFire()); m->appendObjectIDArgument(target->getID());
        } else if (command.target == Target::Object) {
            GameMessage *m = message(GameMessage::MSG_DO_WEAPON_AT_OBJECT, player);
            m->appendIntegerArgument(button->getWeaponSlot()); m->appendObjectIDArgument(target->getID());
            m->appendIntegerArgument(button->getMaxShotsToFire());
        } else if (command.target == Target::Position) {
            GameMessage *m = message(GameMessage::MSG_DO_WEAPON_AT_LOCATION, player);
            m->appendIntegerArgument(button->getWeaponSlot()); m->appendLocationArgument(location(action.at("position")));
            m->appendIntegerArgument(button->getMaxShotsToFire()); m->appendObjectIDArgument(INVALID_ID);
        } else {
            GameMessage *m = message(GameMessage::MSG_DO_WEAPON, player);
            m->appendIntegerArgument(button->getWeaponSlot()); m->appendIntegerArgument(button->getMaxShotsToFire());
        }
        break;
    case Family::SwitchWeapon: message(GameMessage::MSG_SWITCH_WEAPONS, player)->appendIntegerArgument(button->getWeaponSlot()); break;
    case Family::Hack: message(GameMessage::MSG_INTERNET_HACK, player); break;
    case Family::Overcharge: message(GameMessage::MSG_TOGGLE_OVERCHARGE, player); break;
    case Family::Evacuate: throw std::runtime_error("Evacuate is its own action kind");
    }
}
} // namespace training_commands

#include "PreRTS.h"
#include "Common/TrainingExporter.h"
#include "Common/ReplayLogic.h"
#include "Common/TrainingCommands.h"
#include "Common/TrainingSightingMemory.h"
#include "Common/ActionManager.h"
#include "Common/Energy.h"
#include "Common/GlobalData.h"
#include "Common/MessageStream.h"
#include "Common/Money.h"
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
#include "GameLogic/PartitionManager.h"
#include "GameClient/ControlBar.h"
#include "GameClient/Drawable.h"
#include "GameLogic/TerrainLogic.h"
#include "GameLogic/Module/AIUpdate.h"
#include "GameLogic/AI.h"
#include "GameLogic/AIStateMachine.h"
#include "GameLogic/Module/DozerAIUpdate.h"
#include "GameLogic/Module/SupplyTruckAIUpdate.h"
#include "GameLogic/Module/StealthUpdate.h"
#include "GameLogic/Module/BodyModule.h"
#include "GameLogic/Module/ProductionUpdate.h"
#include "GameLogic/Module/ContainModule.h"
#include "GameLogic/Module/SpawnBehavior.h"
#include "GameLogic/Module/LifetimeUpdate.h"
#include "GameNetwork/GeneralsOnline/json.hpp"
#include "Common/TrainingBinary.h"
#include "gitinfo.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <optional>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {
// Counts allocations owned by the exporter's JSON containers and string objects.
// std::string character buffers and allocations in engine code are not included.
bool profileAllocations = false;
struct AllocationCounter { unsigned long long calls = 0, bytes = 0; };
AllocationCounter jsonAllocations;
template<class T> struct ExportAllocator {
    using value_type = T;
    ExportAllocator() = default;
    template<class U> ExportAllocator(const ExportAllocator<U> &) noexcept {}
    T *allocate(size_t count) {
        T *result = std::allocator<T>{}.allocate(count);
        if (profileAllocations) {
            ++jsonAllocations.calls;
            jsonAllocations.bytes += count * sizeof(T);
        }
        return result;
    }
    void deallocate(T *pointer, size_t count) noexcept { std::allocator<T>{}.deallocate(pointer, count); }
    template<class U> bool operator==(const ExportAllocator<U> &) const noexcept { return true; }
    template<class U> bool operator!=(const ExportAllocator<U> &) const noexcept { return false; }
};
using Json = nlohmann::basic_json<nlohmann::ordered_map, std::vector, std::string,
    bool, std::int64_t, std::uint64_t, double, ExportAllocator>;
constexpr int gridSize = 32;
std::filesystem::path outputDirectory, temporaryPath, finalPath;
std::ofstream output;
bool binaryFormat = false;
std::unique_ptr<training_binary::Writer<Json>> binaryWriter;
std::array<training_binary::Buffer, MAX_PLAYER_COUNT> pendingDirectBuffers;
// The vendored serializer supports an output adapter. Reuse both its scratch
// storage and the growing byte buffer instead of dump()'s per-record objects.
struct RecordWriter {
    std::string bytes;
    nlohmann::detail::serializer<Json> serializer;
    RecordWriter() : serializer(nlohmann::detail::output_adapter<char>(bytes), ' ', Json::error_handler_t::replace) {
        bytes.reserve(64*1024);
    }
};
std::unique_ptr<RecordWriter> recordWriter;
Json pendingHeader;
int intervalMilliseconds = 100;
bool active = false, failed = false, mapWritten = false;
// Scheduled quiet capture (binary only): a preparation controller on stdin/stdout
// names the quiet cadence frames its event schedule can select. Other quiet rows
// keep their metadata and sighting-memory updates but omit the observation.
// Command rows and each player's first row always carry an observation.
bool scheduledQuietCapture = false;
struct QuietDirective {
    bool current = false;            // invalidated by every row written with an observation
    std::vector<unsigned> frames;    // quiet frames, from the sync frame on, that need one
};
std::array<QuietDirective, MAX_PLAYER_COUNT> quietDirectives;
bool fullPlacementMask = false;
// Keep the pre-command observation until dispatch supplies the actual result.
// Raw messages stay intact for independent replay-command reconciliation.
std::array<Json, MAX_PLAYER_COUNT> pendingSamples;
std::array<unsigned, MAX_PLAYER_COUNT> pendingPlayers;
size_t pendingCount = 0;
std::vector<std::pair<const GameMessage *, std::pair<size_t, size_t>>> pendingBuildCommands;
// Commands resolved at dispatch against the real selection (training_commands).
std::vector<std::pair<const GameMessage *, std::pair<size_t, size_t>>> pendingResolvedCommands;
unsigned samples = 0, commands = 0, waitSamples = 0;
unsigned crcMessages = 0;
struct Sighting : training_sightings::Flags {
    Json value;
    const ThingTemplate *type = nullptr;
};
using Sightings = std::vector<Sighting>;
std::array<Sightings, MAX_PLAYER_COUNT> sightings;
std::array<unsigned, MAX_PLAYER_COUNT> observedSightingRemovals{}, temporarySightingsForgotten{};
std::array<std::optional<unsigned>, MAX_PLAYER_COUNT> lastSample, lastCommand;
std::array<Json, MAX_PLAYER_COUNT> submissions, validations;
std::array<Json, MAX_PLAYER_COUNT> liveObservations;
std::array<std::vector<unsigned>, MAX_PLAYER_COUNT> replaySelection;
std::array<std::array<std::vector<unsigned>, 10>, MAX_PLAYER_COUNT> replayTeams;
constexpr double placementTau = 6.2831853071795864769;

// Opt-in host timings; never serialized into policy observations or CRC state.
enum ExportScope { EXPORT_CAPTURE, EXPORT_OBSERVATION, EXPORT_ENTITY, EXPORT_TASK,
    EXPORT_DOCK, EXPORT_RESUME, EXPORT_PLACEMENT, EXPORT_JSON_WRITE,
    EXPORT_PLACEMENT_OVERLAP, EXPORT_PLACEMENT_PATH, EXPORT_PLACEMENT_TERRAIN,
    EXPORT_COMMAND_SET, EXPORT_DOCK_SCAN, EXPORT_JSON_DUMP, EXPORT_STREAM_WRITE,
    EXPORT_TASK_DEFAULTS, EXPORT_ENTITY_BASE, EXPORT_CAPABILITIES, EXPORT_ENTITY_LINKS,
    EXPORT_PRODUCTION, EXPORT_MENUS, EXPORT_SIGHTING_FLAGS, EXPORT_WORLD_SCAN,
    EXPORT_SIGHTING_COPY, EXPORT_VISIBILITY_GRID, EXPORT_OBSERVATION_ASSEMBLY,
    EXPORT_VISIBILITY_CHECK, EXPORT_ACTIONS, EXPORT_SAMPLE_ASSEMBLY,
    EXPORT_RECORD_DESTROY, EXPORT_DIRECT_OBSERVATION, EXPORT_DIRECT_ENTITY, EXPORT_DIRECT_TASK, EXPORT_COMMANDS,
    EXPORT_COMMAND_BUTTONS, EXPORT_COMMAND_OBJECT_TARGETS, EXPORT_COMMAND_TARGETS, EXPORT_SCOPES };
bool profileExport = false;
struct ScopeCounter { double seconds = 0; unsigned long long calls = 0; };
ScopeCounter exportCounters[EXPORT_SCOPES];
std::chrono::steady_clock::time_point exportStarted;
class ExportTimer {
    ExportScope scope;
    std::chrono::steady_clock::time_point started;
public:
    explicit ExportTimer(ExportScope value) : scope(value) {
        if (profileExport) started = std::chrono::steady_clock::now();
    }
    void stop() {
        if (profileExport && scope != EXPORT_SCOPES) {
            exportCounters[scope].seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            ++exportCounters[scope].calls;
            scope = EXPORT_SCOPES;
        }
    }
    ~ExportTimer() { stop(); }
};

Json reservedObject(size_t fields)
{
    Json result = Json::object();
    // ordered_json uses vector<pair<const string, Json>>. Growing it after
    // inserting a nested value can copy that entire value on this toolchain.
    result.get_ref<Json::object_t &>().reserve(fields);
    return result;
}

Json templateCatalog();
Json observationPolicy();

size_t templateCount()
{
    size_t count = 0;
    for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate()) ++count;
    return count;
}

Json reservedArray(size_t elements)
{
    Json result = Json::array();
    result.get_ref<Json::array_t &>().reserve(elements);
    return result;
}

template<class String> void setString(Json &value, const String &text)
{
    if (value.is_string()) {
        auto &stored = value.get_ref<std::string &>();
        if (stored != text) stored = text;
    } else value = text;
}

Json &arraySlot(Json &array, size_t index)
{
    if (!array.is_array()) array = Json::array();
    auto &items = array.get_ref<Json::array_t &>();
    if (index >= items.size()) items.resize(index+1);
    return items[index];
}

void reuseJson(Json &destination, const Json &source)
{
    if (&destination == &source) return;
    if (source.is_string()) {
        setString(destination, source.get_ref<const std::string &>());
    } else if (source.is_array()) {
        if (!destination.is_array()) destination = reservedArray(source.size());
        auto &items = destination.get_ref<Json::array_t &>();
        items.resize(source.size());
        for (size_t i = 0; i < items.size(); ++i) reuseJson(items[i], source[i]);
    } else if (source.is_object() && destination.is_object() && source.size() == destination.size()) {
        const auto &values = source.get_ref<const Json::object_t &>();
        auto &items = destination.get_ref<Json::object_t &>();
        auto item = items.begin();
        for (const auto &value : values) if ((item++)->first != value.first) {
            destination = source;
            return;
        }
        item = items.begin();
        for (const auto &value : values) reuseJson((item++)->second, value.second);
    } else destination = source;
}

Json taskStatus(const char *status, Json result = Json())
{
    if (!result.is_object()) result = reservedObject(2);
    result.erase("id");
    setString(result["status"], status);
    return result;
}

Json placementGrid(int player, unsigned builderID, const std::string &name, int angleIndex, int bins)
{
    ExportTimer timer(EXPORT_PLACEMENT);
    if (bins < 16 || bins > 512 || (bins & (bins-1)) || angleIndex < 0 || angleIndex >= 16)
        throw std::runtime_error("Unsupported placement grid/angle");
    Object *builder = TheGameLogic->findObjectByID(static_cast<ObjectID>(builderID));
    Player *owner = ThePlayerList->getNthPlayer(player);
    const ThingTemplate *what = TheThingFactory->findTemplate(name.c_str(), FALSE);
    if (!builder || builder->getControllingPlayer() != owner || builder->isEffectivelyDead() || !what)
        throw std::runtime_error("Unknown owned builder/template");
    const CommandSet *set = TheControlBar->findCommandSet(builder->getCommandSetString());
    bool offered = false;
    if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
        const CommandButton *button = set->getCommandButton(i);
        if (button && button->getCommandType() == GUI_COMMAND_DOZER_CONSTRUCT && button->getThingTemplate() == what)
            offered = true;
    }
    if (!offered) throw std::runtime_error("Builder does not offer this building");
    const auto start = std::chrono::steady_clock::now();
    Region3D bounds; TheTerrainLogic->getExtent(&bounds);
    const float angle = placementTau * angleIndex/16;
    const unsigned options = BuildAssistant::TERRAIN_RESTRICTIONS | BuildAssistant::NO_OBJECT_OVERLAP | BuildAssistant::CLEAR_PATH
        | BuildAssistant::SHROUD_REVEALED | BuildAssistant::OBSERVED_PLACEMENT_PREVIEW
        | BuildAssistant::NO_PLACEMENT_FEEDBACK;
    static std::vector<unsigned char> packed;
    packed.assign((bins*bins+7)/8, 0);
    unsigned legal = 0;
    for (int y = 0; y < bins; ++y) for (int x = 0; x < bins; ++x) {
        Coord3D p;
        p.x = static_cast<float>(bounds.lo.x+(x+.5)*(static_cast<double>(bounds.hi.x)-bounds.lo.x)/bins);
        p.y = static_cast<float>(bounds.lo.y+(y+.5)*(static_cast<double>(bounds.hi.y)-bounds.lo.y)/bins);
        p.z = TheTerrainLogic->getGroundHeight(p.x, p.y);
        if (TheBuildAssistant->isLocationLegalToBuild(&p, what, angle, options, builder, nullptr) == LBC_OK) {
            const int index = y*bins+x; packed[index/8] |= 1 << (index%8); ++legal;
        }
    }
    const char hex[] = "0123456789abcdef";
    std::string mask; mask.reserve(packed.size()*2);
    for (unsigned char value : packed) { mask += hex[value >> 4]; mask += hex[value & 15]; }
    Json result = reservedObject(12);
    result["schema"] = 1;
    result["builder"] = builderID;
    result["template"] = name;
    result["angle_index"] = angleIndex;
    result["angle_bins"] = 16;
    result["bins"] = bins;
    result["mask_hex"] = std::move(mask);
    result["legal_sites"] = legal;
    result["frame"] = TheGameLogic->getFrame();
    result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    result["path_check"] = "per-builder terrain-only UI reachability; hidden structures ignored";
    return result;
}

Json replayPlacements(int player, const Json &actions)
{
    Json result = Json::array();
    auto &selected = replaySelection[player];
    auto &teams = replayTeams[player];
    auto owned = [player](unsigned id) {
        Object *object = TheGameLogic->findObjectByID(static_cast<ObjectID>(id));
        return object && !object->isEffectivelyDead() && object->getControllingPlayer() == ThePlayerList->getNthPlayer(player);
    };
    for (const Json &action : actions) {
        const std::string &name = action["name"].get_ref<const std::string &>();
        const Json &args = action["arguments"];
        if (name == "MSG_DESTROY_SELECTED_GROUP") { selected.clear(); continue; }
        if (name == "MSG_CREATE_SELECTED_GROUP" || name == "MSG_CREATE_SELECTED_GROUP_NO_SOUND") {
            if (args[0]["value"].get<bool>()) selected.clear();
            for (size_t i = 1; i < args.size(); ++i) {
                const unsigned id = args[i]["value"];
                if (std::find(selected.begin(), selected.end(), id) == selected.end()) selected.push_back(id);
            }
        } else if (name == "MSG_REMOVE_FROM_SELECTED_GROUP") {
            for (const Json &arg : args) {
                const unsigned id = arg["value"];
                selected.erase(std::remove(selected.begin(), selected.end(), id), selected.end());
            }
        } else if (!name.empty() && name.back() >= '0' && name.back() <= '9'
            && (name.find("MSG_CREATE_TEAM") == 0 || name.find("MSG_SELECT_TEAM") == 0 || name.find("MSG_ADD_TEAM") == 0)) {
            const int index = name.back()-'0';
            if (name.find("MSG_CREATE_TEAM") == 0) {
                teams[index].clear();
                for (const Json &arg : args) if (owned(arg["value"].get<unsigned>())) {
                    const unsigned id = arg["value"];
                    for (auto &team : teams) team.erase(std::remove(team.begin(), team.end(), id), team.end());
                    teams[index].push_back(id);
                }
            } else if (name.find("MSG_SELECT_TEAM") == 0) selected = teams[index];
            else for (unsigned id : teams[index]) if (std::find(selected.begin(), selected.end(), id) == selected.end()) selected.push_back(id);
        } else if (name == "MSG_DOZER_CONSTRUCT") {
            unsigned builderID = 0, builderCount = 0;
            for (unsigned id : selected) if (owned(id)) { builderID = id; ++builderCount; }
            if (builderCount != 1) continue; // The Python adapter also excludes this batch.
            const ThingTemplate *what = TheThingFactory->findByTemplateID(args[0]["value"].get<unsigned short>());
            if (!what) continue;
            const float rawAngle = args[2]["value"];
            const int angle = (static_cast<int>(std::nearbyint(rawAngle/placementTau*16))%16+16)%16;
            Json grid = placementGrid(player, builderID, what->getName().str(), angle, 256);
            Object *builder = TheGameLogic->findObjectByID(static_cast<ObjectID>(builderID));
            Coord3D original; original.x = args[1]["value"][0]; original.y = args[1]["value"][1];
            original.z = TheTerrainLogic->getGroundHeight(original.x, original.y);
            Region3D bounds; TheTerrainLogic->getExtent(&bounds);
            const unsigned options = BuildAssistant::TERRAIN_RESTRICTIONS | BuildAssistant::NO_OBJECT_OVERLAP
                | BuildAssistant::SHROUD_REVEALED | BuildAssistant::CLEAR_PATH | BuildAssistant::NO_PLACEMENT_FEEDBACK;
            Json audit = {{"original", static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(
                &original, what, rawAngle, options, builder, nullptr))}};
            for (int resolution : {256, 512}) {
                Coord3D quantized = original;
                const int x = std::max(0, std::min(resolution-1, static_cast<int>((original.x-bounds.lo.x)/(bounds.hi.x-bounds.lo.x)*resolution)));
                const int y = std::max(0, std::min(resolution-1, static_cast<int>((original.y-bounds.lo.y)/(bounds.hi.y-bounds.lo.y)*resolution)));
                quantized.x = static_cast<float>(bounds.lo.x+(x+.5)*(static_cast<double>(bounds.hi.x)-bounds.lo.x)/resolution);
                quantized.y = static_cast<float>(bounds.lo.y+(y+.5)*(static_cast<double>(bounds.hi.y)-bounds.lo.y)/resolution);
                quantized.z = TheTerrainLogic->getGroundHeight(quantized.x, quantized.y);
                audit[std::to_string(resolution)] = {
                    {"position_only", static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(&quantized, what, rawAngle, options, builder, nullptr))},
                    {"position_and_angle", static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(&quantized, what, angle*placementTau/16, options, builder, nullptr))}};
            }
            audit["angle_only"] = static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(&original, what, angle*placementTau/16, options, builder, nullptr));
            grid["round_trip_audit"] = audit; // Diagnostics only; never a neural input.
            result.push_back(grid);
        }
    }
    return result;
}

void write(const Json &record, const training_binary::Buffer *prepared = nullptr)
{
    ExportTimer timer(EXPORT_JSON_WRITE);
    if (binaryFormat) {
        binaryWriter->write(record, prepared);
        return;
    }
    if (!recordWriter) recordWriter = std::make_unique<RecordWriter>();
    std::string &serialized = recordWriter->bytes;
    serialized.clear();
    {
        ExportTimer timer(EXPORT_JSON_DUMP);
        recordWriter->serializer.dump(record, false, false, 0);
    }
    {
        ExportTimer timer(EXPORT_STREAM_WRITE);
        output << serialized << '\n';
    }
    if (!output) throw std::runtime_error("Cannot write training export");
}

void flushSamples()
{
    for (size_t i = 0; i < pendingCount; ++i) {
        const auto player = pendingPlayers[i];
        write(pendingSamples[player], binaryFormat ? &pendingDirectBuffers[player] : nullptr);
    }
    // Retain one record per player: command-only samples must not swap viewers
    // through the same scratch tree between regular cadence boundaries.
    pendingCount = 0;
    pendingBuildCommands.clear();
    pendingResolvedCommands.clear();
}

// Ask the controller which quiet frames need an observation, once per row that
// carried one: the event schedule changes only on such rows. Every earlier row
// of this player is already written (flushSamples runs first each tick).
bool quietObservationWanted(int player, unsigned frame)
{
    QuietDirective &directive = quietDirectives[player];
    if (!directive.current) {
        output.flush();
        if (!output) throw std::runtime_error("Cannot flush training export for the capture controller");
        std::cout << "GAIX " << Json({{"event", "capture"}, {"player", player}, {"frame", frame}}).dump() << std::endl;
        std::string line;
        if (!std::getline(std::cin, line)) throw std::runtime_error("Capture controller closed");
        const Json reply = Json::parse(line);
        if (!reply.is_object() || reply.at("player").get<int>() != player || reply.at("frame").get<unsigned>() != frame)
            throw std::runtime_error("Capture directive does not answer this request");
        directive.frames.clear();
        for (const Json &value : reply.at("frames")) {
            const unsigned wanted = value.get<unsigned>();
            if (wanted < frame) throw std::runtime_error("Capture directive names a past frame");
            directive.frames.push_back(wanted);
        }
        directive.current = true;
    }
    return std::find(directive.frames.begin(), directive.frames.end(), frame) != directive.frames.end();
}

Json position(const Coord3D *p, Json result = Json())
{
    if (!result.is_array()) result = reservedArray(3);
    auto &values = result.get_ref<Json::array_t &>();
    values.resize(3);
    values[0] = p->x; values[1] = p->y; values[2] = p->z;
    return result;
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
    // are unattackable objects. Keep unattackable structures/bridges: these
    // can still be persistent, meaningful map obstacles.
    return !t->isKindOf(KINDOF_UNATTACKABLE) || t->isKindOf(KINDOF_STRUCTURE)
        || t->isKindOf(KINDOF_BRIDGE);
}

bool retainedNeutralTemplate(const ThingTemplate *t)
{
    // Objects of non-playable sides are kept only for gameplay roles: economy,
    // capture, garrison/transport, repair, crates, hazards, crossings and armed
    // faction actors. Barrels, fences, walls, piers, ruins, animals, civilians and
    // civilian traffic are scenery clutter, like foliage. Static template data only.
    static const KindOfType roles[] = {KINDOF_SUPPLY_SOURCE, KINDOF_SUPPLY_SOURCE_ON_PREVIEW,
        KINDOF_TECH_BUILDING, KINDOF_CAPTURABLE, KINDOF_CASH_GENERATOR, KINDOF_REPAIR_PAD, KINDOF_HEAL_PAD,
        KINDOF_CRATE, KINDOF_MINE, KINDOF_BOOBY_TRAP, KINDOF_DEMOTRAP, KINDOF_BRIDGE, KINDOF_LANDMARK_BRIDGE,
        KINDOF_BRIDGE_TOWER, KINDOF_GARRISONABLE_UNTIL_DESTROYED};
    for (KindOfType role : roles)
        if (t->isKindOf(role)) return true;
    // Civilian cars, boats, aircraft and trains carry CAN_ATTACK (and cars a weapon
    // set) for ambient behaviour; only faction-side armed objects are actors.
    if (t->isKindOf(KINDOF_CAN_ATTACK) && t->getDefaultOwningSide() != "Civilian") return true;
    const ModuleInfo &modules = t->getBehaviorModuleInfo();
    for (int i = 0; i < modules.getCount(); ++i)
        if (std::strstr(modules.getNthName(i).str(), "Contain")) return true;
    return false;
}

bool retainedEntity(const Object *o)
{
    // Ownership is observable; playable sides keep every retained template.
    const Player *owner = o->getControllingPlayer();
    return retainedEntityTemplate(o->getTemplate())
        && ((owner && owner->isPlayableSide()) || retainedNeutralTemplate(o->getTemplate()));
}

bool persistentSightingTemplate(const ThingTemplate *t)
{
    const ModuleInfo &modules = t->getBehaviorModuleInfo();
    for (int i = 0; i < modules.getCount(); ++i) {
        const AsciiString &name = modules.getNthName(i);
        // Delivery/strike aircraft and the circling gunship are useful while
        // visible, but their old positions do not belong in persistent memory.
        if (name == "DeliverPayloadAIUpdate" || name == "SpectreGunshipUpdate") return false;
        if (name == "LifetimeUpdate") {
            const auto *lifetime = static_cast<const LifetimeUpdateModuleData *>(modules.getNthData(i));
            if (lifetime->m_minFrames || lifetime->m_maxFrames) return false;
        }
    }
    return true;
}

Json taskReference(ObjectID id, Player *owner, Json result = Json())
{
    if (id == INVALID_ID) return taskStatus("none", std::move(result));
    Object *target = TheGameLogic->findObjectByID(id);
    // Missing, dead and hidden objects share one response. Never refresh a
    // hidden target's attributes through the owner's internal AI pointer.
    if (target && !target->isEffectivelyDead() && retainedEntity(target)
        && (target->getControllingPlayer() == owner || TrainingObjectVisibleTo(target, owner->getPlayerIndex())))
    {
        if (!result.is_object()) result = reservedObject(2);
        setString(result["status"], "current");
        result["id"] = static_cast<unsigned>(id);
        return result;
    }
    return taskStatus("unavailable", std::move(result));
}

Json unitTask(Object *object, Json j = Json())
{
    ExportTimer timer(EXPORT_TASK);
    AIUpdateInterface *ai = object->getAIUpdateInterface();
    Player *owner = object->getControllingPlayer();
    ExportTimer defaults(EXPORT_TASK_DEFAULTS);
    if (!j.is_object()) j = reservedObject(12);
    setString(j["kind"], "unknown");
    j["available"] = true;
    j["moving"] = false;
    j["attacking"] = false;
    j["waiting_for_path"] = false;
    j["busy"] = false;
    j["disabled"] = static_cast<bool>(object->isDisabled());
    Json previousDestination = std::move(j["destination"]);
    j["destination"] = nullptr;
    j["target"] = taskStatus("none", std::move(j["target"]));
    j["combat_target"] = taskStatus("none", std::move(j["combat_target"]));
    Json cargo = std::move(j["cargo"]);
    if (!cargo.is_object()) cargo = reservedObject(4);
    cargo.erase("boxes"); cargo.erase("capacity"); cargo.erase("ferrying");
    cargo["available"] = false;
    j["cargo"] = std::move(cargo);
    j["preferred_dock"] = taskStatus("none", std::move(j["preferred_dock"]));
    defaults.stop();
    if (ai) {
        const AIStateType state = ai->getAIStateType();
        bool point = false, path = false, targetTask = false;
        switch (state) {
        case AI_IDLE: setString(j["kind"], "idle"); break;
        case AI_MOVE_TO: case AI_MOVE_AND_TIGHTEN: setString(j["kind"], "move"); point = true; targetTask = true; break;
        case AI_FOLLOW_PATH: setString(j["kind"], "move"); path = true; break;
        case AI_FOLLOW_EXITPRODUCTION_PATH:
        case AI_FOLLOW_WAYPOINT_PATH_AS_TEAM: case AI_FOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS:
        case AI_FOLLOW_WAYPOINT_PATH_AS_TEAM_EXACT: case AI_FOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS_EXACT:
            setString(j["kind"], "move"); break;
        case AI_ATTACK_MOVE_TO: setString(j["kind"], "attack_move"); point = true; break;
        case AI_ATTACKFOLLOW_WAYPOINT_PATH_AS_INDIVIDUALS: case AI_ATTACKFOLLOW_WAYPOINT_PATH_AS_TEAM:
            setString(j["kind"], "attack_move"); break;
        case AI_ATTACK_OBJECT: case AI_FORCE_ATTACK_OBJECT: case AI_ATTACK_AND_FOLLOW_OBJECT:
            setString(j["kind"], "attack"); targetTask = true; break;
        case AI_ATTACK_POSITION: setString(j["kind"], "attack"); point = true; break;
        case AI_DOCK: setString(j["kind"], "dock"); targetTask = true; break;
        case AI_GUARD: case AI_GUARD_TUNNEL_NETWORK: case AI_GUARD_RETALIATE: setString(j["kind"], "guard"); break;
        default: setString(j["kind"], "other"); break;
        }
        ObjectID goal = targetTask ? ai->getGoalObjectID() : INVALID_ID;
        DozerAIInterface *dozer = ai->getDozerAIInterface();
        if (dozer && dozer->getCurrentTask() != DOZER_TASK_INVALID) {
            const DozerTask task = dozer->getCurrentTask();
            setString(j["kind"], task == DOZER_TASK_BUILD ? "construct" : task == DOZER_TASK_REPAIR ? "repair" : "other");
            goal = dozer->getTaskTarget(task);
            point = path = false;
        }
        j["target"] = taskReference(goal, owner, std::move(j["target"]));
        if (goal != INVALID_ID) {
            if (j["target"]["status"] == "current")
                j["destination"] = position(TheGameLogic->findObjectByID(goal)->getPosition(), std::move(previousDestination));
        } else if (point) {
            j["destination"] = position(ai->getGoalPosition(), std::move(previousDestination));
        } else if (path && ai->getGoalPathEnd()) {
            j["destination"] = position(ai->getGoalPathEnd(), std::move(previousDestination));
        }
        j["moving"] = static_cast<bool>(ai->isMoving());
        j["attacking"] = static_cast<bool>(ai->isAttacking() || object->testStatus(OBJECT_STATUS_IS_ATTACKING));
        j["waiting_for_path"] = static_cast<bool>(ai->isWaitingForPath());
        j["busy"] = static_cast<bool>(ai->isBusy());
        j["combat_target"] = taskReference(ai->getCurrentVictimID(), owner, std::move(j["combat_target"]));
        SupplyTruckAIInterface *supply = ai->getSupplyTruckAIInterface();
        if (supply) {
            const bool ferrying = supply->isCurrentlyFerryingSupplies();
            const int boxes = supply->getNumberBoxes();
            Json cargo = std::move(j["cargo"]);
            cargo["available"] = true;
            cargo["boxes"] = boxes;
            cargo["capacity"] = supply->getMaxBoxes();
            cargo["ferrying"] = ferrying;
            j["cargo"] = std::move(cargo);
            j["preferred_dock"] = taskReference(supply->getPreferredDockID(), owner, std::move(j["preferred_dock"]));
            // The supply state machine delivers whenever it carries any boxes.
            if (ferrying) setString(j["kind"], boxes > 0 ? "return_supplies" : "collect");
        }
    } else setString(j["kind"], "idle");
    ProductionUpdateInterface *production = object->getProductionUpdateInterface();
    if (object->getConstructionPercent() != -1) setString(j["kind"], "construct");
    else if (production && production->firstProduction() && j["kind"] == "idle") setString(j["kind"], "produce");
    return j;
}

Json commandContext(const std::array<Json, MAX_PLAYER_COUNT> &records, Player *owner, unsigned frame, Json j = Json())
{
    const Json &record = records.at(owner->getPlayerIndex());
    if (record.is_null()) {
        if (!j.is_object() || j.size() != 1 || !j.contains("available")) j = reservedObject(1);
        j["available"] = false;
        return j;
    }
    if (j.is_object()) j.erase("age_seconds");
    reuseJson(j, record);
    j["age_seconds"] = static_cast<double>(frame-j["frame"].get<unsigned>()) / static_cast<int>(LOGICFRAMES_PER_SECOND);
    if (j.contains("target") && j["target"].contains("id")) {
        const ObjectID id = static_cast<ObjectID>(j["target"]["id"].get<unsigned>());
        j["target"] = taskReference(id, owner, std::move(j["target"]));
    }
    return j;
}

struct ObservationLookups {
    std::vector<std::pair<const AsciiString *, const CommandSet *>> commandSets;
    // Docks that pass TrainingCanDock's target test (alive, owned or visible).
    std::vector<Object *> docks;
    // Owned and currently visible objects: the only legal command targets.
    training_commands::Candidates commandTargets;
    training_commands::ObjectCommands commands;
    training_commands::PlayerCommands player;
};
// Reuse storage, but never retain lookup results across snapshots/game ticks.
ObservationLookups observationLookups;

const CommandSet *commandSetFor(Object *object, ObservationLookups &lookups)
{
    const AsciiString &name = object->getCommandSetString();
    for (const auto &entry : lookups.commandSets)
        if (*entry.first == name) return entry.second;
    ExportTimer timer(EXPORT_COMMAND_SET);
    const CommandSet *set = TheControlBar->findCommandSet(name);
    lookups.commandSets.emplace_back(&name, set);
    return set;
}

Json entity(Object *object, bool own, unsigned frame, ObservationLookups &lookups, Json j = Json())
{
    ExportTimer timer(EXPORT_ENTITY);
    Player *owner = object->getControllingPlayer();
    BodyModuleInterface *body = object->getBodyModule();
    ExportTimer base(EXPORT_ENTITY_BASE);
    if (!j.is_object() || j.value("own", !own) != own) j = reservedObject(own ? 32 : 9);
    j["id"] = static_cast<unsigned>(object->getID());
    setString(j["type"], object->getTemplate()->getName().str());
    j["owner"] = owner ? static_cast<int>(owner->getPlayerIndex()) : -1;
    j["position"] = position(object->getPosition(), std::move(j["position"]));
    j["health"] = body ? body->getHealth() : 0.0f;
    j["max_health"] = body ? body->getMaxHealth() : 0.0f;
    j["last_seen_frame"] = frame;
    j["visible"] = true;
    j["own"] = own;
    base.stop();
    if (own) {
        // No game tick or INI override occurs during this read-only snapshot.
        // Resolve each current command-set name once, then discard the cache.
        const CommandSet *set = commandSetFor(object, lookups);
        j["task"] = unitTask(object, std::move(j["task"]));
        ExportTimer capabilities(EXPORT_CAPABILITIES);
        j["radius"] = object->getGeometryInfo().getBoundingCircleRadius();
        // Current capabilities belong only to the owner. In particular,
        // garrisons and disabled/deploying weapons need more than a type name.
        j["has_weapon"] = static_cast<bool>(object->hasAnyWeapon());
        j["can_attack"] = static_cast<bool>(TrainingCanAttack(object));
        j["has_atk_move_cmd"] = hasAttackMoveCommand(set);
        j["can_attack_move"] = j["can_attack"].get<bool>() && j["has_atk_move_cmd"].get<bool>()
            && !object->isKindOf(KINDOF_IMMOBILE) && object->getAIUpdateInterface();
        capabilities.stop();
        // Exact owner command legality, limited to owned/currently visible
        // destinations. Never probe remembered hidden entities for this mask.
        if (!j["dock_targets"].is_array()) j["dock_targets"] = Json::array();
        j["dock_targets"].clear();
        if (object->getAIUpdateInterface()) {
            ExportTimer timer(EXPORT_DOCK);
            for (Object *target : lookups.docks)
                if (TrainingCanDock(object, target))
                    j["dock_targets"].push_back(static_cast<unsigned>(target->getID()));
        }
        if (!j["resume_targets"].is_array()) j["resume_targets"] = Json::array();
        j["resume_targets"].clear();
        if (object->getAIUpdateInterface() && object->getAIUpdateInterface()->getDozerAIInterface()) {
            ExportTimer timer(EXPORT_RESUME);
            for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject())
                if (TrainingCanResumeConstruction(object, target))
                    j["resume_targets"].push_back(static_cast<unsigned>(target->getID()));
        }
        ExportTimer links(EXPORT_ENTITY_LINKS);
        Object *producer = TheGameLogic->findObjectByID(object->getProducerID());
        Object *builder = TheGameLogic->findObjectByID(object->getBuilderID());
        if (producer && producer->getControllingPlayer() == owner)
            j["producer_id"] = static_cast<unsigned>(producer->getID());
        else j.erase("producer_id");
        if (builder && builder->getControllingPlayer() == owner)
            j["builder_id"] = static_cast<unsigned>(builder->getID());
        else j.erase("builder_id");
        j["construction_percent"] = object->getConstructionPercent();
        setString(j["ai_state"], object->getAIUpdateInterface()
            ? object->getAIUpdateInterface()->getCurrentStateName().str() : "");
        links.stop();
        ExportTimer queue(EXPORT_PRODUCTION);
        if (!j["production_queue"].is_array()) j["production_queue"] = Json::array();
        size_t produced = 0;
        ProductionUpdateInterface *production = object->getProductionUpdateInterface();
        if (production) {
            for (const ProductionEntry *p = production->firstProduction(); p; p = production->nextProduction(p)) {
                Json &entry = arraySlot(j["production_queue"], produced++);
                if (!entry.is_object()) entry = reservedObject(6);
                entry["production_id"] = static_cast<unsigned>(p->getProductionID());
                entry["kind"] = static_cast<int>(p->getProductionType());
                entry["progress"] = p->getPercentComplete();
                entry["remaining_quantity"] = p->getProductionQuantityRemaining();
                // Unit and upgrade pointers share a union: only the type discriminates.
                const bool unit = p->getProductionType() == PRODUCTION_UNIT && p->getProductionObject();
                const bool upgrade = p->getProductionType() == PRODUCTION_UPGRADE && p->getProductionUpgrade();
                if (unit) setString(entry["unit_type"], p->getProductionObject()->getName().str());
                else entry.erase("unit_type");
                if (upgrade) setString(entry["upgrade"], p->getProductionUpgrade()->getUpgradeName().str());
                else entry.erase("upgrade");
            }
        }
        j["production_queue"].get_ref<Json::array_t &>().resize(produced);
        queue.stop();
        // These are the owner's current UI capabilities, never enemy state.
        ExportTimer menus(EXPORT_MENUS);
        if (!j["buildable_templates"].is_array()) j["buildable_templates"] = Json::array();
        if (!j["trainable_templates"].is_array()) j["trainable_templates"] = Json::array();
        size_t buildable = 0, trainable = 0;
        if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
            const CommandButton *button = set->getCommandButton(i);
            if (!button || !button->getThingTemplate()) continue;
            const ThingTemplate *what = button->getThingTemplate();
            if (button->getCommandType() == GUI_COMMAND_DOZER_CONSTRUCT
                && TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK)
                setString(arraySlot(j["buildable_templates"], buildable++), what->getName().str());
            if (button->getCommandType() == GUI_COMMAND_UNIT_BUILD && production
                && TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK
                && production->canQueueCreateUnit(what) == CANMAKE_OK)
                setString(arraySlot(j["trainable_templates"], trainable++), what->getName().str());
        }
        j["buildable_templates"].get_ref<Json::array_t &>().resize(buildable);
        j["trainable_templates"].get_ref<Json::array_t &>().resize(trainable);
        menus.stop();
        ExportTimer commandTimer(EXPORT_COMMANDS);
        auto &state = lookups.commands;
        training_commands::objectCommands(object, set, lookups.commandTargets, state);
        j["commands"] = Json::array();
        for (const char *name : state.available) j["commands"].push_back(name);
        j["command_cooldowns"] = Json::array();
        for (const auto &item : state.charging)
            j["command_cooldowns"].push_back({{"command", item.first}, {"ready_percent", item.second}});
        j["command_targets"] = Json::array();
        for (const auto &item : state.targets) j["command_targets"].push_back({{"command", item.first}, {"targets", item.second}});
        j["enter_targets"] = state.enterTargets;
        j["container_id"] = state.containerID;
        j["occupants"] = state.occupants;
        j["capacity"] = state.capacity;
        j["upgrades"] = Json::array();
        for (const char *name : state.upgrades) j["upgrades"].push_back(name);
        j["can_guard"] = state.canGuard;
        j["repair_targets"] = state.repairTargets;
        j["repair_depots"] = state.repairDepots;
    }
    return j;
}

Json argument(const GameMessage *message, int index)
{
    const GameMessageArgumentType *a = message->getArgument(index);
    Json j = reservedObject(2);
    j["type"] = static_cast<int>(message->getArgumentDataType(index));
    switch (message->getArgumentDataType(index)) {
        case ARGUMENTDATATYPE_INTEGER: j["value"] = a->integer; break;
        case ARGUMENTDATATYPE_REAL: j["value"] = a->real; break;
        case ARGUMENTDATATYPE_BOOLEAN: j["value"] = static_cast<bool>(a->boolean); break;
        case ARGUMENTDATATYPE_OBJECTID: j["value"] = static_cast<unsigned>(a->objectID); break;
        case ARGUMENTDATATYPE_DRAWABLEID: j["value"] = static_cast<unsigned>(a->drawableID); break;
        case ARGUMENTDATATYPE_TEAMID: j["value"] = a->teamID; break;
        case ARGUMENTDATATYPE_LOCATION: j["value"] = position(&a->location); break;
        case ARGUMENTDATATYPE_PIXEL: j["value"] = Json::array({a->pixel.x, a->pixel.y}); break;
        case ARGUMENTDATATYPE_PIXELREGION:
            j["value"] = Json::array({a->pixelRegion.lo.x, a->pixelRegion.lo.y,
                                     a->pixelRegion.hi.x, a->pixelRegion.hi.y}); break;
        case ARGUMENTDATATYPE_TIMESTAMP: j["value"] = a->timestamp; break;
        case ARGUMENTDATATYPE_WIDECHAR: j["value"] = static_cast<unsigned>(a->wChar); break;
        default: throw std::runtime_error("Unknown command argument type");
    }
    return j;
}

Json actionsFor(int player)
{
    ExportTimer timer(EXPORT_ACTIONS);
    Json actions; // No allocation for the many unsampled, command-free ticks.
    for (GameMessage *m = TheCommandList->getFirstMessage(); m; m = m->next()) {
        const GameMessage::Type type = m->getType();
        if (m->getPlayerIndex() != player || type <= GameMessage::MSG_BEGIN_NETWORK_MESSAGES
            || type >= GameMessage::MSG_END_NETWORK_MESSAGES || type == GameMessage::MSG_LOGIC_CRC
            || type == GameMessage::MSG_SET_REPLAY_CAMERA) continue;
        Json action = reservedObject(4);
        action["type_id"] = static_cast<int>(type);
        action["name"] = m->getCommandAsString();
        action["arguments"] = reservedArray(m->getArgumentCount());
        for (int i = 0; i < m->getArgumentCount(); ++i) action["arguments"].push_back(argument(m, i));
        if (type == GameMessage::MSG_DOZER_CONSTRUCT) {
            action["build_result"] = {{"accepted", false}, {"status", "not_dispatched"}};
            pendingBuildCommands.emplace_back(m, std::make_pair(static_cast<size_t>(player), actions.size()));
        }
        if (training_commands::covers(m)) {
            action["command_result"] = {{"resolved", false}, {"reason", "not_dispatched"}};
            pendingResolvedCommands.emplace_back(m, std::make_pair(static_cast<size_t>(player), actions.size()));
        }
        actions.push_back(std::move(action));
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
    Json heights = reservedArray(gridSize*gridSize), cliffs = reservedArray(gridSize*gridSize);
    for (int y = 0; y < gridSize; ++y) for (int x = 0; x < gridSize; ++x) {
        const Coord3D p = gridPoint(region, x, y);
        heights.push_back(TheTerrainLogic->getGroundHeight(p.x, p.y));
        cliffs.push_back(static_cast<bool>(TheTerrainLogic->isCliffCell(p.x, p.y)));
    }
    Json sources = Json::array();
    // Resource locations are static, public map information. Do not export
    // remaining supplies, player-owned sources, or any opponent objects here.
    for (Object *o = TheGameLogic->getFirstObject(); o; o = o->getNextObject())
        if (o->isKindOf(KINDOF_SUPPLY_SOURCE) && (!o->getControllingPlayer()
            || o->getControllingPlayer()->getPlayerIndex() < 2))
            sources.push_back({{"id", static_cast<unsigned>(o->getID())}, {"type", o->getTemplate()->getName().str()},
                               {"position", position(o->getPosition())}});
    Json result = reservedObject(7);
    result["record"] = "map";
    result["grid_size"] = gridSize;
    result["supply_sources"] = std::move(sources);
    result["extent_low"] = position(&region.lo);
    result["extent_high"] = position(&region.hi);
    result["height"] = std::move(heights);
    result["cliff"] = std::move(cliffs);
    return result;
}

void mapRecord()
{
    // Replay setup loads map.ini after Begin. Resolve its final template
    // overrides before declaring the exact classes that the encoder excludes.
    pendingHeader["observation_policy"] = observationPolicy();
    write(pendingHeader);
    pendingHeader = Json();
    write(mapFeatures());
    mapWritten = true;
}

Json observation(Player *player, unsigned frame, Json result = Json())
{
    ExportTimer timer(EXPORT_OBSERVATION);
    const int index = static_cast<int>(player->getPlayerIndex());
    if (!result.is_object()) result = reservedObject(10);
    Json own = std::move(result["own_entities"]), known = std::move(result["known_nonowned_entities"]);
    if (!own.is_array()) own = Json::array();
    if (!known.is_array()) known = Json::array();
    size_t owned = 0;
    ObservationLookups &lookups = observationLookups;
    lookups.commandSets.clear();
    lookups.docks.clear();
    {
        ExportTimer timer(EXPORT_DOCK_SCAN);
        // canDockAt first requires a DockUpdateInterface, and TrainingCanDock
        // a live, owned or visible target. Share those necessary tests across
        // actors, preserving world-list order and the final engine legality check.
        lookups.commandTargets.clear();
        for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject()) {
            if (!retainedEntity(target)) continue;
            bool dock = false;
            for (BehaviorModule **module = target->getBehaviorModules(); *module; ++module)
                if ((*module)->getDockUpdateInterface()) {
                    dock = true;
                    break;
                }
            const bool seen = !target->isEffectivelyDead()
                && (target->getControllingPlayer() == player || TrainingObjectVisibleTo(target, index));
            if (dock && seen) lookups.docks.push_back(target);
            if (seen && !target->isDestroyed()) lookups.commandTargets.add(target, dock);
        }
    }
    auto &memory = sightings[index];
    ExportTimer flags(EXPORT_SIGHTING_FLAGS);
    for (auto &entry : memory) { entry.visible = false; entry.value["visible"] = false; }
    flags.stop();
    ExportTimer world(EXPORT_WORLD_SCAN);
    for (Object *o = TheGameLogic->getFirstObject(); o; o = o->getNextObject()) {
        if (!retainedEntity(o)) continue;
        if (o->isEffectivelyDead() || o->isDestroyed()) continue;
        const bool isOwned = o->getControllingPlayer() == player;
        if (isOwned) {
            Json &slot = arraySlot(own, owned++);
            slot = entity(o, true, frame, lookups, std::move(slot));
            const unsigned id = static_cast<unsigned>(o->getID());
            const auto found = training_sightings::find(memory, id);
            if (found != memory.end() && found->id == id) memory.erase(found);
        } else if (TrainingObjectVisibleTo(o, index)) {
            const unsigned id = static_cast<unsigned>(o->getID());
            auto found = training_sightings::find(memory, id);
            if (found == memory.end() || found->id != id) found = memory.emplace(found);
            found->id = id;
            found->visible = true;
            if (found->type != o->getTemplate()) {
                found->type = o->getTemplate();
                found->persistent = persistentSightingTemplate(found->type);
            }
            found->value = entity(o, false, frame, lookups, std::move(found->value));
        }
    }
    world.stop();
    // Pruning depends on observed visibility and static template metadata only.
    temporarySightingsForgotten[index] += static_cast<unsigned>(training_sightings::forgetHiddenTemporary(memory));
    ExportTimer copy(EXPORT_SIGHTING_COPY);
    own.get_ref<Json::array_t &>().resize(owned);
    size_t remembered = 0;
    for (const auto &entry : memory) reuseJson(arraySlot(known, remembered++), entry.value);
    known.get_ref<Json::array_t &>().resize(remembered);
    copy.stop();
    ExportTimer grid(EXPORT_VISIBILITY_GRID);
    Json visibility = std::move(result["visibility_grid"]);
    if (!visibility.is_array()) visibility = reservedArray(gridSize*gridSize);
    visibility.get_ref<Json::array_t &>().resize(gridSize*gridSize);
    const Region3D region = extent();
    for (int y = 0; y < gridSize; ++y) for (int x = 0; x < gridSize; ++x) {
        const Coord3D p = gridPoint(region, x, y);
        visibility[y*gridSize+x] = static_cast<int>(ThePartitionManager->getShroudStatusForPlayer(index, &p));
    }
    grid.stop();
    ExportTimer assembly(EXPORT_OBSERVATION_ASSEMBLY);
    Json playerState = std::move(result["player"]);
    if (!playerState.is_object()) playerState = reservedObject(10);
    playerState["index"] = index;
    setString(playerState["faction"], player->getPlayerTemplate()->getName().str());
    playerState["cash"] = player->getMoney()->countMoney();
    playerState["power_production"] = player->getEnergy()->getProduction();
    playerState["power_consumption"] = player->getEnergy()->getConsumption();
    playerState["radar"] = static_cast<bool>(player->hasRadar());
    playerState["rank"] = player->getRankLevel();
    playerState["skill_points"] = player->getSkillPoints();
    playerState["science_points"] = player->getSciencePurchasePoints();
    playerState["dead"] = static_cast<bool>(player->isPlayerDead());
    auto &commands = lookups.player;
    training_commands::playerCommands(player, commands);
    for (const auto &field : {std::make_pair("sciences", &commands.sciences),
                              std::make_pair("purchasable_sciences", &commands.purchasableSciences),
                              std::make_pair("upgrades", &commands.upgrades)}) {
        playerState[field.first] = Json::array();
        for (const char *name : *field.second) playerState[field.first].push_back(name);
    }
    result["player"] = std::move(playerState);
    result["own_entities"] = std::move(own);
    result["known_nonowned_entities"] = std::move(known);
    result["visibility_grid"] = std::move(visibility);
    result["observation_policy_version"] = 7;
    result["last_submission"] = commandContext(submissions, player, frame, std::move(result["last_submission"]));
    result["last_validation"] = commandContext(validations, player, frame, std::move(result["last_validation"]));
    return result;
}

#include "Common/TrainingDirectCapture.h"
} // namespace

Bool TrainingExportProfilingEnabled() { return profileExport; }
void TrainingExportProfilePlacement(TrainingPlacementProfilePart part, double seconds)
{
    auto &counter = exportCounters[EXPORT_PLACEMENT_OVERLAP + static_cast<int>(part)];
    counter.seconds += seconds;
    ++counter.calls;
}

void TrainingExportProfileCommands(TrainingCommandsProfilePart part, double seconds)
{
    auto &counter = exportCounters[EXPORT_COMMAND_BUTTONS + static_cast<int>(part)];
    counter.seconds += seconds;
    ++counter.calls;
}

Bool TrainingObjectVisibleTo(const Object *object, Int playerIndex)
{
    ExportTimer timer(EXPORT_VISIBILITY_CHECK);
    const Player *viewer = ThePlayerList->getNthPlayer(playerIndex);
    if (!object || !viewer || object->isEffectivelyDead() || object->isDestroyed()) return FALSE;
    if (object->getControllingPlayer() == viewer) return TRUE;
    // Contained objects have no partition data and report CLEAR as a fallback.
    // Visibility of the transport/building does not reveal its occupants.
    if (object->getContainedBy()) return FALSE;
    const ObjectShroudStatus shroud = object->getShroudedStatus(playerIndex);
    if (shroud != OBJECTSHROUD_CLEAR && shroud != OBJECTSHROUD_PARTIAL_CLEAR) return FALSE;
    if (StealthUpdate *stealth = object->getStealth()) {
        // The engine makes detection visible globally, but each viewer still
        // needs current shroud visibility. Never query the observed/local player.
        const StealthLookType look = stealth->getStealthLookForPlayer(viewer);
        // Disguises need an apparent-identity contract; do not disclose real IDs.
        return look != STEALTHLOOK_INVISIBLE && look != STEALTHLOOK_DISGUISED_ENEMY;
    }
    return !object->testStatus(OBJECT_STATUS_STEALTHED) || object->testStatus(OBJECT_STATUS_DETECTED)
        || viewer->getRelationship(object->getTeam()) == ALLIES;
}

void TrainingObservationReset()
{
    for (auto &memory : sightings) memory.clear();
    for (auto &memory : directSightings) memory.clear();
    observedSightingRemovals.fill(0); temporarySightingsForgotten.fill(0);
    submissions.fill(Json()); validations.fill(Json());
    liveObservations.fill(Json());
    observationLookups.commandSets.clear(); observationLookups.docks.clear();
}

void TrainingObserveObjectRemoval(const Object *object)
{
    if (!TheGlobalData || !TheGlobalData->m_headless || !ThePlayerList || !object) return;
    const unsigned id = static_cast<unsigned>(object->getID());
    for (int player = 0; player < MAX_PLAYER_COUNT; ++player)
        if (training_sightings::retireObserved(sightings[player], directSightings[player], id,
            [object, player] { return TrainingObjectVisibleTo(object, player); }))
            ++observedSightingRemovals[player];
}

void TrainingObserveCommand(const GameMessage *message, const AIGroup *selection)
{
    if (!TheGlobalData->m_headless) return;
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
        training_commands::Json action;
        if (training_commands::resolve(message, selection, action, reason)) resolved = Json::parse(action.dump());
        else if (reason.empty()) reason = "unresolved";
        if (active && !failed) {
            const auto found = std::find_if(pendingResolvedCommands.begin(), pendingResolvedCommands.end(),
                [message](const auto &entry) { return entry.first == message; });
            if (found != pendingResolvedCommands.end()) try {
                Json result = resolved.is_null() ? Json{{"resolved", false}, {"reason", reason}}
                                                 : Json{{"resolved", true}, {"action", resolved}};
                pendingSamples.at(found->second.first)["actions"].at(found->second.second)["command_result"] = std::move(result);
            } catch (const std::exception &e) {
                failed = true;
                printf("Training command resolution capture failed: %s\n", e.what());
            }
        }
    }
    Json j = std::move(submissions[owner->getPlayerIndex()]);
    if (!j.is_object()) j = reservedObject(8);
    j["available"] = true;
    j["frame"] = TheGameLogic->getFrame();
    setString(j["kind"], "other");
    if (!j["actors"].is_array()) j["actors"] = Json::array();
    j["actors"].clear();
    j["target"] = taskStatus("none", std::move(j["target"]));
    setString(j["result"], "submitted");
    if (selection) {
        j["actors"].get_ref<Json::array_t &>().reserve(selection->getAllIDs().size());
        for (ObjectID id : selection->getAllIDs()) j["actors"].push_back(static_cast<unsigned>(id));
    }
    int point = -1, target = -1, what = -1;
    switch (type) {
    case GameMessage::MSG_DO_MOVETO: setString(j["kind"], "move"); point = 0; break;
    case GameMessage::MSG_DO_ATTACKMOVETO: setString(j["kind"], "attack_move"); point = 0; break;
    case GameMessage::MSG_DO_ATTACK_OBJECT: setString(j["kind"], "attack"); target = 0; break;
    case GameMessage::MSG_DOCK: setString(j["kind"], "dock"); target = 0; break;
    case GameMessage::MSG_RESUME_CONSTRUCTION: setString(j["kind"], "resume_construction"); target = 0; break;
    case GameMessage::MSG_DO_STOP: setString(j["kind"], "stop"); break;
    case GameMessage::MSG_SELL: setString(j["kind"], "sell"); break;
    case GameMessage::MSG_DOZER_CONSTRUCT: setString(j["kind"], "build"); point = 1; what = 0; break;
    case GameMessage::MSG_QUEUE_UNIT_CREATE: setString(j["kind"], "train"); what = 0; break;
    case GameMessage::MSG_SET_RALLY_POINT:
        setString(j["kind"], "rally"); point = 1;
        arraySlot(j["actors"], 0) = static_cast<unsigned>(message->getArgument(0)->objectID);
        j["actors"].get_ref<Json::array_t &>().resize(1); break;
    default: break;
    }
    if (resolved.is_object()) {
        // Canonical actors, target and position; the raw message layouts vary.
        setString(j["kind"], resolved["kind"].get_ref<const std::string &>());
        j["actors"].clear();
        if (resolved.contains("units")) for (const Json &id : resolved["units"]) j["actors"].push_back(id);
        if (resolved.contains("producer")) j["actors"].push_back(resolved["producer"]);
        if (resolved.contains("target"))
            j["target"] = taskReference(static_cast<ObjectID>(resolved["target"].get<unsigned>()), owner, std::move(j["target"]));
        if (resolved.contains("position")) reuseJson(j["position"], resolved["position"]);
        else j.erase("position");
        j.erase("template");
        submissions[owner->getPlayerIndex()] = std::move(j);
        return;
    }
    if (point >= 0 && message->getArgumentCount() > point && message->getArgumentDataType(point) == ARGUMENTDATATYPE_LOCATION)
        j["position"] = position(&message->getArgument(point)->location, std::move(j["position"]));
    else j.erase("position");
    if (target >= 0 && message->getArgumentCount() > target && message->getArgumentDataType(target) == ARGUMENTDATATYPE_OBJECTID)
        j["target"] = taskReference(message->getArgument(target)->objectID, owner, std::move(j["target"]));
    if (what >= 0 && message->getArgumentCount() > what && message->getArgumentDataType(what) == ARGUMENTDATATYPE_INTEGER) {
        const ThingTemplate *t = TheThingFactory->findByTemplateID(message->getArgument(what)->integer);
        if (t) setString(j["template"], t->getName().str());
        else j.erase("template");
    } else j.erase("template");
    submissions[owner->getPlayerIndex()] = std::move(j);
}

void TrainingObserveValidation(Int player, const std::string &feedback)
{
    const Json result = Json::parse(feedback);
    const Json &action = result.at("action");
    if (!action.is_object()) return;
    Player *owner = ThePlayerList->getNthPlayer(player);
    Json j = std::move(validations.at(player));
    if (!j.is_object()) j = reservedObject(9);
    j["available"] = true;
    j["frame"] = TheGameLogic->getFrame();
    if (action.contains("kind") && action["kind"].is_string())
        setString(j["kind"], action["kind"].get_ref<const std::string &>());
    else setString(j["kind"], "other");
    if (!j["actors"].is_array()) j["actors"] = Json::array();
    j["actors"].clear();
    j["target"] = taskStatus("none", std::move(j["target"]));
    setString(j["result"], result.at("accepted").get<bool>() ? "accepted" : "rejected");
    auto addActor = [&](const Json &id) {
        if (!id.is_number_unsigned() && !id.is_number_integer()) return;
        Object *o = TheGameLogic->findObjectByID(static_cast<ObjectID>(id.get<unsigned>()));
        if (o && o->getControllingPlayer() == owner) j["actors"].push_back(id);
    };
    if (action.contains("units") && action["units"].is_array())
        for (const Json &id : action["units"]) addActor(id);
    for (const char *key : {"builder", "producer"}) if (action.contains(key)) addActor(action[key]);
    if (action.contains("target") && action["target"].is_number())
        j["target"] = taskReference(static_cast<ObjectID>(action["target"].get<unsigned>()), owner, std::move(j["target"]));
    if (action.contains("template") && action["template"].is_string())
        setString(j["template"], action["template"].get_ref<const std::string &>());
    else j.erase("template");
    if (action.contains("position") && action["position"].is_array() && action["position"].size() == 3
        && std::all_of(action["position"].begin(), action["position"].end(), [](const Json &v) { return v.is_number(); }))
        reuseJson(j["position"], action["position"]);
    else j.erase("position");
    static const std::string empty;
    const std::string &reason = result.contains("reason") ? result["reason"].get_ref<const std::string &>() : empty;
    setString(j["reason"], reason.empty() ? "none" : reason.find("observable") != std::string::npos ? "unavailable_target"
        : reason.find("placement") != std::string::npos || reason.find("location") != std::string::npos ? "placement"
        : reason.find("afford") != std::string::npos || reason.find("cash") != std::string::npos ? "resources"
        : reason.find("cannot") != std::string::npos || reason.find("available") != std::string::npos ? "capability" : "invalid");
    validations[player] = std::move(j);
}
std::string TrainingBuildPlacementJSON(Int player, unsigned builder, const std::string &what, Int angle, Int bins)
{
    return placementGrid(player, builder, what, angle, bins).dump();
}
std::string TrainingObservationJSON(Int index)
{
    Player *player = ThePlayerList->getNthPlayer(index);
    if (!player) throw std::runtime_error("Unknown observation player");
    Json &snapshot = liveObservations.at(index);
    snapshot = observation(player, TheGameLogic->getFrame(), std::move(snapshot));
    return snapshot.dump();
}
std::string TrainingMapJSON() { return mapFeatures().dump(); }
namespace {
Json templateCatalog()
{
    Json catalog = reservedObject(templateCount());
    for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate())
        catalog[std::to_string(t->getTemplateID())] = t->getName().str();
    return catalog;
}

} // namespace
std::string TrainingTemplateCatalogJSON() { return templateCatalog().dump(); }

Bool TrainingHasAttackMoveCommand(Object *object)
{
    return hasAttackMoveCommand(TheControlBar->findCommandSet(object->getCommandSetString()));
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
        if (passengers) for (Object *passenger : *passengers)
            if (contain->isPassengerAllowedToFire(passenger->getID()) && TrainingCanAttack(passenger))
                return TRUE;
    }
    SpawnBehaviorInterface *spawn = object->getSpawnBehaviorInterface();
    return spawn && spawn->canAnySlavesAttack();
}

Bool TrainingCanDock(Object *actor, Object *target)
{
    if (!actor || !target || actor == target || actor->isEffectivelyDead() || target->isEffectivelyDead()
        || !actor->getControllingPlayer() || !actor->getAIUpdateInterface()) return FALSE;
    const int player = actor->getControllingPlayer()->getPlayerIndex();
    if (target->getControllingPlayer() != actor->getControllingPlayer() && !TrainingObjectVisibleTo(target, player))
        return FALSE;
    return TheActionManager->canDockAt(actor, target, CMD_FROM_PLAYER);
}

Bool TrainingCanResumeConstruction(Object *actor, Object *target)
{
    if (!actor || !target || actor == target || actor->isEffectivelyDead() || target->isEffectivelyDead()
        || !actor->getControllingPlayer() || actor->getControllingPlayer() != target->getControllingPlayer()
        || !actor->getAIUpdateInterface() || !actor->getAIUpdateInterface()->getDozerAIInterface()) return FALSE;
    // This is an owned foundation, so no hidden opponent state enters legality.
    // The engine handles active builders and its configured post-death delay.
    return TheActionManager->canResumeConstructionOf(actor, target, CMD_FROM_PLAYER);
}

namespace {
Json observationPolicy()
{
    std::vector<const ThingTemplate *> classes;
    for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate())
        classes.push_back(static_cast<const ThingTemplate *>(t->getFinalOverride()));
    std::stable_sort(classes.begin(), classes.end(), [](const auto *a, const auto *b) {
        return std::strcmp(a->getName().str(), b->getName().str()) < 0;
    });
    Json excluded = reservedArray(classes.size());
    Json visibleOnly = reservedArray(classes.size());
    Json neutralExcluded = reservedArray(classes.size());
    for (size_t i = 0; i < classes.size(); ++i) {
        // Preserve the old map's sorted names and last-value-wins semantics.
        if (i+1 < classes.size() && classes[i]->getName() == classes[i+1]->getName()) continue;
        if (!retainedEntityTemplate(classes[i])) { excluded.push_back(classes[i]->getName().str()); continue; }
        if (!persistentSightingTemplate(classes[i])) visibleOnly.push_back(classes[i]->getName().str());
        if (!retainedNeutralTemplate(classes[i])) neutralExcluded.push_back(classes[i]->getName().str());
    }
    Json playable = Json::array();
    for (int i = 0; i < ThePlayerList->getPlayerCount(); ++i)
        if (ThePlayerList->getNthPlayer(i)->isPlayableSide()) playable.push_back(i);
    Json result = reservedObject(8);
    result["version"] = 7;
    result["excluded_templates"] = std::move(excluded);
    result["visible_only_templates"] = std::move(visibleOnly);
    // These templates are omitted unless a playable side controls the object.
    result["neutral_excluded_templates"] = std::move(neutralExcluded);
    result["playable_players"] = std::move(playable);
    result["visibility"] = "explicit-player shroud and engine stealth look; visible foundations retained; nonowned occupants and disguised identities omitted";
    result["sighting_memory"] = "witnessed removals retired; delivery/strike and finite-lifetime objects visible only; ordinary hidden sightings frozen";
    result["neutral_entities"] = "non-playable sides keep supply, tech/capturable, garrison/transport, repair, crate, hazard, bridge and armed faction-side objects only; civilian traffic is scenery";
    return result;
}

} // namespace
std::string TrainingObservationPolicyJSON() { return observationPolicy().dump(); }

void TrainingExporterConfigure(const char *directory) { outputDirectory = directory; }
void TrainingExporterSetInterval(Int milliseconds) { intervalMilliseconds = milliseconds; }
void TrainingExporterSetPlacementMode(Bool fullMask) { fullPlacementMask = fullMask; }
void TrainingExporterSetBinaryFormat(Bool binary) { binaryFormat = binary; }
void TrainingExporterSetScheduledQuietCapture(Bool scheduled) { scheduledQuietCapture = scheduled; }
void TrainingExporterBuildResult(const GameMessage *message, const Object *building)
{
    if (!active || failed) return;
    const auto found = std::find_if(pendingBuildCommands.begin(), pendingBuildCommands.end(),
        [message](const auto &entry) { return entry.first == message; });
    if (found == pendingBuildCommands.end()) return;
    try {
        Json result = {{"accepted", building != nullptr}, {"status", building ? "created" : "rejected"}};
        if (building) result["building_id"] = static_cast<unsigned>(building->getID());
        pendingSamples.at(found->second.first)["actions"].at(found->second.second)["build_result"] = std::move(result);
    } catch (const std::exception &e) {
        failed = true;
        printf("Training build-result capture failed: %s\n", e.what());
    }
}
Bool TrainingExporterIsEnabled() { return !outputDirectory.empty(); }

Bool TrainingExporterBegin(const AsciiString &replay)
{
    if (!TrainingExporterIsEnabled()) return TRUE;
    try {
        if (!TheGlobalData->m_headless) throw std::runtime_error("Training export requires -headless");
        if (scheduledQuietCapture && !binaryFormat) throw std::runtime_error("Scheduled quiet capture requires binary format");
        std::filesystem::create_directories(outputDirectory);
        finalPath = outputDirectory / (std::filesystem::path(replay.str()).filename().string()
            + (binaryFormat ? ".training.bin" : ".training.jsonl"));
        temporaryPath = finalPath;
        temporaryPath += ".part";
        output.open(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open training output");
        binaryWriter.reset();
        if (binaryFormat) binaryWriter = std::make_unique<training_binary::Writer<Json>>(output);
        active = true;
        profileAllocations = std::getenv("GENERALS_AI_PROFILE_ALLOCATIONS") != nullptr;
        jsonAllocations = {};
        profileExport = std::getenv("GENERALS_AI_PROFILE_EXPORT") != nullptr;
        for (auto &counter : exportCounters) counter = ScopeCounter{};
        exportStarted = std::chrono::steady_clock::now();
        failed = mapWritten = false;
        samples = commands = waitSamples = crcMessages = 0;
        pendingSamples.fill(Json()); pendingCount = 0; pendingBuildCommands.clear(); pendingResolvedCommands.clear();
        for (auto &buffer : pendingDirectBuffers) buffer.clear();
        TrainingObservationReset();
        lastSample.fill(std::nullopt); lastCommand.fill(std::nullopt);
        quietDirectives.fill(QuietDirective{});
        for (auto &selection : replaySelection) selection.clear();
        for (auto &teams : replayTeams) for (auto &team : teams) team.clear();
        pendingHeader = reservedObject(21);
        pendingHeader["record"] = "header";
        pendingHeader["schema_version"] = 1;
        pendingHeader["replay"] = replay.str();
        pendingHeader["engine_git_sha"] = static_cast<const char *>(GitSHA1);
        pendingHeader["engine_git_dirty"] = GitUncommittedChanges;
        pendingHeader["engine_exe_crc"] = TheGlobalData->m_exeCRC;
        pendingHeader["engine_ini_crc"] = TheGlobalData->m_iniCRC;
        pendingHeader["ticks_per_second"] = static_cast<int>(LOGICFRAMES_PER_SECOND);
        pendingHeader["interval_ms"] = intervalMilliseconds;
        pendingHeader["quiet_capture"] = scheduledQuietCapture ? "scheduled" : "all";
        pendingHeader["replay_logic"] = TheReplayLogic == REPLAY_LOGIC_081326 ? "081326" : "current";
        pendingHeader["controlled_faction"] = "FactionChina";
        pendingHeader["placement_mode"] = fullPlacementMask ? "full_mask" : "fast";
        pendingHeader["template_catalog"] = templateCatalog();
        pendingHeader["observation_boundary"] = "after recorder update, before this tick's command batch";
        pendingHeader["action_format"] = "ordered engine messages; selection messages retained; camera/CRC omitted";
        pendingHeader["visibility_policy"] = "per-player shroud and engine stealth look; no hidden target refresh";
        return TRUE;
    } catch (const std::exception &e) {
        printf("Training export: %s\n", e.what());
        output.close(); active = false;
        return FALSE;
    }
}

void TrainingExporterCapturePreCommands()
{
    if (!active || failed || !TheGameLogic->isInGame() || !ThePartitionManager) return;
    ExportTimer timer(EXPORT_CAPTURE);
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
        const unsigned cadence = std::max(1, (intervalMilliseconds * LOGICFRAMES_PER_SECOND + 999) / 1000);
        for (int i = 0; i < ThePlayerList->getPlayerCount(); ++i) {
            Player *p = ThePlayerList->getNthPlayer(i);
            if (!p || !p->getPlayerTemplate() || strcmp(p->getPlayerTemplate()->getName().str(), "FactionChina")) continue;
            Json actions = actionsFor(i);
            if (actions.empty() && frame % cadence != 0) continue;
            if (lastSample[i].has_value() && lastSample[i] == frame) continue;
            ExportTimer assembly(EXPORT_SAMPLE_ASSEMBLY);
            Json sample = std::move(pendingSamples[i]);
            if (!sample.is_object()) sample = reservedObject(9);
            setString(sample["record"], "sample");
            sample["frame"] = frame;
            sample["player_index"] = i;
            sample["delta_ticks"] = lastSample[i].has_value() ? frame - *lastSample[i] : 0;
            sample["time_seconds"] = static_cast<double>(frame) / static_cast<int>(LOGICFRAMES_PER_SECOND);
            if (binaryFormat) sample.erase("observation");
            else sample["observation"] = observation(p, frame, std::move(sample["observation"]));
            const bool waiting = actions.empty();
            const unsigned actionCount = static_cast<unsigned>(actions.size());
            const bool observed = !scheduledQuietCapture || !waiting || !lastSample[i].has_value()
                || quietObservationWanted(i, frame);
            if (observed) {
                sample.erase("light");
                quietDirectives[i].current = false;
            } else sample["light"] = true;
            setString(sample["decision"], waiting ? "wait" : "commands");
            if (actions.is_null()) actions = Json::array();
            sample["actions"] = std::move(actions);
            assembly.stop();
            // Replay selection bookkeeping is needed only for full-grid export.
            if (binaryFormat && observed)
                directObservation(p, frame, sample["actions"], pendingDirectBuffers[i]);
            else if (binaryFormat) {
                directSightingUpdate(p, frame);
                pendingDirectBuffers[i].clear();
            }
            else if (fullPlacementMask) sample["observation"]["build_placements"] = replayPlacements(i, sample["actions"]);
            sample["ticks_since_previous_command"] = lastCommand[i].has_value()
                ? Json(frame - *lastCommand[i]) : Json(nullptr);
            pendingSamples[i] = std::move(sample);
            pendingPlayers[pendingCount++] = i;
            lastSample[i] = frame;
            ++samples;
            if (waiting) ++waitSamples;
            else { lastCommand[i] = frame; commands += actionCount; }
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
        if (active) flushSamples();
        if (active) write({{"record", "footer"}, {"valid", valid},
                           {"playback_completed", static_cast<bool>(playbackSucceeded)},
                           {"crc_mismatch", static_cast<bool>(crcMismatch)},
                           {"recorded_crc_messages_seen", crcMessages},
                           {"recorded_crc_messages_compared", TheRecorder->getRecordedCRCComparisons()},
                           {"replay_stream_eof", static_cast<bool>(TheRecorder->reachedPlaybackEOF())},
                           {"last_recorded_command_frame", TheRecorder->getLastPlaybackCommandFrame()},
                           {"samples", samples}, {"commands", commands}, {"wait_samples", waitSamples},
                           {"final_frame", TheGameLogic->getFrame()},
                           {"observed_sighting_removals", observedSightingRemovals},
                           {"temporary_sightings_forgotten", temporarySightingsForgotten},
                           {"result", valid ? "recorded simulation CRC checks passed" : "not training eligible"}});
        output.close();
        if (active) {
            binaryWriter.reset();
            std::filesystem::path destination = finalPath;
            if (!valid) destination += ".failed";
            if (!MoveFileExW(temporaryPath.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Cannot finalize training output");
            printf("Training export: %u samples, %u commands, %u waits -> %s\n",
                   samples, commands, waitSamples, destination.string().c_str());
        }
    } catch (const std::exception &e) {
        printf("Training export finish failed: %s\n", e.what()); valid = false;
    }
    {
        // Records now survive between snapshots. Measure their actual teardown
        // once, rather than timing moved-from temporaries on every sample.
        ExportTimer destroy(EXPORT_RECORD_DESTROY);
        pendingSamples.fill(Json());
        pendingCount = 0;
        pendingBuildCommands.clear();
    pendingResolvedCommands.clear();
    }
    if (profileExport) {
        const char *names[] = {"capture", "observation", "entity", "task", "dock_targets",
                              "resume_targets", "placement", "json_write",
                              "placement_overlap", "placement_path", "placement_terrain", "command_set", "dock_scan",
                              "json_dump", "stream_write", "task_defaults", "entity_base", "capabilities", "entity_links",
                              "production", "menus", "sighting_flags", "world_scan", "sighting_copy", "visibility_grid",
                              "observation_assembly", "visibility_check", "actions", "sample_assembly", "record_destroy",
                              "direct_observation", "direct_entity", "direct_task", "commands",
                              "command_buttons", "command_object_targets", "command_targets"};
        Json timings = Json::object();
        for (int i = 0; i < EXPORT_SCOPES; ++i)
            timings[names[i]] = {{"seconds", exportCounters[i].seconds}, {"calls", exportCounters[i].calls}};
        Json report = {{"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-exportStarted).count()},
                       {"scopes_nested", timings}};
        printf("TrainingExportProfile %s\n", report.dump().c_str());
    }
    if (profileAllocations) {
        const AllocationCounter measured = jsonAllocations;
        printf("TrainingExportAllocations {\"json_calls\":%llu,\"json_requested_bytes\":%llu}\n",
               measured.calls, measured.bytes);
    }
    active = false;
    profileExport = false;
    profileAllocations = false;
    recordWriter.reset();
    return valid;
}

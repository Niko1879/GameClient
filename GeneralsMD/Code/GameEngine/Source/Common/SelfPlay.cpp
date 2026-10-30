#include "PreRTS.h"
#include "Common/SelfPlay.h"
#include "Common/TrainingExporter.h"
#include "Common/TrainingCommands.h"
#include "Common/BuildAssistant.h"
#include "Common/GameEngine.h"
#include "Common/GlobalData.h"
#include "Common/MessageStream.h"
#include "Common/Money.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "Common/PlayerTemplate.h"
#include "Common/RandomValue.h"
#include "Common/Recorder.h"
#include "Common/ThingFactory.h"
#include "Common/ThingTemplate.h"
#include "GameClient/ControlBar.h"
#include "GameClient/ChallengeGenerals.h"
#include "GameLogic/GameLogic.h"
#include "GameLogic/Object.h"
#include "GameLogic/TerrainLogic.h"
#include "GameLogic/Module/ProductionUpdate.h"
#include "GameNetwork/GameInfo.h"
#include "GameNetwork/GeneralsOnline/json.hpp"
#include "gitinfo.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::ordered_json;
bool enabled = false, live = false, awaitingReset = false, mapPending = false, recording = false;
bool retaliationPending = false;
unsigned nextDecision = 0, maxFrames = 36000;
int requestID = 0;
int playerIndices[2] = {-1, -1};
bool botSlots[2] = {false, false};
Json episodeSetup;
Json pendingReset, lastFeedback = Json::array();
std::vector<GameMessage *> pendingCRCs;
struct LeaveEpisode {};
struct CloseEnvironment {};

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
bool skirmishFaction(const PlayerTemplate *p)
{
    if (!p || !p->isPlayableSide() || p->isObserver() || p->getStartingBuilding().isEmpty()) return false;
    const GeneralPersona *general = TheChallengeGenerals->getGeneralByTemplateName(p->getName());
    return !general || general->isStartingEnabled();
}
void send(Json response)
{
    response["request_id"] = requestID;
    std::cout << "GAI " << response.dump() << std::endl;
}
Json receive()
{
    std::string line;
    if (!std::getline(std::cin, line)) throw CloseEnvironment();
    require(line.size() <= 1024 * 1024, "Request too large");
    Json request = Json::parse(line);
    require(request.is_object(), "Expected request object");
    requestID = request.at("request_id").get<int>();
    return request;
}
Object *ownObject(int player, const Json &id)
{
    Object *o = TheGameLogic->findObjectByID(static_cast<ObjectID>(id.get<unsigned>()));
    require(o && !o->isEffectivelyDead() && o->getControllingPlayer()
        && o->getControllingPlayer()->getPlayerIndex() == player, "Unit is missing or not owned by this player");
    return o;
}
Coord3D location(const Json &value)
{
    require(value.is_array() && (value.size() == 2 || value.size() == 3), "Expected position [x,y,z]");
    Coord3D p = {value[0].get<float>(), value[1].get<float>(), 0};
    Region3D extent;
    TheTerrainLogic->getExtent(&extent);
    require(std::isfinite(p.x) && std::isfinite(p.y) && p.x >= extent.lo.x && p.x <= extent.hi.x
        && p.y >= extent.lo.y && p.y <= extent.hi.y, "Position outside map");
    // Terrain height, not an arbitrary policy supplied altitude.
    p.z = TheTerrainLogic->getGroundHeight(p.x, p.y);
    return p;
}
bool offers(Object *object, const ThingTemplate *what, GUICommandType command)
{
    const CommandSet *set = TheControlBar->findCommandSet(object->getCommandSetString());
    if (!set) return false;
    for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
        const CommandButton *button = set->getCommandButton(i);
        if (button && button->getCommandType() == command && button->getThingTemplate() == what) return true;
    }
    return false;
}
GameMessage *message(GameMessage::Type type, int player)
{
    GameMessage *m = newInstance(GameMessage)(type);
    m->friend_setPlayerIndex(player);
    TheCommandList->appendMessage(m);
    return m;
}
void stopRecording()
{
    // Close the file before the final state is sent, so the client can move it.
    if (recording && TheRecorder->getMode() == RECORDERMODETYPE_RECORD) TheRecorder->stopRecording();
    recording = false;
    for (GameMessage *m : pendingCRCs) deleteInstance(m);
    pendingCRCs.clear();
}
bool commandKind(const std::string &kind)
{
    return kind == "upgrade" || kind == "science" || kind == "ability" || kind == "enter" || kind == "exit" || kind == "evacuate"
        || kind == "guard" || kind == "guard_air" || kind == "cancel_production" || kind == "cancel_construction"
        || kind == "scatter" || kind == "add_waypoint" || kind == "force_attack" || kind == "force_attack_ground"
        || kind == "repair" || kind == "get_repaired";
}
void select(int player, const Json &units)
{
    // Explicit selection makes actions independent of UI/hotkey history.
    message(GameMessage::MSG_DESTROY_SELECTED_GROUP, player);
    GameMessage *m = message(GameMessage::MSG_CREATE_SELECTED_GROUP, player);
    m->appendBooleanArgument(TRUE);
    for (const Json &id : units) m->appendObjectIDArgument(static_cast<ObjectID>(id.get<unsigned>()));
}
void validateAction(int player, Json &a, float snapRadius, float snapStep, Json &feedback)
{
    const std::string kind = a.at("kind").get<std::string>();
    if (kind == "surrender") return;
    if (commandKind(kind)) { training_commands::validate(player, a); return; }
    if (kind == "resume_construction") {
        Object *builder = ownObject(player, a.at("builder"));
        Object *target = TheGameLogic->findObjectByID(static_cast<ObjectID>(a.at("target").get<unsigned>()));
        require(TrainingCanResumeConstruction(builder, target), "Cannot resume construction at this target");
        return;
    }
    if (kind == "move" || kind == "attack_move" || kind == "attack" || kind == "stop" || kind == "sell" || kind == "dock") {
        const Json &units = a.at("units");
        require(units.is_array() && !units.empty() && units.size() <= 64, "Expected 1..64 owned units");
        for (const Json &id : units) {
            Object *actor = ownObject(player, id);
            if (kind == "attack" || kind == "attack_move")
                require(TrainingCanAttack(actor), "Selected actor cannot attack");
            if (kind == "attack_move")
                require(TrainingHasAttackMoveCommand(actor) && !actor->isKindOf(KINDOF_IMMOBILE)
                    && actor->getAIUpdateInterface(), "Selected actor cannot attack-move");
        }
        if (kind == "move" || kind == "attack_move") location(a.at("position"));
        if (kind == "attack") {
            Object *target = TheGameLogic->findObjectByID(static_cast<ObjectID>(a.at("target").get<unsigned>()));
            // Identical rejection for hidden/dead/nonexistent targets: feedback
            // must not disclose whether a remembered hidden object still exists.
            require(TrainingObjectVisibleTo(target, player), "Attack target is not currently observable");
        }
        if (kind == "dock") {
            Object *target = TheGameLogic->findObjectByID(static_cast<ObjectID>(a.at("target").get<unsigned>()));
            // Uniform rejection also covers hidden/dead/nonexistent targets.
            for (const Json &id : units)
                require(TrainingCanDock(ownObject(player, id), target), "Cannot dock at this target");
        }
        return;
    }
    if (kind == "build" || kind == "train") {
        Object *producer = ownObject(player, a.at(kind == "build" ? "builder" : "producer"));
        const ThingTemplate *what = TheThingFactory->findTemplate(a.at("template").get<std::string>().c_str(), FALSE);
        require(what, "Unknown template");
        require(offers(producer, what, kind == "build" ? GUI_COMMAND_DOZER_CONSTRUCT : GUI_COMMAND_UNIT_BUILD),
                "Template is not offered by this unit's command set");
        if (kind == "build") {
            require(TheBuildAssistant->canMakeUnit(producer, what) == CANMAKE_OK, "Cannot build: requirements/money/disabled");
            const Coord3D requested = location(a.at("position"));
            const float angle = a.value("angle", 0.0f);
            require(std::isfinite(angle), "Invalid building angle");
            // Match DozerAIUpdate::construct, including its first shroud check.
            // Otherwise a request can be admitted and then silently dropped.
            // Full-mask policies choose an exact site. Fast policies choose an
            // approximate site and snap to the nearest legal point in a bounded
            // disc, deterministically and without advancing time.
            std::vector<std::pair<int, int>> nearby{{0, 0}};
            if (snapRadius > 0) {
                nearby.clear();
                const int extent = static_cast<int>(std::floor(snapRadius / snapStep));
                for (int y = -extent; y <= extent; ++y) for (int x = -extent; x <= extent; ++x)
                    if ((x*x + y*y)*snapStep*snapStep <= snapRadius*snapRadius) nearby.emplace_back(x, y);
                std::sort(nearby.begin(), nearby.end(), [](const auto &a, const auto &b) {
                    const int da = a.first*a.first + a.second*a.second, db = b.first*b.first + b.second*b.second;
                    return da != db ? da < db : a < b;
                });
            }
            Region3D bounds;
            TheTerrainLogic->getExtent(&bounds);
            bool found = false;
            for (unsigned attempt = 0; attempt < nearby.size(); ++attempt) {
                feedback["placement_attempts"] = attempt + 1;
                Coord3D p = requested;
                p.x += nearby[attempt].first * snapStep;
                p.y += nearby[attempt].second * snapStep;
                if (p.x < bounds.lo.x || p.x > bounds.hi.x || p.y < bounds.lo.y || p.y > bounds.hi.y) continue;
                p.z = TheTerrainLogic->getGroundHeight(p.x, p.y);
                if (TheBuildAssistant->isLocationLegalToBuild(&p, what, angle,
                    BuildAssistant::TERRAIN_RESTRICTIONS | BuildAssistant::CLEAR_PATH |
                    BuildAssistant::NO_OBJECT_OVERLAP | BuildAssistant::SHROUD_REVEALED,
                    producer, nullptr) != LBC_OK) continue;
                a["position"] = Json::array({p.x, p.y, p.z});
                feedback["placement_adjusted"] = attempt != 0;
                feedback["placement_distance"] = std::hypot(p.x-requested.x, p.y-requested.y);
                feedback["placement_snap_radius"] = snapRadius;
                found = true;
                break;
            }
            require(found, "Illegal building location");
        } else {
            ProductionUpdateInterface *production = producer->getProductionUpdateInterface();
            require(TheBuildAssistant->canMakeUnit(producer, what) == CANMAKE_OK, "Unit requirements or funds unavailable");
            require(production && production->canQueueCreateUnit(what) == CANMAKE_OK, "Cannot queue unit");
        }
        return;
    }
    if (kind == "rally") { ownObject(player, a.at("producer")); location(a.at("position")); return; }
    throw std::runtime_error("Unsupported action kind");
}
void applyAction(int player, const Json &a)
{
    const std::string kind = a.at("kind").get<std::string>();
    if (commandKind(kind)) {
        training_commands::apply(player, a);
    } else if (kind == "surrender") {
        message(GameMessage::MSG_SELF_DESTRUCT, player)->appendBooleanArgument(TRUE);
    } else if (kind == "build") {
        select(player, Json::array({a.at("builder")}));
        GameMessage *m = message(GameMessage::MSG_DOZER_CONSTRUCT, player);
        m->appendIntegerArgument(TheThingFactory->findTemplate(a.at("template").get<std::string>().c_str())->getTemplateID());
        m->appendLocationArgument(location(a.at("position")));
        m->appendRealArgument(a.value("angle", 0.0f));
    } else if (kind == "resume_construction") {
        select(player, Json::array({a.at("builder")}));
        message(GameMessage::MSG_RESUME_CONSTRUCTION, player)->appendObjectIDArgument(
            static_cast<ObjectID>(a.at("target").get<unsigned>()));
    } else if (kind == "train") {
        Object *producer = ownObject(player, a.at("producer"));
        select(player, Json::array({a.at("producer")}));
        GameMessage *m = message(GameMessage::MSG_QUEUE_UNIT_CREATE, player);
        m->appendIntegerArgument(TheThingFactory->findTemplate(a.at("template").get<std::string>().c_str())->getTemplateID());
        m->appendIntegerArgument(producer->getProductionUpdateInterface()->requestUniqueUnitID());
    } else if (kind == "rally") {
        GameMessage *m = message(GameMessage::MSG_SET_RALLY_POINT, player);
        m->appendObjectIDArgument(static_cast<ObjectID>(a.at("producer").get<unsigned>()));
        m->appendLocationArgument(location(a.at("position")));
    } else {
        select(player, a.at("units"));
        GameMessage::Type type = GameMessage::MSG_DO_STOP;
        if (kind == "move") type = GameMessage::MSG_DO_MOVETO;
        else if (kind == "attack_move") type = GameMessage::MSG_DO_ATTACKMOVETO;
        else if (kind == "attack") type = GameMessage::MSG_DO_ATTACK_OBJECT;
        else if (kind == "sell") type = GameMessage::MSG_SELL;
        else if (kind == "dock") type = GameMessage::MSG_DOCK;
        GameMessage *m = message(type, player);
        if (kind == "move" || kind == "attack_move") m->appendLocationArgument(location(a.at("position")));
        else if (kind == "attack" || kind == "dock") m->appendObjectIDArgument(static_cast<ObjectID>(a.at("target").get<unsigned>()));
    }
}
void identifyPlayers()
{
    for (int slot = 0; slot < 2; ++slot) {
        AsciiString name;
        name.format("player%d", slot);
        playerIndices[slot] = -1;
        for (int i = 0; i < ThePlayerList->getPlayerCount(); ++i) {
            Player *p = ThePlayerList->getNthPlayer(i);
            if (p && p->getPlayerNameKey() == NAMEKEY(name.str())) playerIndices[slot] = i;
        }
        require(playerIndices[slot] >= 0, "Could not identify controlled player");
    }
}
Json state()
{
    identifyPlayers();
    bool dead[2];
    Json observations = Json::object(), rewards = Json::object();
    for (int slot = 0; slot < 2; ++slot) {
        const int i = playerIndices[slot];
        dead[slot] = ThePlayerList->getNthPlayer(i)->isPlayerDead();
        observations[std::to_string(i)] = Json::parse(TrainingObservationJSON(i));
    }
    const bool terminated = dead[0] || dead[1];
    const bool truncated = !terminated && TheGameLogic->getFrame() >= maxFrames;
    for (int slot = 0; slot < 2; ++slot)
        rewards[std::to_string(playerIndices[slot])] = terminated && dead[0] != dead[1] ? (dead[slot] ? -1 : 1) : 0;
    Json response = {{"event", "state"}, {"frame", TheGameLogic->getFrame()},
                     {"players", Json::array({playerIndices[0], playerIndices[1]})},
                     {"observations", observations}, {"rewards", rewards},
                     {"terminated", terminated}, {"truncated", truncated},
                     {"action_feedback", lastFeedback}};
    response["controlled_players"] = Json::array();
    for (int slot = 0; slot < 2; ++slot)
        if (!botSlots[slot]) response["controlled_players"].push_back(playerIndices[slot]);
    if (mapPending) {
        response["map"] = Json::parse(TrainingMapJSON());
        response["observation_policy"] = Json::parse(TrainingObservationPolicyJSON());
        response["setup"] = episodeSetup;
        response["setup"]["resolved_factions"] = Json::array();
        for (int slot = 0; slot < 2; ++slot)
            response["setup"]["resolved_factions"].push_back(ThePlayerList->getNthPlayer(playerIndices[slot])->getPlayerTemplate()->getName().str());
        mapPending = false;
    }
    return response;
}
void step(const Json &request)
{
    const unsigned ticks = request.at("ticks").get<unsigned>();
    require(ticks > 0 && ticks <= 600, "Step ticks must be 1..600");
    const float snapRadius = request.value("build_snap_radius", 0.0f), snapStep = request.value("build_snap_step", 4.0f);
    require(std::isfinite(snapRadius) && std::isfinite(snapStep) && snapRadius >= 0 && snapRadius <= 128
            && snapStep >= 1 && snapStep <= 32, "Invalid bounded placement search");
    const Json &actions = request.at("actions");
    require(actions.is_object() && actions.size() <= 2, "Expected action batches by player index");
    // Validate request structure before injecting any commands, so a protocol
    // error cannot leave a partially submitted batch at the paused boundary.
    for (auto it = actions.begin(); it != actions.end(); ++it) {
        require(it.key() == std::to_string(playerIndices[0]) || it.key() == std::to_string(playerIndices[1]),
                "Unknown controlled player");
        for (int slot = 0; slot < 2; ++slot)
            require(!botSlots[slot] || it.key() != std::to_string(playerIndices[slot]), "Cannot command a built-in AI player");
        require(it.value().is_array() && it.value().size() <= 16, "At most 16 actions per player/step");
    }
    lastFeedback = Json::array();
    // Invalid policy proposals are rejected individually and reported; they never reach the engine dispatcher.
    for (int slot = 0; slot < 2; ++slot) {
        const int player = playerIndices[slot];
        const std::string key = std::to_string(player);
        if (!actions.contains(key)) continue;
        require(actions[key].is_array() && actions[key].size() <= 16, "At most 16 actions per player/step");
        for (const Json &a : actions[key]) {
            Json feedback = {{"player", player}, {"action", a}, {"accepted", false}};
            try {
                Json resolved = a;
                validateAction(player, resolved, snapRadius, snapStep, feedback);
                applyAction(player, resolved);
                feedback["accepted"] = true;
                if (resolved != a) feedback["executed_action"] = resolved;
            } catch (const std::exception &e) {
                feedback["reason"] = e.what();
            }
            lastFeedback.push_back(feedback);
            TrainingObserveValidation(player, feedback.dump());
        }
    }
    nextDecision = TheGameLogic->getFrame() + ticks;
}
void reset(const Json &request)
{
    const std::string map = request.at("map").get<std::string>();
    const int seed = request.value("seed", 1);
    const std::string opponent = request.value("opponent_kind", std::string("policy"));
    const std::string faction = request.value("opponent_faction", std::string("random"));
    const int learningSide = request.value("learning_side", 0);
    require(opponent == "policy" || opponent == "hard_ai", "Unknown opponent kind");
    require(learningSide == 0 || learningSide == 1, "Learning side must be 0 or 1");
    int china = -1, botFaction = PLAYERTEMPLATE_RANDOM;
    for (int i = 0; i < ThePlayerTemplateStore->getPlayerTemplateCount(); ++i) {
        const PlayerTemplate *p = ThePlayerTemplateStore->getNthPlayerTemplate(i);
        if (p->getName() == AsciiString("FactionChina")) china = i;
        if (skirmishFaction(p) && p->getName() == AsciiString(faction.c_str())) botFaction = i;
    }
    require(china >= 0, "Vanilla China faction unavailable");
    require(opponent != "hard_ai" || faction == "random" || botFaction >= 0, "Unknown playable opponent faction");
    maxFrames = request.value("max_frames", 36000u);
    require(maxFrames > 0 && maxFrames <= 3600000, "Invalid episode frame limit");
    require(!map.empty(), "Map is required");
    const Json recordReplay = request.value("record_replay", Json(false));
    require(recordReplay.is_boolean(), "record_replay must be a boolean");
    stopRecording();
    if (TheGameLogic->isInGame()) TheGameLogic->clearGameData(FALSE);
    // The isolated profile's Last Replay; the client renames it per episode.
    recording = recordReplay.get<bool>();
    TheGameEngine->setQuitting(FALSE);
    TheCommandList->reset(); TheMessageStream->reset();
    TheGameInfo = nullptr;
    delete TheSkirmishGameInfo;
    TheSkirmishGameInfo = NEW SkirmishGameInfo;
    TheSkirmishGameInfo->init();
    TheSkirmishGameInfo->reset();
    TheSkirmishGameInfo->enterGame();
    for (int i = 0; i < MAX_SLOTS; ++i) TheSkirmishGameInfo->getSlot(i)->setState(SLOT_CLOSED);
    for (int slot = 0; slot < 2; ++slot) {
        UnicodeString name;
        name.format(L"Agent%d", slot);
        GameSlot *s = TheSkirmishGameInfo->getSlot(slot);
        botSlots[slot] = opponent == "hard_ai" && slot != learningSide;
        // SLOT_BRUTAL_AI is the UI's Hard difficulty (DIFFICULTY_HARD).
        s->setState(botSlots[slot] ? SLOT_BRUTAL_AI : SLOT_PLAYER, name, slot + 1);
        s->setColor(slot); s->setStartPos(slot); s->setPlayerTemplate(botSlots[slot] ? botFaction : china); s->setTeamNumber(-1);
        s->setAccept(); s->setMapAvailability(TRUE);
    }
    Money cash;
    cash.deposit(request.value("starting_cash", 10000u), FALSE);
    TheSkirmishGameInfo->setStartingCash(cash);
    TheSkirmishGameInfo->setMap(map.c_str());
    TheSkirmishGameInfo->setSeed(seed);
    TheSkirmishGameInfo->setUseStats(FALSE);
    TheSkirmishGameInfo->setLocalIP(opponent == "hard_ai" ? learningSide + 1 : 1);
    TheSkirmishGameInfo->startGame(0);
    TheWritableGlobalData->m_pendingFile = map.c_str();
    TrainingObservationReset();
    InitRandom(seed);
    GameMessage *m = message(GameMessage::MSG_NEW_GAME, 0);
    m->appendIntegerArgument(GAME_SKIRMISH);
    m->appendIntegerArgument(DIFFICULTY_NORMAL);
    m->appendIntegerArgument(0);
    // Let newly spawned objects perform their initialization update before
    // accepting policy orders; a frame-zero dozer order can be overwritten.
    nextDecision = 1;
    lastFeedback = Json::array();
    mapPending = true;
    retaliationPending = true;
    episodeSetup = {{"opponent_kind", opponent}, {"opponent_faction", faction}, {"learning_side", learningSide},
                    {"starting_cash", request.value("starting_cash", 10000u)}, {"record_replay", recording}};
    live = true;
}

Json actionMenus()
{
    Json menus = Json::object();
    for (const ThingTemplate *t = TheThingFactory->firstTemplate(); t; t = t->friend_getNextTemplate()) {
        const std::string name = t->getName().str();
        // China can capture foreign producers and acquire other factions' units.
        // Publish all static production menus; observations supply live legality.
        const CommandSet *set = TheControlBar->findCommandSet(t->friend_getCommandSetString());
        if (!set) continue;
        for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
            const CommandButton *button = set->getCommandButton(i);
            if (!button || !button->getThingTemplate()) continue;
            const GUICommandType command = button->getCommandType();
            if (command != GUI_COMMAND_DOZER_CONSTRUCT && command != GUI_COMMAND_UNIT_BUILD) continue;
            const std::string key = name + (command == GUI_COMMAND_DOZER_CONSTRUCT ? "|build" : "|train");
            if (!menus.contains(key)) menus[key] = Json::array();
            menus[key].push_back(button->getThingTemplate()->getName().str());
        }
    }
    return menus;
}
} // namespace

void SelfPlayConfigure() { enabled = true; }
Bool SelfPlayIsEnabled() { return enabled; }

void SelfPlayRecordCommands()
{
    // After policy commands are injected and before they execute, matching the
    // frame a normal game records them on. Also records MSG_NEW_GAME and CRCs.
    if (!enabled || !recording) return;
    // Logic CRCs go to TheMessageStream, which this loop never propagates. A normal
    // client propagates a frame-F CRC into the frame-F+1 command list, and replay
    // playback aligns CRCs on that latency, so hold each one for one frame.
    for (GameMessage *m : pendingCRCs) TheCommandList->appendMessage(m);
    pendingCRCs.clear();
    for (GameMessage *m = TheMessageStream->getFirstMessage(); m;) {
        GameMessage *next = m->next();
        if (m->getType() == GameMessage::MSG_LOGIC_CRC) {
            TheMessageStream->removeMessage(m);
            pendingCRCs.push_back(m);
        }
        m = next;
    }
    TheRecorder->updateRecord();
}

void SelfPlayPreCommands()
{
    if (!enabled || !live || !TheGameLogic->isInGame()) return;
    const unsigned frame = TheGameLogic->getFrame();
    identifyPlayers();
    const bool ended = ThePlayerList->getNthPlayer(playerIndices[0])->isPlayerDead()
        || ThePlayerList->getNthPlayer(playerIndices[1])->isPlayerDead();
    if (frame < nextDecision && frame < maxFrames && !ended) return;
    if (retaliationPending) {
        // A human client posts its retaliation option (on by default) as a game
        // message; replays record it. Controlled slots get that same default.
        retaliationPending = false;
        for (int slot = 0; slot < 2; ++slot) {
            if (botSlots[slot]) continue;
            GameMessage *m = message(GameMessage::MSG_ENABLE_RETALIATION_MODE, playerIndices[slot]);
#if RETAIL_COMPATIBLE_CRC
            m->appendIntegerArgument(playerIndices[slot]);
#endif
            m->appendBooleanArgument(TRUE);
        }
    }
    Json response = state();
    const bool over = response["terminated"].get<bool>() || response["truncated"].get<bool>();
    if (over) stopRecording();
    send(response);
    if (over) throw LeaveEpisode();
    for (;;) {
        try {
            const Json request = receive();
            const std::string command = request.at("command").get<std::string>();
            if (command == "close") { send({{"event", "closed"}}); throw CloseEnvironment(); }
            if (command == "reset") { pendingReset = request; awaitingReset = true; throw LeaveEpisode(); }
            if (command == "placement") {
                const int player = request.at("player").get<int>();
                require((player == playerIndices[0] && !botSlots[0]) || (player == playerIndices[1] && !botSlots[1]), "Unknown controlled player");
                send({{"event", "placement"}, {"placement", Json::parse(TrainingBuildPlacementJSON(player,
                    request.at("builder").get<unsigned>(), request.at("template").get<std::string>(),
                    request.at("angle_index").get<int>(), request.at("bins").get<int>()))}});
                continue; // The simulation and the actor deadline remain paused.
            }
            require(command == "step", "Expected step/reset/close");
            step(request);
            return;
        } catch (const std::exception &e) { send({{"event", "error"}, {"message", e.what()}}); }
    }
}

Int SelfPlayRun()
{
    if (!TheGlobalData->m_headless || !TheGlobalData->m_simulateReplays.empty() || TrainingExporterIsEnabled()) {
        printf("Self-play requires -headless, without -replay or -exportTraining.\n");
        return 1;
    }
    Json factions = Json::array();
    for (int i = 0; i < ThePlayerTemplateStore->getPlayerTemplateCount(); ++i) {
        const PlayerTemplate *p = ThePlayerTemplateStore->getNthPlayerTemplate(i);
        // Match normal skirmish/random selection, including locked generals.
        if (skirmishFaction(p)) factions.push_back(p->getName().str());
    }
    send({{"event", "ready"}, {"protocol_version", 1}, {"playable_factions", factions},
          {"placement_angle_bins", 16}, {"action_menus", actionMenus()},
          {"ticks_per_second", static_cast<int>(LOGICFRAMES_PER_SECOND)},
          {"engine_git_sha", static_cast<const char *>(GitSHA1)}, {"engine_git_dirty", GitUncommittedChanges},
          {"engine_exe_crc", TheGlobalData->m_exeCRC}, {"engine_ini_crc", TheGlobalData->m_iniCRC},
          {"template_catalog", Json::parse(TrainingTemplateCatalogJSON())},
          {"command_catalog", training_commands::catalog()},
          {"observation_policy", Json::parse(TrainingObservationPolicyJSON())}});
    try {
        for (;;) {
            try {
                Json request;
                if (awaitingReset) { request = pendingReset; awaitingReset = false; }
                else request = receive();
                const std::string command = request.at("command").get<std::string>();
                if (command == "close") { send({{"event", "closed"}}); break; }
                require(command == "reset", "No active episode; reset first");
                reset(request);
                unsigned startup = 0;
                try {
                    for (;;) {
                        TheGameLogic->UPDATE();
                        // No rendering, audio, network or wall-clock pacing in this loop.
                        if (!TheGameLogic->isInGame()) {
                            require(++startup < 300, "Match failed to initialize or ended unexpectedly");
                        } else startup = 0;
                    }
                } catch (LeaveEpisode &) { live = false; }
            } catch (const std::exception &e) { live = false; send({{"event", "error"}, {"message", e.what()}}); }
        }
    } catch (CloseEnvironment &) {}
    live = false;
    stopRecording();
    if (TheGameLogic->isInGame()) TheGameLogic->clearGameData(FALSE);
    return 0;
}

#pragma once

// Included inside TrainingExporter.cpp's private namespace, after the engine
// helpers. Per-frame state is written straight into the final scalar columns.
// Json is read only for existing sparse command receipts, never built for an
// entity, task, queue, visibility grid or remembered sighting on this path.
using DirectBuffer = training_binary::Buffer;
using DirectCursor = training_binary::Cursor;

struct DirectOutput {
    training_binary::Writer<Json> &writer;
    DirectBuffer &buffer;
    DirectCursor object(uint32_t node, std::initializer_list<const char *> fields) {
        return writer.object(buffer, node, fields);
    }
    DirectCursor object(DirectCursor &parent, std::initializer_list<const char *> fields) {
        return object(parent.next(), fields);
    }
    DirectCursor array(DirectCursor &parent, size_t size) { return writer.array(buffer, parent.next(), size); }
    template<class T> void put(DirectCursor &parent, const T &value) {
        if constexpr (std::is_arithmetic_v<T>) writer.number(buffer, parent.next(), value);
        else if constexpr (std::is_enum_v<T>) writer.number(buffer, parent.next(), static_cast<std::underlying_type_t<T>>(value));
        else writer.string(buffer, parent.next(), std::string_view(value));
    }
    void null(DirectCursor &parent) { writer.null(buffer, parent.next()); }
    void point(DirectCursor &parent, const Coord3D *p) {
        auto xyz = array(parent, 3);
        put(xyz, p->x); put(xyz, p->y); put(xyz, p->z);
    }
};

struct DirectReference {
    const char *status = "none";
    ObjectID id = INVALID_ID;
    Object *target = nullptr;
};

DirectReference directReference(ObjectID id, Player *owner)
{
    if (id == INVALID_ID) return {};
    Object *target = TheGameLogic->findObjectByID(id);
    if (target && !target->isEffectivelyDead() && retainedEntity(target)
        && (target->getControllingPlayer() == owner || TrainingObjectVisibleTo(target, owner->getPlayerIndex())))
        return {"current", id, target};
    return {"unavailable", id, nullptr};
}

void directReference(DirectOutput &out, DirectCursor &parent, const DirectReference &reference)
{
    auto fields = reference.target ? out.object(parent, {"status", "id"}) : out.object(parent, {"status"});
    out.put(fields, reference.status);
    if (reference.target) out.put(fields, static_cast<unsigned>(reference.id));
    fields.finish();
}

void directTask(DirectOutput &out, DirectCursor &parent, Object *object)
{
    ExportTimer timer(EXPORT_DIRECT_TASK);
    AIUpdateInterface *ai = object->getAIUpdateInterface();
    Player *owner = object->getControllingPlayer();
    const char *kind = "unknown";
    bool moving = false, attacking = false, waiting = false, busy = false;
    bool point = false, path = false, targetTask = false;
    DirectReference target, combat, dock;
    const Coord3D *destination = nullptr;
    Coord3D savedDestination{};
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
        moving = static_cast<bool>(ai->isMoving());
        attacking = static_cast<bool>(ai->isAttacking() || object->testStatus(OBJECT_STATUS_IS_ATTACKING));
        waiting = static_cast<bool>(ai->isWaitingForPath());
        busy = static_cast<bool>(ai->isBusy());
        combat = directReference(ai->getCurrentVictimID(), owner);
        supply = ai->getSupplyTruckAIInterface();
        if (supply) {
            ferrying = supply->isCurrentlyFerryingSupplies();
            boxes = supply->getNumberBoxes();
            dock = directReference(supply->getPreferredDockID(), owner);
            if (ferrying) kind = boxes > 0 ? "return_supplies" : "collect";
        }
    } else kind = "idle";
    ProductionUpdateInterface *production = object->getProductionUpdateInterface();
    if (object->getConstructionPercent() != -1) kind = "construct";
    else if (production && production->firstProduction() && !strcmp(kind, "idle")) kind = "produce";
    auto fields = out.object(parent, {"kind", "available", "moving", "attacking", "waiting_for_path",
        "busy", "disabled", "destination", "target", "combat_target", "cargo", "preferred_dock"});
    out.put(fields, kind); out.put(fields, true); out.put(fields, moving); out.put(fields, attacking);
    out.put(fields, waiting); out.put(fields, busy); out.put(fields, static_cast<bool>(object->isDisabled()));
    if (destination) out.point(fields, destination); else out.null(fields);
    directReference(out, fields, target); directReference(out, fields, combat);
    auto cargo = supply ? out.object(fields, {"available", "boxes", "capacity", "ferrying"})
                        : out.object(fields, {"available"});
    out.put(cargo, supply != nullptr);
    if (supply) { out.put(cargo, boxes); out.put(cargo, supply->getMaxBoxes()); out.put(cargo, ferrying); }
    cargo.finish();
    directReference(out, fields, dock);
    fields.finish();
}

struct DirectSighting : training_sightings::Flags {
    unsigned lastSeen = 0;
    int owner = -1;
    std::string type;
    Coord3D position{};
    float health = 0, maximumHealth = 0;
    const ThingTemplate *typeTemplate = nullptr;
};
std::array<std::vector<DirectSighting>, MAX_PLAYER_COUNT> directSightings;
struct DirectScratch {
    std::vector<unsigned> docks, resumes;
    std::vector<const ProductionEntry *> production;
    std::vector<const ThingTemplate *> buildable, trainable;
    std::vector<const char *> contextFields;
    std::vector<std::pair<int, int>> gridCells;
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
    result.visible = true;
}

void writeDirectSighting(DirectOutput &out, DirectCursor &parent, const DirectSighting &entity)
{
    auto fields = out.object(parent, {"id", "type", "owner", "position", "health", "max_health",
                                      "last_seen_frame", "visible", "own"});
    out.put(fields, entity.id); out.put(fields, entity.type); out.put(fields, entity.owner);
    out.point(fields, &entity.position); out.put(fields, entity.health); out.put(fields, entity.maximumHealth);
    out.put(fields, entity.lastSeen); out.put(fields, entity.visible); out.put(fields, false);
    fields.finish();
}

void directEntity(DirectOutput &out, DirectCursor &parent, Object *object, unsigned frame, ObservationLookups &lookups)
{
    ExportTimer timer(EXPORT_DIRECT_ENTITY);
    Player *owner = object->getControllingPlayer();
    BodyModuleInterface *body = object->getBodyModule();
    Object *producer = TheGameLogic->findObjectByID(object->getProducerID());
    Object *builder = TheGameLogic->findObjectByID(object->getBuilderID());
    const bool hasProducer = producer && producer->getControllingPlayer() == owner;
    const bool hasBuilder = builder && builder->getControllingPlayer() == owner;
    const char *names[] = {"id", "type", "owner", "position", "health", "max_health", "last_seen_frame", "visible", "own",
        "task", "radius", "has_weapon", "can_attack", "has_atk_move_cmd", "can_attack_move", "dock_targets", "resume_targets",
        "construction_percent", "ai_state", "production_queue", "buildable_templates", "trainable_templates",
        "commands", "command_cooldowns", "command_targets", "enter_targets", "container_id", "occupants", "capacity",
        "upgrades", "can_guard", "repair_targets", "repair_depots", hasProducer ? "producer_id" : "builder_id", "builder_id"};
    const size_t count = 33+hasProducer+hasBuilder;
    auto fields = out.writer.literalObject(out.buffer, parent.next(), names, count);
    out.put(fields, static_cast<unsigned>(object->getID())); out.put(fields, object->getTemplate()->getName().str());
    out.put(fields, owner ? static_cast<int>(owner->getPlayerIndex()) : -1);
    out.point(fields, object->getPosition());
    out.put(fields, body ? body->getHealth() : 0.0f); out.put(fields, body ? body->getMaxHealth() : 0.0f);
    out.put(fields, frame); out.put(fields, true); out.put(fields, true);
    const CommandSet *set = commandSetFor(object, lookups);
    directTask(out, fields, object);
    ExportTimer capabilities(EXPORT_CAPABILITIES);
    out.put(fields, object->getGeometryInfo().getBoundingCircleRadius());
    out.put(fields, static_cast<bool>(object->hasAnyWeapon()));
    const bool canAttack = static_cast<bool>(TrainingCanAttack(object));
    const bool attackMove = hasAttackMoveCommand(set);
    out.put(fields, canAttack); out.put(fields, attackMove);
    out.put(fields, canAttack && attackMove && !object->isKindOf(KINDOF_IMMOBILE) && object->getAIUpdateInterface());
    capabilities.stop();
    auto &docks = directScratch.docks;
    docks.clear();
    if (object->getAIUpdateInterface()) {
        ExportTimer timer(EXPORT_DOCK);
        for (Object *target : lookups.docks) if (TrainingCanDock(object, target)) docks.push_back(static_cast<unsigned>(target->getID()));
    }
    auto dockList = out.array(fields, docks.size());
    for (const auto id : docks) out.put(dockList, id);
    auto &resumes = directScratch.resumes;
    resumes.clear();
    if (object->getAIUpdateInterface() && object->getAIUpdateInterface()->getDozerAIInterface()) {
        ExportTimer timer(EXPORT_RESUME);
        for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject())
            if (TrainingCanResumeConstruction(object, target)) resumes.push_back(static_cast<unsigned>(target->getID()));
    }
    auto resumeList = out.array(fields, resumes.size());
    for (const auto id : resumes) out.put(resumeList, id);
    out.put(fields, object->getConstructionPercent());
    out.put(fields, object->getAIUpdateInterface() ? object->getAIUpdateInterface()->getCurrentStateName().str() : "");
    ExportTimer productionTimer(EXPORT_PRODUCTION);
    auto &queue = directScratch.production;
    queue.clear();
    ProductionUpdateInterface *production = object->getProductionUpdateInterface();
    if (production) for (const ProductionEntry *p = production->firstProduction(); p; p = production->nextProduction(p)) queue.push_back(p);
    auto entries = out.array(fields, queue.size());
    for (const auto *p : queue) {
        // Unit and upgrade pointers share a union: only the type discriminates.
        const bool unit = p->getProductionType() == PRODUCTION_UNIT && p->getProductionObject();
        const bool upgrade = p->getProductionType() == PRODUCTION_UPGRADE && p->getProductionUpgrade();
        const char *keys[] = {"production_id", "kind", "progress", "remaining_quantity", unit ? "unit_type" : "upgrade", "upgrade"};
        auto entry = out.writer.literalObject(out.buffer, entries.next(), keys, 4+unit+upgrade);
        out.put(entry, static_cast<unsigned>(p->getProductionID())); out.put(entry, static_cast<int>(p->getProductionType()));
        out.put(entry, p->getPercentComplete()); out.put(entry, p->getProductionQuantityRemaining());
        if (unit) out.put(entry, p->getProductionObject()->getName().str());
        if (upgrade) out.put(entry, p->getProductionUpgrade()->getUpgradeName().str());
        entry.finish();
    }
    productionTimer.stop();
    ExportTimer menus(EXPORT_MENUS);
    auto &buildable = directScratch.buildable;
    auto &trainable = directScratch.trainable;
    buildable.clear(); trainable.clear();
    if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
        const CommandButton *button = set->getCommandButton(i);
        if (!button || !button->getThingTemplate()) continue;
        const ThingTemplate *what = button->getThingTemplate();
        if (button->getCommandType() == GUI_COMMAND_DOZER_CONSTRUCT && TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK)
            buildable.push_back(what);
        if (button->getCommandType() == GUI_COMMAND_UNIT_BUILD && production
            && TheBuildAssistant->canMakeUnit(object, what) == CANMAKE_OK && production->canQueueCreateUnit(what) == CANMAKE_OK)
            trainable.push_back(what);
    }
    auto buildList = out.array(fields, buildable.size());
    for (const auto *what : buildable) out.put(buildList, what->getName().str());
    auto trainList = out.array(fields, trainable.size());
    for (const auto *what : trainable) out.put(trainList, what->getName().str());
    menus.stop();
    {
        ExportTimer timer(EXPORT_COMMANDS);
        auto &state = lookups.commands;
        training_commands::objectCommands(object, set, lookups.commandTargets, state);
        auto available = out.array(fields, state.available.size());
        for (const char *name : state.available) out.put(available, name);
        auto cooldowns = out.array(fields, state.charging.size());
        for (const auto &item : state.charging) {
            auto entry = out.object(cooldowns, {"command", "ready_percent"});
            out.put(entry, item.first); out.put(entry, item.second);
            entry.finish();
        }
        auto targets = out.array(fields, state.targets.size());
        for (const auto &item : state.targets) {
            auto entry = out.object(targets, {"command", "targets"});
            out.put(entry, item.first);
            auto ids = out.array(entry, item.second.size());
            for (const unsigned id : item.second) out.put(ids, id);
            entry.finish();
        }
        auto enter = out.array(fields, state.enterTargets.size());
        for (const unsigned id : state.enterTargets) out.put(enter, id);
        out.put(fields, state.containerID); out.put(fields, state.occupants); out.put(fields, state.capacity);
        auto upgrades = out.array(fields, state.upgrades.size());
        for (const char *name : state.upgrades) out.put(upgrades, name);
        out.put(fields, state.canGuard);
        for (const auto *list : {&state.repairTargets, &state.repairDepots}) {
            auto ids = out.array(fields, list->size());
            for (const unsigned id : *list) out.put(ids, id);
        }
    }
    if (hasProducer) out.put(fields, static_cast<unsigned>(producer->getID()));
    if (hasBuilder) out.put(fields, static_cast<unsigned>(builder->getID()));
    fields.finish();
}

void directContext(DirectOutput &out, DirectCursor &parent, const std::array<Json, MAX_PLAYER_COUNT> &records,
                   Player *owner, unsigned frame)
{
    const Json &record = records.at(owner->getPlayerIndex());
    if (record.is_null()) {
        auto fields = out.object(parent, {"available"}); out.put(fields, false); return;
    }
    auto &keys = directScratch.contextFields;
    keys.clear();
    for (auto it = record.begin(); it != record.end(); ++it) if (it.key() != "age_seconds") keys.push_back(it.key().c_str());
    keys.push_back("age_seconds");
    auto fields = out.writer.object(out.buffer, parent.next(), keys.data(), keys.size());
    for (auto it = record.begin(); it != record.end(); ++it) {
        if (it.key() == "age_seconds") continue;
        if (it.key() == "target" && it.value().contains("id"))
            directReference(out, fields, directReference(static_cast<ObjectID>(it.value()["id"].get<unsigned>()), owner));
        else out.writer.copyJson(out.buffer, fields.next(), it.value());
    }
    out.put(fields, static_cast<double>(frame-record["frame"].get<unsigned>()) / static_cast<int>(LOGICFRAMES_PER_SECOND));
    fields.finish();
}

void directPlacements(DirectOutput &out, DirectCursor &parent, int player, const Json &actions);

// One sample's pass over the world: owned objects leave sighting memory (a full
// observation also writes them), visible others refresh theirs, and hidden
// temporary objects are forgotten. Light samples run exactly these updates.
template<class Own> void observeObjects(Player *player, unsigned frame, Own own)
{
    const int index = static_cast<int>(player->getPlayerIndex());
    auto &memory = directSightings[index];
    for (auto &item : memory) item.visible = false;
    for (Object *object = TheGameLogic->getFirstObject(); object; object = object->getNextObject()) {
        if (!retainedEntity(object) || object->isEffectivelyDead() || object->isDestroyed()) continue;
        const unsigned id = static_cast<unsigned>(object->getID());
        if (object->getControllingPlayer() == player) {
            own(object);
            const auto found = std::lower_bound(memory.begin(), memory.end(), id,
                [](const DirectSighting &item, unsigned key) { return item.id < key; });
            if (found != memory.end() && found->id == id) memory.erase(found);
        } else if (TrainingObjectVisibleTo(object, index)) {
            auto found = std::lower_bound(memory.begin(), memory.end(), id,
                [](const DirectSighting &item, unsigned key) { return item.id < key; });
            if (found == memory.end() || found->id != id) found = memory.emplace(found);
            refreshDirectSighting(*found, object, frame);
        }
    }
    temporarySightingsForgotten[index] += static_cast<unsigned>(training_sightings::forgetHiddenTemporary(memory));
}

void directSightingUpdate(Player *player, unsigned frame)
{
    ExportTimer timer(EXPORT_DIRECT_OBSERVATION);
    observeObjects(player, frame, [](Object *) {});
}

void directObservation(Player *player, unsigned frame, const Json &actions, DirectBuffer &buffer)
{
    ExportTimer timer(EXPORT_DIRECT_OBSERVATION);
    DirectOutput out{*binaryWriter, buffer};
    out.writer.begin(buffer);
    const char *names[] = {"player", "own_entities", "known_nonowned_entities", "visibility_grid",
        "observation_policy_version", "last_submission", "last_validation", "build_placements"};
    auto root = out.writer.literalObject(buffer, 0, names, fullPlacementMask ? 8 : 7);
    const int index = static_cast<int>(player->getPlayerIndex());
    auto state = out.object(root, {"index", "faction", "cash", "power_production", "power_consumption", "radar", "rank",
                                   "skill_points", "science_points", "dead", "sciences", "purchasable_sciences", "upgrades"});
    out.put(state, index); out.put(state, player->getPlayerTemplate()->getName().str());
    out.put(state, player->getMoney()->countMoney()); out.put(state, player->getEnergy()->getProduction());
    out.put(state, player->getEnergy()->getConsumption()); out.put(state, static_cast<bool>(player->hasRadar()));
    out.put(state, player->getRankLevel()); out.put(state, player->getSkillPoints());
    out.put(state, player->getSciencePurchasePoints()); out.put(state, static_cast<bool>(player->isPlayerDead()));
    ObservationLookups &lookups = observationLookups;
    training_commands::playerCommands(player, lookups.player);
    for (const auto *names : {&lookups.player.sciences, &lookups.player.purchasableSciences, &lookups.player.upgrades}) {
        auto list = out.array(state, names->size());
        for (const char *name : *names) out.put(list, name);
    }
    state.finish();
    lookups.commandSets.clear(); lookups.docks.clear(); lookups.commandTargets.clear();
    size_t owned = 0;
    {
        ExportTimer timer(EXPORT_DOCK_SCAN);
        for (Object *target = TheGameLogic->getFirstObject(); target; target = target->getNextObject()) {
            if (!retainedEntity(target)) continue;
            if (!target->isEffectivelyDead() && !target->isDestroyed() && target->getControllingPlayer() == player) ++owned;
            const bool dock = hasDockInterface(target);
            // TrainingCanDock also requires a live, owned or visible target.
            const bool seen = !target->isEffectivelyDead()
                && (target->getControllingPlayer() == player || TrainingObjectVisibleTo(target, index));
            if (dock && seen) lookups.docks.push_back(target);
            if (seen && !target->isDestroyed()) lookups.commandTargets.add(target, dock);
        }
    }
    auto own = out.array(root, owned);
    observeObjects(player, frame, [&](Object *object) { directEntity(out, own, object, frame, lookups); });
    own.finish();
    auto &memory = directSightings[index];
    auto known = out.array(root, memory.size());
    for (const auto &item : memory) writeDirectSighting(out, known, item);
    auto visibility = out.array(root, gridSize*gridSize);
    auto &cells = directScratch.gridCells;
    if (cells.empty()) { // the map does not change during a game
        const Region3D region = extent();
        cells.resize(gridSize*gridSize);
        for (int y = 0; y < gridSize; ++y) for (int x = 0; x < gridSize; ++x) {
            const Coord3D p = gridPoint(region, x, y);
            ThePartitionManager->worldToCell(p.x, p.y, &cells[y*gridSize+x].first, &cells[y*gridSize+x].second);
        }
    }
    for (const auto &cell : cells)
        out.put(visibility, static_cast<int>(ThePartitionManager->getShroudStatusForPlayer(index, cell.first, cell.second)));
    out.put(root, 7);
    directContext(out, root, submissions, player, frame);
    directContext(out, root, validations, player, frame);
    if (fullPlacementMask) directPlacements(out, root, index, actions);
    root.finish();
}

struct DirectPlacement {
    unsigned builder;
    const ThingTemplate *what;
    float rawAngle;
    int angle;
    Coord3D original;
};
std::vector<DirectPlacement> directPlacementRequests;

void directPlacements(DirectOutput &out, DirectCursor &parent, int player, const Json &actions)
{
    auto &requests = directPlacementRequests;
    requests.clear();
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
            unsigned builder = 0, count = 0;
            for (unsigned id : selected) if (owned(id)) { builder = id; ++count; }
            if (count != 1) continue;
            const ThingTemplate *what = TheThingFactory->findByTemplateID(args[0]["value"].get<unsigned short>());
            if (!what) continue;
            const float rawAngle = args[2]["value"];
            const int angle = (static_cast<int>(std::nearbyint(rawAngle/placementTau*16))%16+16)%16;
            Coord3D original{}; original.x = args[1]["value"][0]; original.y = args[1]["value"][1];
            requests.push_back({builder, what, rawAngle, angle, original});
        }
    }
    auto grids = out.array(parent, requests.size());
    static std::vector<unsigned char> packed;
    static std::string mask;
    for (auto request : requests) {
        ExportTimer timer(EXPORT_PLACEMENT);
        Object *builder = TheGameLogic->findObjectByID(static_cast<ObjectID>(request.builder));
        Player *owner = ThePlayerList->getNthPlayer(player);
        if (!builder || builder->getControllingPlayer() != owner || builder->isEffectivelyDead())
            throw std::runtime_error("Unknown owned builder/template");
        const CommandSet *set = TheControlBar->findCommandSet(builder->getCommandSetString());
        bool offered = false;
        if (set) for (int i = 0; i < MAX_COMMANDS_PER_SET; ++i) {
            const CommandButton *button = set->getCommandButton(i);
            if (button && button->getCommandType() == GUI_COMMAND_DOZER_CONSTRUCT && button->getThingTemplate() == request.what)
                offered = true;
        }
        if (!offered) throw std::runtime_error("Builder does not offer this building");
        const auto started = std::chrono::steady_clock::now();
        Region3D bounds; TheTerrainLogic->getExtent(&bounds);
        const float angle = placementTau * request.angle/16;
        const unsigned options = BuildAssistant::TERRAIN_RESTRICTIONS | BuildAssistant::NO_OBJECT_OVERLAP | BuildAssistant::CLEAR_PATH
            | BuildAssistant::SHROUD_REVEALED | BuildAssistant::OBSERVED_PLACEMENT_PREVIEW | BuildAssistant::NO_PLACEMENT_FEEDBACK;
        packed.assign((256*256+7)/8, 0);
        unsigned legal = 0;
        for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) {
            Coord3D p;
            p.x = static_cast<float>(bounds.lo.x+(x+.5)*(static_cast<double>(bounds.hi.x)-bounds.lo.x)/256);
            p.y = static_cast<float>(bounds.lo.y+(y+.5)*(static_cast<double>(bounds.hi.y)-bounds.lo.y)/256);
            p.z = TheTerrainLogic->getGroundHeight(p.x, p.y);
            if (TheBuildAssistant->isLocationLegalToBuild(&p, request.what, angle, options, builder, nullptr) == LBC_OK) {
                const int index = y*256+x; packed[index/8] |= 1 << (index%8); ++legal;
            }
        }
        const char hex[] = "0123456789abcdef";
        mask.clear(); mask.reserve(packed.size()*2);
        for (unsigned char value : packed) { mask += hex[value >> 4]; mask += hex[value & 15]; }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        auto grid = out.object(grids, {"schema", "builder", "template", "angle_index", "angle_bins", "bins", "mask_hex",
                                      "legal_sites", "frame", "seconds", "path_check", "round_trip_audit"});
        out.put(grid, 1); out.put(grid, request.builder); out.put(grid, request.what->getName().str());
        out.put(grid, request.angle); out.put(grid, 16); out.put(grid, 256); out.put(grid, mask);
        out.put(grid, legal); out.put(grid, TheGameLogic->getFrame()); out.put(grid, seconds);
        out.put(grid, "per-builder terrain-only UI reachability; hidden structures ignored");
        auto audit = out.object(grid, {"original", "256", "512", "angle_only"});
        request.original.z = TheTerrainLogic->getGroundHeight(request.original.x, request.original.y);
        const unsigned auditOptions = BuildAssistant::TERRAIN_RESTRICTIONS | BuildAssistant::NO_OBJECT_OVERLAP
            | BuildAssistant::SHROUD_REVEALED | BuildAssistant::CLEAR_PATH | BuildAssistant::NO_PLACEMENT_FEEDBACK;
        out.put(audit, static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(
            &request.original, request.what, request.rawAngle, auditOptions, builder, nullptr)));
        for (int resolution : {256, 512}) {
            Coord3D quantized = request.original;
            const int x = std::max(0, std::min(resolution-1, static_cast<int>((request.original.x-bounds.lo.x)/(bounds.hi.x-bounds.lo.x)*resolution)));
            const int y = std::max(0, std::min(resolution-1, static_cast<int>((request.original.y-bounds.lo.y)/(bounds.hi.y-bounds.lo.y)*resolution)));
            quantized.x = static_cast<float>(bounds.lo.x+(x+.5)*(static_cast<double>(bounds.hi.x)-bounds.lo.x)/resolution);
            quantized.y = static_cast<float>(bounds.lo.y+(y+.5)*(static_cast<double>(bounds.hi.y)-bounds.lo.y)/resolution);
            quantized.z = TheTerrainLogic->getGroundHeight(quantized.x, quantized.y);
            auto pair = out.object(audit, {"position_only", "position_and_angle"});
            out.put(pair, static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(
                &quantized, request.what, request.rawAngle, auditOptions, builder, nullptr)));
            out.put(pair, static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(
                &quantized, request.what, request.angle*placementTau/16, auditOptions, builder, nullptr)));
        }
        out.put(audit, static_cast<int>(TheBuildAssistant->isLocationLegalToBuild(
            &request.original, request.what, request.angle*placementTau/16, auditOptions, builder, nullptr)));
        grid.finish(); audit.finish();
    }
    grids.finish();
}

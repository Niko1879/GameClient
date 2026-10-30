#pragma once
#include <string>

class AsciiString;
class Object;
class AIGroup;
class GameMessage;

// Configuration is separate from simulation GlobalData: it must not enter game CRCs.
void TrainingExporterConfigure(const char *directory);
void TrainingExporterSetInterval(Int milliseconds);
void TrainingExporterSetPlacementMode(Bool fullMask);
void TrainingExporterSetBinaryFormat(Bool binary);
void TrainingExporterSetScheduledQuietCapture(Bool scheduled);
void TrainingExporterBuildResult(const GameMessage *message, const Object *building);
Bool TrainingExporterIsEnabled();
Bool TrainingExporterBegin(const AsciiString &replay);
void TrainingExporterCapturePreCommands();
Bool TrainingExporterFinish(Bool playbackSucceeded, Bool crcMismatch);
// Optional performance counters, outside exported observations and game state.
enum TrainingPlacementProfilePart { TP_OVERLAP, TP_PATH, TP_TERRAIN };
Bool TrainingExportProfilingEnabled();
void TrainingExportProfilePlacement(TrainingPlacementProfilePart part, double seconds);
// training_commands::objectCommands: menu buttons (including object-target
// scans), object-target scans alone, and the enter/repair/depot target pools.
enum TrainingCommandsProfilePart { TC_BUTTONS, TC_OBJECT_TARGETS, TC_TARGETS };
void TrainingExportProfileCommands(TrainingCommandsProfilePart part, double seconds);

// The same encoder serves replay demonstrations and live self-play policies.
void TrainingObservationReset();
// Call before a live object's death/destroy transition changes visibility.
void TrainingObserveObjectRemoval(const Object *object);
// Observer-only command context, outside simulation state and CRCs.
void TrainingObserveCommand(const GameMessage *message, const AIGroup *selection);
void TrainingObserveValidation(Int playerIndex, const std::string &feedback);
std::string TrainingObservationJSON(Int playerIndex);
std::string TrainingMapJSON();
std::string TrainingTemplateCatalogJSON();
Bool TrainingHasAttackMoveCommand(Object *object);
Bool TrainingCanAttack(Object *object);
// Explicit viewer, never the local/observer player; shared by all policy APIs.
Bool TrainingObjectVisibleTo(const Object *object, Int playerIndex);
Bool TrainingCanDock(Object *actor, Object *target);
Bool TrainingCanResumeConstruction(Object *actor, Object *target);
std::string TrainingObservationPolicyJSON();
// Read-only, player-information-safe placement preview, at exact grid centres.
std::string TrainingBuildPlacementJSON(Int playerIndex, unsigned builder, const std::string &what,
                                     Int angleIndex, Int bins);

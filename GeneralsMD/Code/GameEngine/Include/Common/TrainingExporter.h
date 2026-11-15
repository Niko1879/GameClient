// GeneralsAI training export for retail Zero Hour 1.04 replays (C++98, VC6-compatible).
// A minimal port of the Generals Online exporter: binary format only, fast placement
// mode, quiet rows either all observed or scheduled by the capture controller. The output is byte-compatible with the GOBS1
// files the Python pipeline reads. Export never changes simulation state or CRCs.
#pragma once

class AsciiString;
class Object;
class AIGroup;
class GameMessage;

// Configuration is separate from simulation GlobalData: it must not enter game CRCs.
void TrainingExporterConfigure(const char *directory);
void TrainingExporterSetInterval(Int milliseconds);
// Scheduled: a capture controller on stdin/stdout names the quiet rows that need
// an observation (GameClient scripts/scheduled_capture.py).
void TrainingExporterSetScheduledQuietCapture(Bool scheduled);
Bool TrainingExporterIsEnabled();
Bool TrainingExporterBegin(const AsciiString &replay);
// Observation boundary: after the recorder update, before this tick's command batch.
void TrainingExporterCapturePreCommands();
Bool TrainingExporterFinish(Bool playbackSucceeded, Bool crcMismatch);
void TrainingExporterBuildResult(const GameMessage *message, const Object *building);

// Call before a live object's death/destroy transition changes visibility.
void TrainingObserveObjectRemoval(const Object *object);
// Observer-only command context, outside simulation state and CRCs.
void TrainingObserveCommand(const GameMessage *message, const AIGroup *selection);
// Observer-only record of a special power's launch sounds (SpecialPowerModule::aboutToDoSpecialPower),
// for the audio cues each player hears. Never affects simulation state.
class SpecialPowerTemplate;
void TrainingObserveSpecialPower(const Object *source, const SpecialPowerTemplate *power, const Coord3D *target);

// Explicit viewer, never the local/observer player.
Bool TrainingObjectVisibleTo(const Object *object, Int playerIndex);
Bool TrainingCanAttack(Object *object);

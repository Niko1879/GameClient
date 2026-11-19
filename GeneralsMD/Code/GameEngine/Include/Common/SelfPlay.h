#pragma once

void SelfPlayConfigure();
Bool SelfPlayIsEnabled();
Int SelfPlayRun();
void SelfPlayUpdate();
Bool SelfPlayHasSkirmishActor();
Bool SelfPlayIsSkirmishActorActive();
Bool SelfPlayWaitingForActor();
Bool SelfPlayStartSkirmish(Int modelSlot, Int opponentSlot);
void SelfPlayShutdown();
// Runs at the same observation boundary as TrainingExporterCapturePreCommands.
void SelfPlayPreCommands(Bool endOnly = FALSE);
// Opt-in per episode (reset "record_replay"): writes a standard replay of the episode.
void SelfPlayRecordCommands();

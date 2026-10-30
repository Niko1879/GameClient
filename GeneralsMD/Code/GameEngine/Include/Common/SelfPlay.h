#pragma once

void SelfPlayConfigure();
Bool SelfPlayIsEnabled();
Int SelfPlayRun();
// Runs at the same observation boundary as TrainingExporterCapturePreCommands.
void SelfPlayPreCommands();
// Opt-in per episode (reset "record_replay"): writes a standard replay of the episode.
void SelfPlayRecordCommands();

/*
**	Command & Conquer Generals Zero Hour(tm)
**	Copyright 2025 TheSuperHackers
**
**	This program is free software: you can redistribute it and/or modify
**	it under the terms of the GNU General Public License as published by
**	the Free Software Foundation, either version 3 of the License, or
**	(at your option) any later version.
**
**	This program is distributed in the hope that it will be useful,
**	but WITHOUT ANY WARRANTY; without even the implied warranty of
**	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**	GNU General Public License for more details.
**
**	You should have received a copy of the GNU General Public License
**	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "PreRTS.h"	// This must go first in EVERY cpp file in the GameEngine

#include "Common/ReplaySimulation.h"

#include "Common/GameEngine.h"
#include "Common/LocalFileSystem.h"
#include "Common/Recorder.h"
#if defined(RTS_ZEROHOUR) && RTS_ZEROHOUR
#include "Common/TrainingExporter.h"
#include "Common/Player.h"
#include "Common/PlayerList.h"
#include "GameClient/MapUtil.h"
#include "GameLogic/VictoryConditions.h"
#include "GameNetwork/GameInfo.h"
#define GENERALS_AI_TRAINING_EXPORT 1
#endif
#include "Common/WorkerProcess.h"
#include "GameLogic/GameLogic.h"
#include "GameClient/GameClient.h"


Bool ReplaySimulation::s_isRunning = false;
UnsignedInt ReplaySimulation::s_replayIndex = 0;
UnsignedInt ReplaySimulation::s_replayCount = 0;

namespace
{
int countProcessesRunning(const std::vector<WorkerProcess>& processes)
{
	int numProcessesRunning = 0;
	size_t i = 0;
	for (; i < processes.size(); ++i)
	{
		if (processes[i].isRunning())
			++numProcessesRunning;
	}
	return numProcessesRunning;
}

#ifdef GENERALS_AI_TRAINING_EXPORT
// GeneralsAI: who won, from the engine's multiplayer victory conditions, read the way
// GameSpyGameInfo::generateGameResultsPacket reports results. Keyed by the header's slot.
// victorious/eliminated are only set once a single alliance remains (end_frame > 0);
// defeated_now is the live check (no victory-counting buildings) at the last frame.
void printReplayOutcome()
{
	printf("ReplayOutcome {\"end_frame\":%u,\"players\":[", TheVictoryConditions->getEndFrame());
	Bool first = TRUE;
	for (Int i = 0; i < MAX_SLOTS; ++i)
	{
		Player *p = ThePlayerList->getPlayerFromSlotIndex(i);
		if (!p || p->isPlayerObserver())
			continue;
		GameSlot *slot = TheGameInfo ? TheGameInfo->getSlot(i) : nullptr;
		printf("%s{\"slot\":%d,\"player_index\":%d,\"victorious\":%s,\"eliminated\":%s,\"defeated_now\":%s,"
			"\"has_objects\":%s,\"disconnected\":%s,\"last_frame_in_game\":%d}",
			first ? "" : ",", i, p->getPlayerIndex(),
			TheVictoryConditions->hasAchievedVictory(p) ? "true" : "false",
			TheVictoryConditions->hasBeenDefeated(p) ? "true" : "false",
			TheVictoryConditions->hasSinglePlayerBeenDefeated(p) ? "true" : "false",
			p->hasAnyObjects() ? "true" : "false",
			(slot && slot->disconnected()) ? "true" : "false",
			slot ? (Int)slot->lastFrameInGame() : -1);
		first = FALSE;
	}
	printf("]}\n");
	fflush(stdout);
}
#endif
} // namespace

int ReplaySimulation::simulateReplaysInThisProcess(const std::vector<AsciiString> &filenames)
{
	int numErrors = 0;

	if (!TheGlobalData->m_headless)
	{
		s_isRunning = true;
		s_replayIndex = 0;
		s_replayCount = static_cast<UnsignedInt>(filenames.size());

		// If we are not in headless mode, we need to run the replay in the engine.
		for (; s_replayIndex < s_replayCount; ++s_replayIndex)
		{
			TheRecorder->playbackFile(filenames[s_replayIndex]);
			TheGameEngine->execute();
			if (TheRecorder->sawCRCMismatch())
				numErrors++;
			if (!s_isRunning)
				break;
			TheGameEngine->setQuitting(FALSE);
		}
		s_isRunning = false;
		s_replayIndex = 0;
		s_replayCount = 0;
		return numErrors != 0 ? 1 : 0;
	}
	// Note that we use printf here because this is run from cmd.
	DWORD totalStartTimeMillis = GetTickCount();
	for (size_t i = 0; i < filenames.size(); i++)
	{
		const AsciiString &filename = filenames[i];
		printf("Simulating Replay \"%s\"\n", filename.str());
		fflush(stdout);
		DWORD startTimeMillis = GetTickCount();
#ifdef GENERALS_AI_TRAINING_EXPORT
		if (!TrainingExporterBegin(filename)) { numErrors++; continue; }
#endif
		if (TheRecorder->simulateReplay(filename))
		{
			UnsignedInt totalTimeSec = TheRecorder->getPlaybackFrameCount() / LOGICFRAMES_PER_SECOND;
			while (TheRecorder->isPlaybackInProgress())
			{
				const int progressFrameInterval = 10*60*LOGICFRAMES_PER_SECOND;
				if (TheGameLogic->getFrame() != 0 && TheGameLogic->getFrame() % progressFrameInterval == 0)
				{
					// Print progress report
					UnsignedInt gameTimeSec = TheGameLogic->getFrame() / LOGICFRAMES_PER_SECOND;
					UnsignedInt realTimeSec = (GetTickCount()-startTimeMillis) / 1000;
					printf("Elapsed Time: %02d:%02d Game Time: %02d:%02d/%02d:%02d\n",
							realTimeSec/60, realTimeSec%60, gameTimeSec/60, gameTimeSec%60, totalTimeSec/60, totalTimeSec%60);
					fflush(stdout);
				}
				TheGameLogic->UPDATE();
				if (TheRecorder->sawCRCMismatch())
				{
					numErrors++;
					break;
				}
			}
			UnsignedInt gameTimeSec = TheGameLogic->getFrame() / LOGICFRAMES_PER_SECOND;
			UnsignedInt realTimeSec = (GetTickCount()-startTimeMillis) / 1000;
			printf("Elapsed Time: %02d:%02d Game Time: %02d:%02d/%02d:%02d\n",
					realTimeSec/60, realTimeSec%60, gameTimeSec/60, gameTimeSec%60, totalTimeSec/60, totalTimeSec%60);
			fflush(stdout);
#ifdef GENERALS_AI_TRAINING_EXPORT
			// GeneralsAI: playback evidence for compatibility probes and the exporter.
			const Bool completed = !TheRecorder->isPlaybackInProgress()
				&& TheRecorder->reachedPlaybackEOF()
				&& TheGameLogic->getFrame() >= TheRecorder->getLastPlaybackCommandFrame();
			const MapMetaData *map = TheMapCache ? TheMapCache->findMap(TheGlobalData->m_mapName) : nullptr;
			printf("ReplayValidation {\"completed\":%s,\"crc_mismatch\":%s,"
				"\"crc_comparisons\":%u,\"final_frame\":%u,\"last_command_frame\":%u,"
				"\"stream_eof\":%s,\"ticks_per_second\":%d,\"engine_ini_crc\":\"%08X\","
				"\"loaded_map_crc\":\"%08X\"}\n",
				completed ? "true" : "false", TheRecorder->sawCRCMismatch() ? "true" : "false",
				TheRecorder->getRecordedCRCComparisons(), TheGameLogic->getFrame(),
				TheRecorder->getLastPlaybackCommandFrame(), TheRecorder->reachedPlaybackEOF() ? "true" : "false",
				static_cast<int>(LOGICFRAMES_PER_SECOND), TheGlobalData->m_iniCRC, map ? map->m_CRC : 0);
			fflush(stdout);
			printReplayOutcome();
			if (!TrainingExporterFinish(completed, TheRecorder->sawCRCMismatch())) numErrors++;
#endif
		}
		else
		{
			printf("Cannot open replay\n");
#ifdef GENERALS_AI_TRAINING_EXPORT
			TrainingExporterFinish(FALSE, FALSE);
#endif
			numErrors++;
		}
	}
	if (filenames.size() > 1)
	{
		printf("Simulation of all replays completed. Errors occurred: %d\n", numErrors);

		UnsignedInt realTime = (GetTickCount()-totalStartTimeMillis) / 1000;
		printf("Total Time: %d:%02d:%02d\n", realTime/60/60, realTime/60%60, realTime%60);
		fflush(stdout);
	}

	return numErrors != 0 ? 1 : 0;
}

int ReplaySimulation::simulateReplaysInWorkerProcesses(const std::vector<AsciiString> &filenames, int maxProcesses)
{
#ifdef GENERALS_AI_TRAINING_EXPORT
	if (TrainingExporterIsEnabled())
	{
		printf("Training export requires sequential replay mode (omit -jobs).\n");
		return 1;
	}
#endif
	DWORD totalStartTimeMillis = GetTickCount();

	WideChar exePath[1024];
	GetModuleFileNameW(nullptr, exePath, ARRAY_SIZE(exePath));

	std::vector<WorkerProcess> processes;
	int filenamePositionStarted = 0;
	int filenamePositionDone = 0;
	int numErrors = 0;

	while (true)
	{
		int i;
		for (i = 0; i < processes.size(); i++)
			processes[i].update();

		// Get result of finished processes and print output in order
		while (!processes.empty())
		{
			if (!processes[0].isDone())
				break;
			AsciiString stdOutput = processes[0].getStdOutput();
			printf("%d/%d %s", filenamePositionDone+1, (int)filenames.size(), stdOutput.str());
			DWORD exitcode = processes[0].getExitCode();
			if (exitcode != 0)
				printf("Error!\n");
			fflush(stdout);
			numErrors += exitcode == 0 ? 0 : 1;
			processes.erase(processes.begin());
			filenamePositionDone++;
		}

		int numProcessesRunning = countProcessesRunning(processes);

		// Add new processes when we are below the limit and there are replays left
		while (numProcessesRunning < maxProcesses && filenamePositionStarted < filenames.size())
		{
			UnicodeString filenameWide;
			filenameWide.translate(filenames[filenamePositionStarted]);
			UnicodeString command;
			// Pass -useCwd, so that the worker keeps the working directory that it inherits from this process.
			command.format(L"\"%s\"%s%s -useCwd -replay \"%s\"",
				exePath,
				TheGlobalData->m_windowed ? L" -win" : L"",
				TheGlobalData->m_headless ? L" -headless" : L"",
				filenameWide.str());

			processes.push_back(WorkerProcess());
			processes.back().startProcess(command);

			filenamePositionStarted++;
			numProcessesRunning++;
		}

		if (processes.empty())
			break;

		// Don't waste CPU here, our workers need every bit of CPU time they can get
		Sleep(100);
	}

	DEBUG_ASSERTCRASH(filenamePositionStarted == filenames.size(), ("inconsistent file position 1"));
	DEBUG_ASSERTCRASH(filenamePositionDone == filenames.size(), ("inconsistent file position 2"));

	printf("Simulation of all replays completed. Errors occurred: %d\n", numErrors);

	UnsignedInt realTime = (GetTickCount()-totalStartTimeMillis) / 1000;
	printf("Total Wall Time: %d:%02d:%02d\n", realTime/60/60, realTime/60%60, realTime%60);
	fflush(stdout);

	return numErrors != 0 ? 1 : 0;
}

std::vector<AsciiString> ReplaySimulation::resolveFilenameWildcards(const std::vector<AsciiString> &filenames)
{
	// If some filename contains wildcards, search for actual filenames.
	// Note that we cannot do this in parseReplay because we require TheLocalFileSystem initialized.
	std::vector<AsciiString> filenamesResolved;
	for (std::vector<AsciiString>::const_iterator filename = filenames.begin(); filename != filenames.end(); ++filename)
	{
		if (filename->find('*') || filename->find('?'))
		{
			// An absolute path is searched as is, a relative path is searched in the replay directory.
			AsciiString dir1 = isAbsolutePath(filename->str()) ? AsciiString::TheEmptyString : TheRecorder->getReplayDir();
			AsciiString dir2 = *filename;
			AsciiString wildcard = *filename;
			{
				int len = dir2.getLength();
				while (len)
				{
					char c = dir2.getCharAt(len-1);
					if (isPathSeparator(c))
					{
						wildcard.set(wildcard.str()+dir2.getLength());
						break;
					}
					dir2.removeLastChar();
					len--;
				}
			}

			FilenameList files;
			TheLocalFileSystem->getFileListInDirectory(dir2.str(), dir1.str(), wildcard, files, FALSE);
			for (FilenameList::iterator it = files.begin(); it != files.end(); ++it)
			{
				AsciiString file;
				file.set(it->str() + dir1.getLength());
				filenamesResolved.push_back(file);
			}
		}
		else
			filenamesResolved.push_back(*filename);
	}
	return filenamesResolved;
}

int ReplaySimulation::simulateReplays(const std::vector<AsciiString> &filenames, int maxProcesses)
{
	std::vector<AsciiString> filenamesResolved = resolveFilenameWildcards(filenames);
	if (maxProcesses == SIMULATE_REPLAYS_SEQUENTIAL)
		return simulateReplaysInThisProcess(filenamesResolved);
	else
		return simulateReplaysInWorkerProcesses(filenamesResolved, maxProcesses);
}

#pragma once
#include <cstring>
// Game logic of an earlier GeneralsOnline release, for recorded replay playback only.
// Replays recorded by a release that predates a simulation fix diverge unless played
// with that release's logic. Set by -replayLogic; GameLogic::startNewGame refuses
// anything but the current logic outside playback, so live and online play never use it.
enum ReplayLogic
{
	REPLAY_LOGIC_CURRENT,
	// 081326_QFE3 (exe CRC 3DF43954, commit c929fa068): before the 60Hz fixes for
	// supply-drop payouts (4f9968f11), low-acceleration movement (750ac90fa) and
	// projectile launch orientation (4e990f34d).
	REPLAY_LOGIC_081326,
	// 062026 (exe CRC 48D67663, retail 1.04 INI FEAAE3F3, the 20 Jun 2026 release): also before
	// the 28 Jul TheSuperHackers merge 5c82ffd54 and the 3 Aug community patch code (912f26fd1).
	// replayLogic062026() gates each simulation change since then: exact single-unit move
	// destinations (#2296), particle effects spending logic random values (#2742; headless playback
	// loads the particle templates for this), guard aggressor transitions (#2197), InstantDeath
	// order and scaffold weapons (#2990, #3002), sniping Stinger Sites (#2822), the to-target laser
	// (#2817), every-other debris name (#2597), Helix portable containment (#2868), 1-frame
	// deletion lifetimes (GO #533), poison kill XP and Battle Bus lethality (912f26fd1).
	// Must be chosen on the command line (-replayLogic): particle templates load at engine start.
	REPLAY_LOGIC_062026,
};
extern ReplayLogic TheReplayLogic;

inline bool replayLogicBefore60HzFixes() { return TheReplayLogic == REPLAY_LOGIC_081326 || TheReplayLogic == REPLAY_LOGIC_062026; }
inline bool replayLogic062026() { return TheReplayLogic == REPLAY_LOGIC_062026; }

// The release label used by -replayLogic and the exporter's job/header "logic" field.
inline const char *replayLogicName(ReplayLogic logic)
{
	return logic == REPLAY_LOGIC_081326 ? "081326" : logic == REPLAY_LOGIC_062026 ? "062026" : "current";
}

inline bool parseReplayLogicName(const char *name, ReplayLogic &logic)
{
	for (ReplayLogic candidate : {REPLAY_LOGIC_CURRENT, REPLAY_LOGIC_081326, REPLAY_LOGIC_062026})
		if (strcmp(name, replayLogicName(candidate)) == 0) { logic = candidate; return true; }
	return false;
}

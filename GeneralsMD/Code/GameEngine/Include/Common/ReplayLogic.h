#pragma once
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
};
extern ReplayLogic TheReplayLogic;

inline bool replayLogicBefore60HzFixes() { return TheReplayLogic == REPLAY_LOGIC_081326; }

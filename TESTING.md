# Test Replays

The GeneralsReplays folder contains replays and the required maps that are tested in CI to ensure that the game is retail compatible.

You can also test with these replays locally:
- Copy the replays into a subfolder in your `%USERPROFILE%/Documents/Command and Conquer Generals Zero Hour Data/Replays` folder.
- Copy the maps into `%USERPROFILE%/Documents/Command and Conquer Generals Zero Hour Data/Maps`
- Start the test with this: (copy into a .bat file next to your executable)
```
START /B /W generalszh.exe -jobs 4 -headless -replay subfolder/*.rep > replay_check.log
echo %errorlevel%
PAUSE
```
It will run the game in the background and check that each replay is compatible. You need to use a VC6 build with optimizations and RTS_BUILD_OPTION_DEBUG = OFF, otherwise the game won't be compatible.

## Windowed Neural AI

Build the Zero Hour target locally and run the launcher from the repository root.
The default launcher uses the v11-1500 checkpoint, rating 1900, Tournament Desert,
and the existing Windows ROCm environment.

```bat
cmake --build build/win32-vcpkg --config Release --target z_generals
play_vs_model.cmd
play_vs_model.cmd --watch --faction FactionAmerica
play_vs_model.cmd --skirmish
```

Use `--map` for a custom map. In the Skirmish menu, choose **Neural AI** in an
opponent row. For an AI-versus-AI match, choose **Observer** in the local faction
dropdown and add a second opponent. The current Neural AI checkpoint plays
vanilla China in the Skirmish menu and supports one opponent.

Direct matches write their summary, replay, and profile under
`.cache/headed-play`. Use `--trajectory` to also save the full observation and
action log. Open the recording with
`play_vs_model.cmd --replay <session-folder>` and select **episode** in the replay
menu. The launcher uses the same executable, map, and game-data profile as the
recording. In particular, the profile's Community Patch INI package determines
the template IDs used in replayed build and production commands.

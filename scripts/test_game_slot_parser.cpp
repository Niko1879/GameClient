// Standalone regression checks for the native replay/network slot grammar.
#include "GameNetwork/GameSlotParser.h"
#include <cassert>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
    const std::string suffix = ",C0A80001,1234,TF,-1,3,0,-1,1:";
    const std::string other = "HOther,0,0,TT,1,4,1,-1,0:";
    std::array<GameSlotParser::Slot, 8> slots;
    for (const std::string name : {"hussein:)", "3:2", "m1nus :(", ":Ghost`",
                                  "name,with,commas", "Andy:stream,live;S=still=name", ""})
    {
        assert(GameSlotParser::split("H" + name + suffix + other + "X:X:O:X:X:X:", slots));
        assert(slots[0].text.substr(1, slots[0].nameEnd - 1) == name);
        assert(slots[1].text == other.substr(0, other.size() - 1));
        assert(slots[4].text == "O");
    }
    assert(GameSlotParser::split("CE,-1,3,0,-1:CM,0,4,1,-1:CH,1,2,2,0:X:O:X:X:X:", slots));
    assert(GameSlotParser::split("Hobserver:(,0,0,TT,-1,-2,-1,-1,1:" + other + "X:X:X:X:X:X:", slots));
    std::string repeated;
    for (int i = 0; i < 100; ++i) repeated += other;
    assert(!GameSlotParser::split(repeated + "invalid", slots));
    for (const std::string bad : {"X:X:X:X:X:X:X:", "X:X:X:X:X:X:X:X:X:",
                                  "X:X:X:X:X:X:X:X:garbage", "Hbad,0,0,ZZ,-1,3,0,-1,1:X:X:X:X:X:X:X:",
                                  "Hbad,0,0,TT,-1,3x,0,-1,1:X:X:X:X:X:X:X:",
                                  "CQ,-1,3,0,-1:X:X:X:X:X:X:X:"})
        assert(!GameSlotParser::split(bad, slots));
    std::cout << "Native slot grammar regression checks passed\n";
    if (argc == 2)
    {
        std::ifstream input(argv[1]);
        assert(input.good());
        std::string line;
        unsigned count = 0;
        while (std::getline(input, line))
        {
            const auto tab = line.find('\t');
            assert(tab != std::string::npos);
            std::string text;
            for (std::size_t i = tab + 1; i + 1 < line.size(); i += 2)
                text += static_cast<char>(std::stoul(line.substr(i, 2), nullptr, 16));
            if (!GameSlotParser::split(text, slots))
            {
                std::cerr << "Rejected real header " << line.substr(0, tab) << '\n';
                return 1;
            }
            ++count;
        }
        std::cout << "Parsed " << count << " real replay slot lists\n";
    }
}

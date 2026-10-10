// Action-portfolio contract cases (bots/p_only/tools/derive/search_experiment.py, PONLY_PORTFOLIO_K): the
// generated ctfcombat::choosePortfolio on small hand-built situations, plus the action serialization of both
// carriers. Prints one line per case; bots/p_only/tests/test_search_portfolio.py checks them.
//
//   CASE <name> <movements> <assignments>          then one line per kept assignment:
//   ALT <m0> <m1> <m2> <t0> <t1> <t2> <value>
//   TEXT <carrier> <target> <carrying> <action>    the protocol action a target code serializes to
//
//   g++ -std=c++20 -O2 -Iinference -Ibots/p_only/include -Ibots/p_only/bots -DFUSION_TIER_XXL -DPONLY_FAMILIES \
//       -DCARRIER_C08 -DPONLY_PORTFOLIO_K=3 bots/p_only/tests/portfolio_test.cpp
#define PONLY_NO_MAIN 1
#if defined(CARRIER_C08)
#include "c08_ponly_f.hpp"
namespace cc = multi::ctfcombat;
#define CARRIER_NAME "C08"
#elif defined(CARRIER_V12)
#include "v12_ponly_f.hpp"
namespace cc = ::ctfcombat;
#define CARRIER_NAME "V12"
#else
#error "define CARRIER_C08 or CARRIER_V12"
#endif

#include <cstdio>

namespace {

std::vector<std::string> openMap(int n) {
    std::vector<std::string> m(n, std::string(n, '.'));
    for (int i = 0; i < n; i++) m[0][i] = m[n - 1][i] = m[i][0] = m[i][n - 1] = '#';
    return m;
}

cc::Fighter unit(int id, int team, int x, int y, int hp = 100, int flag = -1) {
    cc::Fighter f;
    f.id = id; f.team = team; f.p = {x, y}; f.hp = hp; f.flag = flag;
    return f;
}

// An enemy that stays where it is with probability 1 and does not fire.
cc::Enemy still(int id, int x, int y, int hp, int flag = -1) {
    cc::Enemy e;
    e.u = unit(id, id / 3, x, y, hp, flag);
    cc::Outcome o;
    o.p = {x, y}; o.prob = 1.0; o.fire = 0.0;
    e.forecast.push_back(o);
    return e;
}

// The unit's only option: stay (with an optional pickup value).
std::vector<cc::Move> stay(const cc::Fighter& f, double pickup = 0) {
    cc::Move m;
    m.p = f.p; m.code = 'S'; m.score = 0; m.pickupValue = pickup;
    return {m};
}

void run(const char* name, const std::array<cc::Fighter, 3>& own, const std::array<std::vector<cc::Move>, 3>& options,
         const std::vector<cc::Enemy>& enemies, int k) {
    const auto map = openMap(15);
    std::vector<std::array<int, 3>> shortlist{{0, 0, 0}};
    std::vector<cc::Choice> alts;
    cc::PortfolioStats ps;
    cc::choosePortfolio(map, own, options, enemies, cc::Params{}, shortlist, k, alts, ps);
    std::printf("CASE %s %lld %zu\n", name, ps.movements, alts.size());
    for (const auto& c : alts)
        std::printf("ALT %d %d %d %d %d %d %.6f\n", c.moveIndex[0], c.moveIndex[1], c.moveIndex[2], c.target[0], c.target[1], c.target[2], c.value);
}

}  // namespace

int main() {
    // Two targets in range of all three shooters: several complete assignments for the one movement.
    {
        std::array<cc::Fighter, 3> own{unit(0, 0, 5, 5), unit(1, 0, 6, 5), unit(2, 0, 7, 5)};
        std::array<std::vector<cc::Move>, 3> opt{stay(own[0]), stay(own[1]), stay(own[2])};
        run("targets", own, opt, {still(3, 6, 6, 34), still(4, 6, 4, 100)}, 3);
    }
    // One one-hit target in range of units 0 and 1 only: never both on it (the second shot adds nothing).
    {
        std::array<cc::Fighter, 3> own{unit(0, 0, 5, 5), unit(1, 0, 6, 5), unit(2, 0, 12, 12)};
        std::array<std::vector<cc::Move>, 3> opt{stay(own[0]), stay(own[1]), stay(own[2])};
        run("overkill", own, opt, {still(3, 5, 6, 34)}, 3);
    }
    // A full-health target needs three hits: the best assignment uses all three shooters.
    {
        std::array<cc::Fighter, 3> own{unit(0, 0, 5, 5), unit(1, 0, 6, 5), unit(2, 0, 7, 5)};
        std::array<std::vector<cc::Move>, 3> opt{stay(own[0]), stay(own[1]), stay(own[2])};
        run("threshold", own, opt, {still(3, 6, 6, 100)}, 3);
    }
    // No enemies, unit 0 stands on a flag: pick up, or explicitly wait.
    {
        std::array<cc::Fighter, 3> own{unit(0, 0, 5, 5), unit(1, 0, 6, 5), unit(2, 0, 7, 5)};
        std::array<std::vector<cc::Move>, 3> opt{stay(own[0], 20.0), stay(own[1]), stay(own[2])};
        run("pickup", own, opt, {}, 3);
    }
    // A carrier never fires: its action stays idle (-1) in every assignment.
    {
        std::array<cc::Fighter, 3> own{unit(0, 0, 5, 5, 100, 0), unit(1, 0, 6, 5), unit(2, 0, 7, 5)};
        std::array<std::vector<cc::Move>, 3> opt{stay(own[0]), stay(own[1]), stay(own[2])};
        run("carrier", own, opt, {still(3, 6, 6, 34), still(4, 6, 4, 100)}, 3);
    }
    // Serialization of every target code, unladen and carrying.
    for (int t : {5, -1, -2, -3})
        for (int carrying = 0; carrying < 2; carrying++) {
#if defined(CARRIER_C08)
            const std::string text = multi::Planner::Action::fromTarget(t).text();  // C08 serializes without the carrier bit
#else
            const std::string text = ponlyActionText(t, carrying);
#endif
            std::printf("TEXT %s %d %d %s\n", CARRIER_NAME, t, carrying, text.c_str());
        }
    return 0;
}

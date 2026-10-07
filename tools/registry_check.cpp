// Prints which exported model (and alpha) the unified dispatch picks for every
// team count; built against inference/models/registry.hpp by the tests.
#include <cstdio>

#include "registry.hpp"

int main() {
    for (int t = 2; t <= 15; t++) {
        const fusion::models::ModelSet* m = fusion::models::forTeams(t);
        std::printf("%d %s %.6f\n", t, m ? m->id : "-", m ? m->alpha : 0.0);
    }
    return 0;
}

#include "mcf_remote.h"
#include "../runtime/utils.h"
#include <thread>
int counter = 0;
std::vector<remote_result> futures;
Channel<SharedDataMCF, 16> data_to;
Channel<ResultDataMCF, 16> data_back;
extern int bea_is_dual_infeasible(arc_t *arc, cost_t red_cost);
void remote1(arc_t *arc, long *basket_size, BASKET *perm[]) {
    /* red_cost = bea_compute_red_cost( arc ); */
    cost_t red_cost;
    if (arc->ident > 0) {
        /* red_cost = bea_compute_red_cost( arc ); */
        red_cost = arc->cost - arc->tail->potential + arc->head->potential;
        if (bea_is_dual_infeasible(arc, red_cost)) {
            (*basket_size)++;
            perm[*basket_size]->a = arc;
            perm[*basket_size]->cost = red_cost;
            perm[*basket_size]->abs_cost = ABS(red_cost);
        }
    }
}
remote_result remote_async(arc_t *arc, long *basket_size, BASKET **perm);
void remote(arc_t *arc, long *basket_size, BASKET *perm[]) {
    counter++;

    futures.push_back(remote_async(arc, basket_size, perm));
    if (counter % 100 == 0) {
        for (auto &result : futures) {
            while (!result.handle.done()) {
                result.handle.resume();
                std::this_thread::yield();
            }
        }
        futures.clear();
    }
}
remote_result remote_async(arc_t *arc, long *basket_size, BASKET **perm) {
    while (true) {
        SharedDataMCF data = {arc, basket_size, perm};
        while (!data_to.send(data)) {
            co_await std::suspend_always{};
        };
        ResultDataMCF back;
        while (!data_back.receive(back)) {
            co_await std::suspend_always{};
        };
        co_return back.i;
    }
}
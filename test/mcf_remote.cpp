#include "mcf_remote.h"
#include "../runtime/utils.h"
#include <thread>
int bea_is_dual_infeasible(arc_t *arc, cost_t red_cost) {
    return ((red_cost < 0 && arc->ident == 1) || (red_cost > 0 && arc->ident == 2));
};
void remote1(arc_t *arc, long *basket_size, BASKET *perm[]) {
    /* red_cost = bea_compute_red_cost( arc ); */
    cost_t red_cost;
    printf("arc: %p %p\n", arc, basket_size);
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

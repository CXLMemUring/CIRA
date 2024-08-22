typedef long cost_t;
typedef struct arc arc_t;
typedef struct node *node_p;
typedef struct arc *arc_p;
typedef long flow_t;

struct arc {
    cost_t cost;
    node_p tail, head;
    int ident;
    arc_p nextout, nextin;
    flow_t flow;
    cost_t org_cost;
};

struct node {
    cost_t potential;
    int orientation;
    node_p child;
    node_p pred;
    node_p sibling;
    node_p sibling_prev;
    arc_p basic_arc;
    arc_p firstout, firstin;
    arc_p arc_tmp;
    flow_t flow;
    long depth;
    int number;
    int time;
};
typedef struct basket {
    arc_t *a;
    cost_t cost;
    cost_t abs_cost;
} BASKET;
#define ABS(x) (((x) >= 0) ? (x) : -(x))

int bea_is_dual_infeasible(arc_t *arc, cost_t red_cost) {
    return ((red_cost < 0 && arc->ident == 1) || (red_cost > 0 && arc->ident == 2));
}

cost_t remote(arc_t *arc, long *basket_size, basket *perm[]) {
    /* red_cost = bea_compute_red_cost( arc ); */
    cost_t red_cost = 0;
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
    return red_cost;
}
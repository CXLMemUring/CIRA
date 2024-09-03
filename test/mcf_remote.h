#ifndef MCF_REMOTE_H
#define MCF_REMOTE_H
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
extern "C" {
int main_ptr();
}
#endif
#ifndef BAYESIAN_OBJ
#define BAYESIAN_OBJ 

#include <math.h>
#include "../common.h"
#include <vector>
#define MAX_BAYESIAN_OBJECTS 50
#define BAYESIAN_HISTORY 5
#define BAYESIAN_DISTANCE_THRESH 50

typedef struct bayesian
{
    int used[BAYESIAN_HISTORY];
    int x[BAYESIAN_HISTORY], y[BAYESIAN_HISTORY];
    //int dist;
} bayesian_t;

void append_old_objs(std::vector<OBJ> obj_list);
void display_history();

#endif
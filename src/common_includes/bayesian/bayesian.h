#ifndef BAYESIAN_OBJ
#define BAYESIAN_OBJ 

#include <math.h>
#include <cmath> 
#include "../common.h"
#include <vector>

#define MAX_BAYESIAN_OBJECTS 10
#define BAYESIAN_HISTORY 10
#define BAYESIAN_DISTANCE_THRESH 100

typedef struct bayesian{
    int predX = 0, predY = 0;
    int used[BAYESIAN_HISTORY];
    int x[BAYESIAN_HISTORY];
    int y[BAYESIAN_HISTORY];
    //int dist;
} bayesian_t;

void append_old_objs(std::vector<OBJ> obj_list);
void display_history();
void predict(int id, int* x, int* y);
std::vector<OBJ> get_predicted_boxes();
#endif
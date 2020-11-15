#include "bayesian.h"
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
using namespace std;

int OLD_OBJS_TOP = 0;
bayesian_t OLD_BAYES_OBJS[MAX_BAYESIAN_OBJECTS];
int QUEUE_IS_EMPTY = 1, QUEUE_IS_FULL = 0;

double distance(int x1, int y1, int x2, int y2) {
    return sqrt( pow(x1-x2, 2) + pow(y1-y2, 2) );
}

int unused_id(int recent) {
    for (int iCount=0; iCount<MAX_BAYESIAN_OBJECTS; iCount++) {
        if (OLD_BAYES_OBJS[iCount].used[recent] == 0)
            return OLD_BAYES_OBJS[iCount].used[recent];
    }
    return 0; // All IDS used up
}

int match_object(int x, int y) {
    int id = 0;
    double old_dist = BAYESIAN_DISTANCE_THRESH;
    int prev = (OLD_OBJS_TOP-1) % BAYESIAN_HISTORY;

        
    for (int jCount=0; jCount<MAX_BAYESIAN_OBJECTS; jCount++) {
        if (OLD_BAYES_OBJS[jCount].used[prev] != 0) {
            int dist = distance(OLD_BAYES_OBJS[jCount].x[prev], OLD_BAYES_OBJS[jCount].y[prev], x, y);
            if (
                dist < BAYESIAN_DISTANCE_THRESH &&
                dist < old_dist
                ) {
                id = jCount;
                old_dist = dist;
            }
        } 
    }

    if (id == 0) {
        id = unused_id(prev);
    }

    return id;
}

void display_history() {
    for (int i=0; i<MAX_BAYESIAN_OBJECTS; i++) {
        printf("ID=%d\n", i);
        for (int j=0; j<BAYESIAN_HISTORY; j++) {
            if (OLD_BAYES_OBJS[i].used[j]) {
                printf("\t(%d, %d) ", OLD_BAYES_OBJS[i].x[j], OLD_BAYES_OBJS[i].y[j]);
            }
        }
        printf("\n------------\n");
    }
}

// CLL
void append_old_objs(std::vector<OBJ> obj_list) {
    int top = OLD_OBJS_TOP % BAYESIAN_HISTORY;
    int iCount=0;

    for (int jCount=0; jCount<MAX_BAYESIAN_OBJECTS; jCount++) {
        OLD_BAYES_OBJS[jCount].used[top] = 0;
    }

    for (OBJ object : obj_list) {
        int id = iCount;
        if (QUEUE_IS_EMPTY)
            id = match_object(object.x, object.y);
        
        OLD_BAYES_OBJS[id].used[top] = 1;
        OLD_BAYES_OBJS[id].x[top] = object.x;
        OLD_BAYES_OBJS[id].y[top] = object.y;

        iCount++;
        if (iCount>MAX_BAYESIAN_OBJECTS)
            break;
    }
    if (QUEUE_IS_EMPTY)
        QUEUE_IS_EMPTY = 0;
    
    if (top == BAYESIAN_HISTORY-1)
        QUEUE_IS_FULL = 1;
    OLD_OBJS_TOP = top+1;
}

int mean_change_position_vector(int* a, int* used) {
    double m = 0;
    int delta = 0;
    int recent = (OLD_OBJS_TOP-1) % BAYESIAN_HISTORY;
    int i = recent;
    for (int iCount=2; iCount<BAYESIAN_HISTORY; iCount++) {
        i = (recent + iCount) % BAYESIAN_HISTORY;
        if (used[i]) {
            delta = a[i] - a[i-1];
            if (abs(delta) < BAYESIAN_DISTANCE_THRESH) {
                //printf(" > ");
                m += delta;
            }
        }
        
        //printf("%d - %d = %d\n", a[i], a[i-1], delta);
    }
    //printf("AVG = %f\n", round(m / BAYESIAN_HISTORY));
    //printf("-----\n");
    return round(m / BAYESIAN_HISTORY);
}


void predict(int id, int* x, int* y) {
    if (!QUEUE_IS_FULL)
        return;
    int recent = (OLD_OBJS_TOP-1) % BAYESIAN_HISTORY;
    //printf("ID=%d, top=%d\n", id, recent);
    *x = OLD_BAYES_OBJS[id].x[recent] + mean_change_position_vector(OLD_BAYES_OBJS[id].x, OLD_BAYES_OBJS[id].used);
    *y = OLD_BAYES_OBJS[id].y[recent] + mean_change_position_vector(OLD_BAYES_OBJS[id].y, OLD_BAYES_OBJS[id].used);
}

std::vector<OBJ> get_predicted_boxes() {
    int recent = (OLD_OBJS_TOP-1) % BAYESIAN_HISTORY;
    std::vector<OBJ> plist;
    for (int iCount=0; iCount<MAX_BAYESIAN_OBJECTS; iCount++) {
        if (OLD_BAYES_OBJS[iCount].used[recent] == 1) {
            OBJ temp;
            temp.name = "P";//format("P_%02d", iCount).c_str();
            predict(iCount, &temp.x, &temp.y);
            temp.w = 10;
            temp.h = 10;
            temp.c = 0.1;
            temp.g = 1; //0   / 255.0;
            temp.b = 1; //211 / 255.0;
            temp.r = 1; //148 /255.0;
            plist.push_back(temp);
        }
    }
    return plist;
}
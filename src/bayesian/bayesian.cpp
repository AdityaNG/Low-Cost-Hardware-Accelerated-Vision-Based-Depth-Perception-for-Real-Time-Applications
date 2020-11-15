#include "bayesian.h"

int OLD_OBJS_TOP = 0;
bayesian_t OLD_BAYES_OBJS[MAX_BAYESIAN_OBJECTS];
int QUEUE_IS_EMPTY = 1;

double distance(int x1, int y1, int x2, int y2) {
    return sqrt( pow(x1-x2, 2) + pow(y1-y2, 2) );
}

int unused_id(int recent) {
    int USED_IDS[MAX_BAYESIAN_OBJECTS];
    for (int iCount=0; iCount<MAX_BAYESIAN_OBJECTS; iCount++) {
        if (OLD_BAYES_OBJS[iCount].used[recent] == 0)
            return OLD_BAYES_OBJS[iCount].used[recent];
    }
    return 0; // All IDS used up
}

int match_object(int x, int y) {
    int id = 0;
    double old_dist = BAYESIAN_DISTANCE_THRESH;
    int prev = OLD_OBJS_TOP % BAYESIAN_HISTORY - 1;

        
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
    OLD_OBJS_TOP = top+1;
}

void predict(int frames_ahead) {
    int recent = OLD_OBJS_TOP % BAYESIAN_HISTORY - 1;
}
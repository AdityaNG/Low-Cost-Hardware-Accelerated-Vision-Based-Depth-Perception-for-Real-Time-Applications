#ifndef COMMON_OBJ
#define COMMON_OBJ
#include <iostream>
typedef struct object {
    std::string name;  // Name of the detection
    int x, y;          // Coordinates
    int w, h;          // Width and height
    float c;           // Confidence
    double r, g, b;
} OBJ;
#endif

#ifdef SPECIAL_STRUCTS
typedef struct {
    double x, y, z;
} Double3;

typedef struct {
    unsigned char x, y, z, w;
} Uchar4;
#endif
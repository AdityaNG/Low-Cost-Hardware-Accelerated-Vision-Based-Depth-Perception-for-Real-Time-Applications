#ifndef COMMON_OBJ
#define COMMON_OBJ 
#include <iostream>
#include <stdio.h>

typedef struct object {
  std::string name; // Name of the detection
  int x, y; // Coordinates
  int w, h; // Width and height
  float c; // Confidence
  double r, g, b;
} OBJ;

#endif

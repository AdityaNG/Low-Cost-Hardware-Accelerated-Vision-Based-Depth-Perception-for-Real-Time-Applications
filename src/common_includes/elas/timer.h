/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
#include <stdint.h>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif

class Timer {
   public:
    Timer() {}

    ~Timer() {}

    void start(std::string title);

    void stop();

    void plotCpp();

    void plot();

    void reset();

   private:
    std::vector<std::string> desc;
    std::vector<timeval> time;

    void push_back_time() {
        timeval curr_time;
        gettimeofday(&curr_time, 0);
        time.push_back(curr_time);
    }

    float getTimeDifferenceMilliseconds(timeval a, timeval b) {
        return ((float)(b.tv_sec - a.tv_sec)) * 1e+3 + ((float)(b.tv_usec - a.tv_usec)) * 1e-3;
    }
};

#endif
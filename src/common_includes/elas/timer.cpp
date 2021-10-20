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

#include "timer.h"
#define YELLOW "\033[93m"
#define RESET "\033[0m"
#define GREEN "\033[92m"

void Timer::start(std::string title) {
    desc.push_back(title);
    push_back_time();
}

void Timer::stop() {
    if (time.size() <= desc.size())
        push_back_time();
}

void Timer::plotCpp() {
    stop();
    float total_time = 0;
    for (int32_t i = 0; i < desc.size(); i++) {
        float curr_time = getTimeDifferenceMilliseconds(time[i], time[i + 1]);
        total_time += curr_time;
        std::cout.width(30);
        std::cout << desc[i] << " ";
        std::cout << std::fixed << std::setprecision(1) << std::setw(6);
        std::cout << curr_time;
        std::cout << " ms" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    std::cout << "                    Total time ";
    std::cout << std::fixed << std::setprecision(1) << std::setw(6);
    std::cout << total_time;
    std::cout << " ms" << std::endl << std::endl;
}

void Timer::plot() {
    stop();
    float total_time = 0;
    printf("\n%s%25s%s\n", YELLOW, "Pre Processing:", RESET);
    for (int32_t i = 0; i < desc.size(); i++) {
        float curr_time = getTimeDifferenceMilliseconds(time[i], time[i + 1]);
        total_time += curr_time;
        printf("%30s %.2lf ms\n", desc[i].c_str(), curr_time);
        if (desc[i] == "Descriptor")
            printf("%s%25s%s\n", YELLOW, "Disparity Calculation:", RESET);
        else if (desc[i] == "Matching")
            printf("%s%25s%s\n", YELLOW, "Post Processing:", RESET);
    }
    printf("========================================\n");
    printf("                    %sTotal time %.2lf ms%s\n", GREEN, total_time, RESET);
    reset();
}

void Timer::reset() {
    desc.clear();
    time.clear();
}
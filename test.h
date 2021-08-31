//
// Created by zhn68 on 2021/8/31.
//

#ifndef OBJLEFT_TEST_H
#define OBJLEFT_TEST_H
#define num_threads 1000000
#define block_width 1000
#define array_size 10
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"

void incres();

#endif //OBJLEFT_TEST_H

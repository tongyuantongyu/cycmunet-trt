//
// Created by TYTY on 2021-12-24 024.
//

#ifndef REAL_ESRGAN_TRT__IMAGE_UTILS_H_
#define REAL_ESRGAN_TRT__IMAGE_UTILS_H_

#include <cstdint>

#if USE_FP32
typedef float pixel;
#elif USE_FP16
typedef uint16_t pixel;
#else
#error "Please define USE_FP32 or USE_FP16"
#endif

#include <memory>

void initImageTool();

std::unique_ptr<uint8_t[]> loadImage(const char *name, uint32_t &width, uint32_t &height);
void saveImage(const char *name, uint8_t *data, uint32_t width, uint32_t height);

#endif //REAL_ESRGAN_TRT__IMAGE_UTILS_H_

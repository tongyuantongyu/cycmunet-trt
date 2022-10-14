//
// Created by TYTY on 2021-12-24 024.
//

#include "image-utils.h"

#define NOMINMAX
#include <wincodec.h>
#include <windows.h>
#include <shlwapi.h>

#include <memory>
#include <iostream>
#include <cwchar>

#if USE_FP32
const auto pixelFormat = GUID_WICPixelFormat96bppRGBFloat;
#elif USE_FP16
const auto pixelFormat = GUID_WICPixelFormat48bppRGBHalf;
#endif

#define HR_CHECK(A) \
    do { \
        HRESULT hr = (A); \
        if (FAILED(hr)) { \
            std::cerr << "Syscall failed: " #A ", with code " << hr << std::endl; \
            exit(2); \
        } \
    } while (0)

IWICImagingFactory *pFactory;

void initImageTool() {
  HR_CHECK(CoInitialize(nullptr));
  HR_CHECK(CoCreateInstance(CLSID_WICImagingFactory,
                            nullptr,
                            CLSCTX_INPROC_SERVER,
                            IID_IWICImagingFactory,
                            (LPVOID *) (&pFactory)));
}

std::unique_ptr<uint8_t[]> loadImage(const char *name, uint32_t &width, uint32_t &height) {
  IWICBitmapDecoder *pDecoder = nullptr;
  IWICBitmapFrameDecode *pFrame = nullptr;
  IWICFormatConverter *pConverter = nullptr;

  HANDLE f = CreateFileA(name, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) {
    fprintf(stderr, "Can't open file for read: %s\n", name);
    exit(1);
  }

  HR_CHECK(pFactory->CreateDecoderFromFileHandle((ULONG_PTR) f, nullptr, WICDecodeMetadataCacheOnDemand, &pDecoder));
  HR_CHECK(pDecoder->GetFrame(0, &pFrame));

  HR_CHECK(pFrame->GetSize(&width, &height));

  auto pixels = std::make_unique<uint8_t[]>(width * height * 3);
  HR_CHECK(pFactory->CreateFormatConverter(&pConverter));
  HR_CHECK(pConverter->Initialize(pFrame,
                                  GUID_WICPixelFormat24bppBGR,
                                  WICBitmapDitherTypeNone,
                                  nullptr,
                                  0.0,
                                  WICBitmapPaletteTypeCustom));

  HR_CHECK(pConverter->CopyPixels(nullptr,
                                  width * 3 * sizeof(uint8_t),
                                  width * height * 3 * sizeof(uint8_t),
                                  reinterpret_cast<BYTE *>(pixels.get())));
  pConverter->Release();
  pFrame->Release();
  pDecoder->Release();

  return pixels;
}

void saveImage(const char *name, uint8_t *data, uint32_t width, uint32_t height) {
  IWICStream *pStream = nullptr;
  IWICBitmapEncoder *pEncoder = nullptr;
  IWICBitmapFrameEncode *pFrame = nullptr;
  IWICBitmap *pSource = nullptr;

  mbstate_t state;
  wchar_t name_temp[256];
  if (mbsrtowcs_s(nullptr, name_temp, &name, 256, &state) != 0) {
    std::cerr << "bad save filename." << std::endl;
    exit(2);
  }

  HR_CHECK(pFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder));
  HR_CHECK(pFactory->CreateStream(&pStream));
  HR_CHECK(pStream->InitializeFromFilename(name_temp, GENERIC_WRITE));
  HR_CHECK(pEncoder->Initialize(pStream, WICBitmapEncoderNoCache));
  HR_CHECK(pEncoder->CreateNewFrame(&pFrame, nullptr));
  auto format = GUID_WICPixelFormat24bppBGR;
  HR_CHECK(pFrame->Initialize(nullptr));
  HR_CHECK(pFrame->SetPixelFormat(&format));
  HR_CHECK(pFrame->SetSize(width, height));

  HR_CHECK(pFactory->CreateBitmapFromMemory(width,
                                            height,
                                            format,
                                            width * 3 * sizeof(uint8_t),
                                            width * height * 3 * sizeof(uint8_t),
                                            reinterpret_cast<BYTE *>(data),
                                            &pSource));

  HR_CHECK(pFrame->WriteSource(pSource, nullptr));

  HR_CHECK(pFrame->Commit());
  HR_CHECK(pFrame->Release());
  HR_CHECK(pSource->Release());
  HR_CHECK(pEncoder->Commit());
  HR_CHECK(pEncoder->Release());
  HR_CHECK(pStream->Release());
}
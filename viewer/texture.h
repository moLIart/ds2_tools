#pragma once
#include <memory>
#include <rhi/qrhi.h>

namespace ds2 {
struct UploadedTexture {
    std::unique_ptr<QRhiTexture> texture;
    int lastMip = 0;
};
UploadedTexture loadTexture(QRhi *rhi, QRhiResourceUpdateBatch *batch, const QString &path);
} // namespace ds2

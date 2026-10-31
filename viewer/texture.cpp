#include "texture.h"
#include <QFile>
#include <QImage>
#include <QtEndian>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ds2 {
namespace {
[[noreturn]] void bad(const QString &s) {
    throw std::runtime_error(s.toStdString());
}
uint32_t u32(const QByteArray &b, qsizetype p) {
    if (p < 0 || p + 4 > b.size())
        bad("Truncated texture");

    return qFromLittleEndian<uint32_t>(b.constData() + p);
}
uint16_t u16(const char *p) {
    return qFromLittleEndian<uint16_t>(p);
}
QImage tga(const QByteArray &b) {
    if (b.size() < 18)
        bad("Truncated TGA header");

    const auto *h = reinterpret_cast<const uint8_t *>(b.constData());
    const int type = h[2], w = u16(b.constData() + 12), height = u16(b.constData() + 14), bits = h[16];
    if (!w || !height || w > 16384 || height > 16384)
        bad("Invalid TGA size");

    if (type != 1 && type != 2 && type != 3 && type != 9 && type != 10 && type != 11)
        bad("Unsupported TGA image type");

    qsizetype pos = 18 + h[0];
    auto pixel = [&](int depth) -> QRgb {
        const int bytes = (depth + 7) / 8;
        if (pos + bytes > b.size())
            bad("Truncated TGA pixels");

        const auto *p = reinterpret_cast<const uint8_t *>(b.constData() + pos);
        pos += bytes;
        if (depth == 8)
            return qRgba(p[0], p[0], p[0], 255);

        if (depth == 15 || depth == 16) {
            const auto v = u16(reinterpret_cast<const char *>(p));
            return qRgba(((v >> 10) & 31) * 255 / 31, ((v >> 5) & 31) * 255 / 31, (v & 31) * 255 / 31,
                         depth == 16 && (h[17] & 15) ? ((v & 32768) ? 255 : 0) : 255);
        }

        if (depth == 24 || depth == 32)
            return qRgba(p[2], p[1], p[0], depth == 32 ? p[3] : 255);

        bad("Unsupported TGA pixel depth");
    };
    QVector<QRgb> palette;
    const int first = u16(b.constData() + 3), count = u16(b.constData() + 5);
    if (h[1]) {
        palette.resize(first + count);
        for (int i = 0; i < count; ++i)
            palette[first + i] = pixel(h[7]);
    }

    auto readPixel = [&]() {
        if (type == 1 || type == 9) {
            if (bits != 8 && bits != 16)
                bad("Unsupported TGA palette index");

            if (pos + bits / 8 > b.size())
                bad("Truncated TGA index");

            const auto idx = bits == 8 ? uint8_t(b[pos]) : u16(b.constData() + pos);
            pos += bits / 8;
            if (idx >= palette.size())
                bad("Invalid TGA palette index");

            return palette[idx];
        }

        if ((type == 3 || type == 11) && bits == 16) {
            if (pos + 2 > b.size())
                bad("Truncated TGA grayscale");

            const int v = uint8_t(b[pos++]), a = uint8_t(b[pos++]);
            return qRgba(v, v, v, a);
        }

        return pixel(bits);
    };
    QImage image(w, height, QImage::Format_RGBA8888);
    const bool top = h[17] & 32, right = h[17] & 16;
    auto store = [&](int i, QRgb v) {
        int x = i % w, y = i / w;
        if (right)
            x = w - 1 - x;

        if (!top)
            y = height - 1 - y;

        // pyglet uploaded TGA bottom-up, with a separate shader V flip.
        auto *p = image.scanLine(height - 1 - y) + x * 4;
        p[0] = qRed(v);
        p[1] = qGreen(v);
        p[2] = qBlue(v);
        p[3] = qAlpha(v);
    };
    int i = 0;
    while (i < w * height) {
        int n = 1;
        bool run = false;
        if (type >= 9) {
            if (pos >= b.size())
                bad("Truncated TGA RLE");

            const uint8_t v = b[pos++];
            n = (v & 127) + 1;
            run = v & 128;
        }

        if (n > w * height - i)
            bad("TGA packet exceeds image");

        if (run) {
            auto p = readPixel();
            while (n--)
                store(i++, p);
        } else
            while (n--)
                store(i++, readPixel());
    }

    return image;
}
std::array<uint8_t, 4> rgb565(uint16_t v) {
    return {uint8_t(((v >> 11) & 31) * 255 / 31), uint8_t(((v >> 5) & 63) * 255 / 63),
            uint8_t((v & 31) * 255 / 31), 255};
}
QImage decodeBC(const char *data, int w, int h, int format) {
    QImage result(w, h, QImage::Format_RGBA8888);
    for (int by = 0; by < h; by += 4)
        for (int bx = 0; bx < w; bx += 4) {
            const auto *p = reinterpret_cast<const uint8_t *>(data);
            data += format == 1 ? 8 : 16;
            uint8_t alpha[16];
            std::fill_n(alpha, 16, 255);
            if (format == 2) {
                for (int i = 0; i < 16; ++i)
                    alpha[i] = ((p[i / 2] >> (4 * (i % 2))) & 15) * 17;

                p += 8;
            }

            if (format == 3) {
                uint8_t a[8] = {p[0], p[1]};
                if (a[0] > a[1])
                    for (int i = 1; i <= 6; ++i)
                        a[i + 1] = ((7 - i) * a[0] + i * a[1]) / 7;
                else {
                    for (int i = 1; i <= 4; ++i)
                        a[i + 1] = ((5 - i) * a[0] + i * a[1]) / 5;

                    a[6] = 0;
                    a[7] = 255;
                }

                uint64_t indices = 0;
                for (int i = 0; i < 6; ++i)
                    indices |= uint64_t(p[i + 2]) << (i * 8);

                for (int i = 0; i < 16; ++i)
                    alpha[i] = a[(indices >> (i * 3)) & 7];

                p += 8;
            }

            const uint16_t a = u16(reinterpret_cast<const char *>(p)),
                           b = u16(reinterpret_cast<const char *>(p + 2));
            std::array<uint8_t, 4> c[4] = {rgb565(a), rgb565(b), {}, {}};
            for (int k = 0; k < 3; ++k) {
                c[2][k] = a > b || format != 1 ? (2 * c[0][k] + c[1][k]) / 3 : (c[0][k] + c[1][k]) / 2;
                c[3][k] = a > b || format != 1 ? (c[0][k] + 2 * c[1][k]) / 3 : 0;
            }

            c[2][3] = 255;
            c[3][3] = a > b || format != 1 ? 255 : 0;
            const uint32_t indices = qFromLittleEndian<uint32_t>(p + 4);
            for (int i = 0; i < 16; ++i)
                if (bx + i % 4 < w && by + i / 4 < h) {
                    auto v = c[(indices >> (2 * i)) & 3];
                    if (format != 1)
                        v[3] = alpha[i];

                    std::memcpy(result.scanLine(by + i / 4) + (bx + i % 4) * 4, v.data(), 4);
                }
        }

    return result;
}
struct DdsInfo {
    uint32_t width, height, mips;
    int bc;
};
DdsInfo ddsInfo(const QByteArray &bytes) {
    if (bytes.size() < 128 || bytes.first(4) != "DDS " || u32(bytes, 4) != 124)
        bad("Invalid DDS header");

    const uint32_t w = u32(bytes, 16), h = u32(bytes, 12), mips = std::max(1u, u32(bytes, 28));
    const uint32_t code = u32(bytes, 84);
    if (!w || !h || w > 16384 || h > 16384 || mips > 15)
        bad("Invalid DDS dimensions/mips");

    const int bc = code == 0x31545844 ? 1 : code == 0x33545844 ? 2 : code == 0x35545844 ? 3 : 0;
    if (!bc)
        bad("Unsupported DDS format (expected DXT1/DXT3/DXT5)");

    if (int(mips) > 1 + int(std::floor(std::log2(std::max(w, h)))))
        bad("DDS has more mip levels than its dimensions permit");

    qsizetype end = 128;
    for (uint32_t level = 0; level < mips; ++level)
        end += ((std::max(1u, w >> level) + 3) / 4) *
               ((std::max(1u, h >> level) + 3) / 4) * (bc == 1 ? 8 : 16);

    if (end > bytes.size())
        bad("Truncated DDS mip chain");

    return {w, h, mips, bc};
}
} // namespace
QImage readTextureImage(const QString &path) {
    if (path.startsWith('$')) {
        QImage image(1, 1, QImage::Format_RGBA8888);
        image.fill(path == "$white$" ? Qt::white : path == "$black$" ? Qt::black : Qt::magenta);
        return image;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        bad(QString("Cannot open texture %1: %2").arg(path, file.errorString()));

    const QByteArray bytes = file.readAll();
    if (file.error() != QFileDevice::NoError)
        bad(QString("Cannot read texture %1: %2").arg(path, file.errorString()));

    if (path.endsWith(".dds", Qt::CaseInsensitive)) {
        const auto info = ddsInfo(bytes);
        return decodeBC(bytes.constData() + 128, int(info.width), int(info.height), info.bc);
    }

    return tga(bytes);
}
UploadedTexture loadTexture(QRhi *rhi, QRhiResourceUpdateBatch *batch, const QString &path) {
    QImage image;
    if (path.startsWith('$')) {
        image = QImage(1, 1, QImage::Format_RGBA8888);
        image.fill(path == "$white$" ? Qt::white : path == "$black$" ? Qt::black : Qt::magenta);
    } else {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly))
            bad(f.errorString());

        const auto b = f.readAll();
        if (path.endsWith(".dds", Qt::CaseInsensitive)) {
            const auto info = ddsInfo(b);
            const auto [w, h, mips, bc] = info;
            const auto format = bc == 1 ? QRhiTexture::BC1 : bc == 2 ? QRhiTexture::BC2 : QRhiTexture::BC3;
            const bool compressed = rhi->isTextureFormatSupported(format);
            auto texture = std::unique_ptr<QRhiTexture>(rhi->newTexture(
                compressed ? format : QRhiTexture::RGBA8, QSize(w, h), 1, QRhiTexture::MipMapped));
            if (!texture->create())
                bad("Cannot create DDS texture");

            QVector<QRhiTextureUploadEntry> entries;
            entries.reserve(mips);
            qsizetype pos = 128;
            for (int level = 0; level < int(mips); ++level) {
                const int lw = std::max(1u, w >> level), lh = std::max(1u, h >> level);
                const int size = ((lw + 3) / 4) * ((lh + 3) / 4) * (bc == 1 ? 8 : 16);
                if (pos + size > b.size())
                    bad("Truncated DDS mip chain");

                if (compressed)
                    entries.append(
                        {0, level, QRhiTextureSubresourceUploadDescription(b.constData() + pos, size)});
                else
                    entries.append(
                        {0, level,
                         QRhiTextureSubresourceUploadDescription(decodeBC(b.constData() + pos, lw, lh, bc))});

                pos += size;
            }

            QRhiTextureUploadDescription desc;
            desc.setEntries(entries.cbegin(), entries.cend());
            batch->uploadTexture(texture.get(), desc);
            return {std::move(texture), int(mips) - 1};
        }

        image = tga(b);
    }

    auto texture = std::unique_ptr<QRhiTexture>(rhi->newTexture(
        QRhiTexture::RGBA8, image.size(), 1, QRhiTexture::MipMapped | QRhiTexture::UsedWithGenerateMips));
    if (!texture->create())
        bad("Cannot create texture");

    batch->uploadTexture(texture.get(), image);
    batch->generateMips(texture.get());
    return {std::move(texture), int(std::floor(std::log2(std::max(image.width(), image.height()))))};
}
} // namespace ds2

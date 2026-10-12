"""Open uploads safely and make the metadata-free copy the model sees."""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

try:  # iPhone HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

NOT_A_PHOTO = "That file doesn't look like a photo. Try a JPG, PNG, WebP or HEIC."
MAX_SIDE = 1024


def open_image(data: bytes, *, load: bool = True) -> Image.Image:
    """Open image bytes or raise ValueError with a message fit to show a user."""
    if not data:
        raise ValueError(NOT_A_PHOTO)
    try:
        img = Image.open(io.BytesIO(data))
        if load:
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise ValueError(NOT_A_PHOTO) from e
    return img


def prepare_for_llm(data: bytes, max_side: int = MAX_SIDE) -> tuple[bytes, str]:
    """Upright, downsized RGB JPEG with no EXIF, GPS or ICC metadata."""
    img = ImageOps.exif_transpose(open_image(data))

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((max_side, max_side), Image.LANCZOS)

    clean = Image.new("RGB", img.size)  # fresh image: nothing carried over in .info
    clean.paste(img)
    buf = io.BytesIO()
    clean.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"

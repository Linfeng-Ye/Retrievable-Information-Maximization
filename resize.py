from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 200_000_000_000
def resize_jpeg(
    in_path: str,
    out_path: str,
    max_size: tuple[int, int] = (1920, 1080),  # (max_width, max_height)
    quality: int = 85,
    progressive: bool = True,
    optimize: bool = True
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with Image.open(in_path) as im:
        # Fix orientation from EXIF (so portrait photos don't save sideways)
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass

        # Efficient downscale (thumbnail keeps aspect ratio and uses good resampling)
        im = im.convert("RGB")
        im.thumbnail(max_size, resample=Image.Resampling.LANCZOS)

        # Keep EXIF if present
        exif = im.info.get("exif", None)

        save_kwargs = dict(
            format="JPEG",
            quality=int(quality),
            progressive=bool(progressive),
            optimize=bool(optimize),
        )
        if exif is not None:
            save_kwargs["exif"] = exif

        im.save(out_path, **save_kwargs)
        
if __name__ == "__main__":
    resize_jpeg(
        in_path="girl_20000x23466.jpg",
        out_path="girl_8000x8000.jpg",
        max_size=(8000, 8000),   # e.g., constrain longest side to 2048 while keeping aspect ratio
        quality=99
    )
    print("Saved:", "output_resized.jpg")
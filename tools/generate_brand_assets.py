"""Genera la versión raster del logotipo para perfiles externos."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fire_app" / "static" / "logo-vigia-telegram.jpg"


def serif_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> None:
    scale = 4
    image = Image.new("RGB", (512 * scale, 512 * scale), "#f3f0e7")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (92 * scale, 48 * scale, 420 * scale, 464 * scale),
        radius=164 * scale,
        fill="#173f2b",
    )
    font = serif_font(250 * scale)
    draw.text(
        (256 * scale, 268 * scale),
        "V",
        font=font,
        fill="white",
        anchor="mm",
        stroke_width=0,
    )
    image.resize((512, 512), Image.Resampling.LANCZOS).save(
        OUTPUT,
        format="JPEG",
        quality=95,
        optimize=True,
    )


if __name__ == "__main__":
    main()

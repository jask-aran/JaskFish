import cairosvg
from PIL import Image, ImageTk
from io import BytesIO

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def debug_text(text):
    return color_text(text, '31')

def svg_to_photo_image(svg_string):
    png_image = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
    image = Image.open(BytesIO(png_image))
    return ImageTk.PhotoImage(image)
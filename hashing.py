import dhash
from PIL import Image

image = Image.open('Death_star1.png')
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
import dhash
from PIL import Image

image = Image.open('Death_star1.png')
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))

# Traditional hashing 2 pixles different in each photo
# 2bd793518de4cdff1ab88b43c8dfdc192b2a402597ff8888db5a685cc1ce69d8 - Death star 1
# 93c6805e43e3104aeb2e5141a5be2be63b02247e812fd2a9445bdd63a14ca54b - Death star 1.1
# 379a53fd2194cb41907554dff4e9e7904711a1ae44087845cbeed03be6953e6f - Death star 1.2

# https://pypi.org/project/dhash/
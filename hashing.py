import dhash
from PIL import Image

image1 = Image.open("DeathStar1.png")
row, col = dhash.dhash_row_col(image1)
hash1 = dhash.format_hex(row, col)
print(dhash.format_hex(row, col))

image2 = Image.open("DeathStar1.1.png")
row, col = dhash.dhash_row_col(image2)
hash2 = dhash.format_hex(row, col)
print(dhash.format_hex(row, col))

image3 = Image.open("DeathStar1.2.png")
row, col = dhash.dhash_row_col(image3)
# int3 = dhash.format_int(row, col)
# print("Int is "+int3)
hash3 = dhash.format_hex(row, col)
print(dhash.format_hex(row, col))

print("\nRest of them Below\n")

image = Image.open("DeathStar2.jpg")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar3.jpeg")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar4.jpg")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar5.png")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar6.png")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar7.png")
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))
image = Image.open("DeathStar8.png")
row, col = dhash.dhash_row_col(image)


# dhash.get_num_bits_different(dhash.dhash_row_colrg(image1), dhash.dhash_row_col(image2))

# Traditional hashing 2 pixles different in each photo
# 2bd793518de4cdff1ab88b43c8dfdc192b2a402597ff8888db5a685cc1ce69d8 - Death star 1
# 93c6805e43e3104aeb2e5141a5be2be63b02247e812fd2a9445bdd63a14ca54b - Death star 1.1
# 379a53fd2194cb41907554dff4e9e7904711a1ae44087845cbeed03be6953e6f - Death star 1.2

# https://pypi.org/project/dhash/

import imagehash
from PIL import Image
import sys
import os
import dhash

# image1 = Image.open("DeathStar1.png")
# row, col = dhash.dhash_row_col(image1)
# hash1 = dhash.format_hex(row, col)
# print(dhash.format_hex(row, col))

# image2 = Image.open("DeathStar1.1.png")
# row, col = dhash.dhash_row_col(image2)
# hash2 = dhash.format_hex(row, col)
# print(dhash.format_hex(row, col))

# image3 = Image.open("DeathStar1.2.png")
# row, col = dhash.dhash_row_col(image3)
# # int3 = dhash.format_int(row, col)
# # print("Int is "+int3)
# hash3 = dhash.format_hex(row, col)
# print(dhash.format_hex(row, col))


# Define the directory path (use '.' for the current working directory)
folder_path = r"c:\Users\Jimmy\Documents\GitHub\CEG4350\CS-4840_Machine_Learning\Images"
pHigh = -1
pLow = 111
dHigh = -1
dLow = 111

# Iterate through each item in the folder
for item in os.listdir(folder_path):
    # Construct the full path of the item
    item_path = os.path.join(folder_path, item)
    pAve = 0
    dAve = 0

    print("\nComparing " + item)

    for each in os.listdir(folder_path):
        if each == item:
            continue

        image1 = Image.open(r"Images\\" + item)
        image2 = Image.open(r"Images\\" + each)

        hashsize = 1024

        phash1 = imagehash.phash(image1, hash_size=hashsize)
        phash2 = imagehash.phash(image2, hash_size=hashsize)

        dhash1 = imagehash.dhash(image1, hash_size=hashsize)
        dhash2 = imagehash.dhash(image2, hash_size=hashsize)

        totalSize = hashsize * hashsize

        pdifference = phash1 - phash2
        ddifference = dhash1 - dhash2

        row, col = dhash.dhash_row_col(image1)
        hash1 = dhash.format_hex(row, col)

        if ((pdifference / totalSize) * 100) < pLow :
            pLow = ((pdifference / totalSize) * 100)

        if ((pdifference / totalSize) * 100) > pHigh:
            pHigh = ((pdifference / totalSize) * 100)

        if ((ddifference / totalSize) * 100) < dLow :
            dLow = ((ddifference / totalSize) * 100)

        if ((ddifference / totalSize) * 100) > dHigh:
            dHigh = ((ddifference / totalSize) * 100)

        pAve += ((pdifference / totalSize) * 100)
        dAve += ((ddifference / totalSize) * 100)

        print(str((pdifference / totalSize) * 100) + " - \t" +str((ddifference / totalSize) * 100)+" - \t"+ each)

    print("Phash Average - \t"+str(((pAve)/((int(len(os.listdir(folder_path)))-(1))))))
    print("Phash High - \t"+str(pHigh))
    print("Phash Low - \t"+str(pLow))

    print("Dhash Average - \t"+str(((dAve)/((int(len(os.listdir(folder_path)))-(1))))))
    print("Dhash High - \t"+str(dHigh))
    print("Dhash Low - \t"+str(dLow))

# image1 = Image.open('image9.png')
# image2 = Image.open('image13.png')

# # Compute pHash for both images
# hashsize = 400
# phash1 = imagehash.phash(image1,hash_size=hashsize)
# phash2 = imagehash.phash(image2,hash_size=hashsize)


# totalSize = hashsize*hashsize

# difference = phash1-phash2
# print (difference)
# print (totalSize)
# print (difference/totalSize)


# print("\nRest of them Below\n")

# image = Image.open("DeathStar2.jpg")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar3.jpeg")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar4.jpg")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar5.png")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar6.png")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar7.png")
# row, col = dhash.dhash_row_col(image)
# print(dhash.format_hex(row, col))
# image = Image.open("DeathStar8.png")
# row, col = dhash.dhash_row_col(image)


# dhash.get_num_bits_different(dhash.dhash_row_colrg(image1), dhash.dhash_row_col(image2))

# Traditional hashing 2 pixles different in each photo
# 2bd793518de4cdff1ab88b43c8dfdc192b2a402597ff8888db5a685cc1ce69d8 - Death star 1
# 93c6805e43e3104aeb2e5141a5be2be63b02247e812fd2a9445bdd63a14ca54b - Death star 1.1
# 379a53fd2194cb41907554dff4e9e7904711a1ae44087845cbeed03be6953e6f - Death star 1.2

# https://pypi.org/project/dhash/

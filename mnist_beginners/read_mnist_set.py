fp = open("./train-images-idx3-ubyte", "rb")

line = fp.read(4)
print(line)
magicNumber = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("magic number: %d" %magicNumber)

line = fp.read(4)
print(line)
numOfImages = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("number of images: %d" %numOfImages)

line = fp.read(4)
print(line)
numOfRows = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("number of rows: %d" %numOfRows)

line = fp.read(4)
print(line)
numOfColumns = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("number of columns: %d" %numOfColumns)

for i in range(0, 28):
    for j in range(0, 28):
        pixel = fp.read(1)
        if j == 27:
            print("%3d" %pixel[0])
        else:
            print("%3d" %pixel[0], end = '')

fp.close()

fp2 = open("./train-labels-idx1-ubyte", "rb")

line = fp2.read(4)
print(line)
magicNumber = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("magic number: %d" %magicNumber)

line = fp2.read(4)
print(line)
numOfItems = (line[0] << 24) + (line[1] << 16) + (line[2] << 8) + line[3]
print("number of items: %d" %numOfItems)

label = fp2.read(1)
print(label[0])

fp2.close()


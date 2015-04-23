__author__ = 'berkaytoku'

import struct

def parse_image_data(filename, count=60000):
    with open("../data/" + filename, "rb") as f:
        byte = f.read(4)
        magicnumber = struct.unpack('>I', byte)[0]
        byte = f.read(4)
        numimages = struct.unpack('>I', byte)[0]
        byte = f.read(4)
        numrows = struct.unpack('>I', byte)[0]
        byte = f.read(4)
        numcolumns = struct.unpack('>I', byte)[0]
        count = 60000 if count <= 0 else min(numimages, count)
        imagearray = []
        for i in range(0, count):
            imagearray.append([])
            for j in range(0, numrows * numcolumns):
                imagearray[i].append(struct.unpack('>B', f.read(1))[0])

    return imagearray

def parse_label_data(filename, count=60000):
    with open("../data/" + filename, "rb") as f:
        byte = f.read(4)
        magicnumber = struct.unpack('>I', byte)[0]
        byte = f.read(4)
        numitems = struct.unpack('>I', byte)[0]
        count = 60000 if count <= 0 else min(numitems, count)
        labelarray = []
        for i in range(0, count):
            labelarray.append(struct.unpack('>B', f.read(1))[0])

    return labelarray
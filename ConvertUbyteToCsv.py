'''
    Converting ubyte files (MNIST Dataset) to importable csv files
'''

class ConvertUbyteToCSV:

    def convert(self, image_file, label_file, out_file, n):
        f = open(image_file, "rb")
        l = open(label_file, "rb")
        o = open(out_file, "w")

        f.read(16)
        l.read(8)
        images = []

        for i in range(n):
            image = [ord(l.read(1))]
            for j in range(28 * 28):
                image.append(ord(f.read(1)))
            images.append(image)

        for image in images:
            o.write(",".join(str(pix) for pix in image) + "\n")
        f.close()
        o.close()
        l.close()

convertUbyteToCsv = ConvertUbyteToCSV()
convertUbyteToCsv.convert("Dataset/train-images.idx3-ubyte", "Dataset/train-labels.idx1-ubyte", "Dataset/mnist_train.csv", 60000)
convertUbyteToCsv.convert("Dataset/t10k-images.idx3-ubyte", "Dataset/t10k-labels.idx1-ubyte", "Dataset/mnist_test.csv", 10000)

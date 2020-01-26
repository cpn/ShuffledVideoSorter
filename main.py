import image_sorter
import sys

def main():
    if(len(sys.argv)<3):
        raise NameError("You need to specify input and output folders")
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    ImageSorter = image_sorter.ImageSorter(input_folder, output_folder)
    ImageSorter.sort()


if __name__=="__main__":
    main()
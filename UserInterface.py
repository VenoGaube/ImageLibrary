from tkinter import *
from pathlib import Path
from shutil import copy2, SameFileError
from PIL import Image

import cv2
import os
import pathlib
import secrets

# file imports
from facenet.src.align import align_dataset_mtcnn
import facenet.src.classifier as classifier
import findImages


def path_finder(path):
    global path_train_raw, path_train_aligned, path_test_raw, path_test_aligned, path_model, path_classifier_pickle
    for element in Path(path).iterdir():
        try:
            if element.is_file():
                if element.name == "20180402-114759.pb":
                    path_model = Path(path) / Path(element.name)
                elif element.name == "my_classifier.pkl":
                    path_classifier_pickle = Path(path) / Path(element.name)
                else:
                    continue
            else:
                if element.name == "train_raw":
                    path_train_raw = Path(path) / Path(element.name)
                elif element.name == "train_aligned":
                    path_train_aligned = Path(path) / Path(element.name)
                elif element.name == "test_raw":
                    path_test_raw = Path(path) / Path(element.name)
                elif element.name == "test_aligned":
                    path_test_aligned = Path(path) / Path(element.name)
                path_finder(element)
        except WindowsError:
            pass
    # print(path_train_raw, path_train_aligned, path_test_raw, path_test_aligned, path_model, path_classifier_pickle)


path_train_raw = ""  # "\\facenet\\data\\images\\train_raw\\"
path_train_aligned = ""  # "\\facenet\\data\\images\\train_aligned\\"

path_test_raw = ""  # "\\facenet\\data\\images\\test_raw\\"
path_test_aligned = ""  # "\\facenet\\data\\images\\test_aligned\\"

path_model = ""  # "\\facenet\\models\\20180402-114759.pb"
path_classifier_pickle = ""  # "\\facenet\\models\\my_classifier.pkl"

src = ""
all_images = 0
interval = 5  # base interval


class AlignArguments:
    def __init__(self, path_raw_folder, path_aligned_folder):
        self.input_dir = str(path_raw_folder)
        self.output_dir = str(path_aligned_folder)
        self.image_size = 160
        self.margin = 32
        self.random_order = True
        self.detect_multiple_faces = False
        self.gpu_memory_fraction = 0.75


class ClassifyArguments:
    def __init__(self, path_aligned_folder, mode):
        self.use_split_dataset = False
        self.data_dir = str(path_aligned_folder)
        self.mode = mode
        self.model = str(path_model)
        self.classifier_filename = str(path_classifier_pickle)
        # Vsi ti spodaj imajo nek default value znotraj funckije parse_arguments v classifier.py
        self.seed = 666
        self.min_nrof_images_per_class = 1
        self.nrof_train_images_per_class = 20
        self.batch_size = 90
        self.image_size = 160
        self.test_data_dir = str(path_aligned_folder)


def call_commands():
    # Train Command
    findImages.resize_images(str(path_train_raw))
    print("Loading Train Command")
    arguments_train_aligned = AlignArguments(path_train_raw, path_train_aligned)
    align_dataset_mtcnn.main(arguments_train_aligned)
    # print("konec 1. command")

    # Test Command
    findImages.resize_images(str(path_test_raw))
    print("Loading Test Command")
    arguments_train_aligned = AlignArguments(path_test_raw, path_test_aligned)
    align_dataset_mtcnn.main(arguments_train_aligned)
    # print("konec 2. command")

    # Train Command
    print("Loading Classifier TRAIN Command")
    arguments_classifier = ClassifyArguments(path_train_aligned, 'TRAIN')
    classifier.main(arguments_classifier)
    # print("konec 3. command")

    # Classify Command
    print("Loading Classifier CLASSIFY Command")
    arguments_classifier = ClassifyArguments(path_test_aligned, 'CLASSIFY')
    classifier.main(arguments_classifier)
    # print("konec 4. command")

    print("Waiting on results...")
    path_result = os.getcwd()
    path_origin = path_test_raw
    for folder in Path(path_origin).iterdir():
        if folder.name == "gallery":
            path_origin = path_origin / folder
    for folder in Path(path_result).iterdir():
        if folder.name == "results":
            path_result = path_result / folder
    print(path_result, path_origin)
    for folder in Path(path_result).iterdir():
        for pic in folder.iterdir():
            for img in Path(path_origin).iterdir():
                print('\rLoading: /', end="")
                if pic.stem == img.stem:
                    os.remove(pic)
                    print('\rLoading: -', end="")
                    copy2(path_origin / img, path_result / folder)
                    break
                print('\rLoading: \\', end="")

    print("\nDone.")
    exit(0)


def person_name():
    if entry.get() != '':
        dir_name = entry.get().upper()
        new_dir = str(path_train_raw) / Path(dir_name)
        # print("new_dir = " + new_dir)
        main_dir = str(path_train_raw)

        pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)
        try:
            copy2(str(src), new_dir)
        except SameFileError:
            print("SameFileError")
            pass
        # Preverimo če je že dovolj slik v vsaki datoteki in tega ne rabimo več gledat pa lahko zaključimo.
        check_number_of_images(main_dir)
        root.destroy()


def check_number_of_images(path):
    # print("path = " + path)
    num_of_folders = 0

    for folder in Path(path).iterdir():
        # print(folder.name)
        if folder.name == "test.txt":
            continue
        num_of_folders += 1
    if num_of_folders == 0:
        return
    # print("num of folders:")
    # print(num_of_folders)
    counter_array = [0] * num_of_folders

    counter = 0
    limit = 20
    j = 0
    i = 0
    for folder in Path(path).iterdir():
        if folder.name == "test.txt":
            continue
        for element in folder.iterdir():
            if element.name == "test.txt":
                continue
            # print("element = %s" % element)
            counter_array[i] += 1
        i += 1
    # print("counter array:")
    # print(counter_array)
    for element in counter_array:
        # spremeni limit gori na tolko kolko naj bo training slik
        if counter_array[j] >= limit:
            counter += 1
            j += 1

    if counter == len(counter_array):
        # končaj z UI displayem in naredi vse še automatsko
        print("Dovolj slik ste izbrali, hvala. Vrnite se čez 1-2 minuti.")
        # tu je treba pol klicat tiste štiri commande za align in train in classify
        root.destroy()
        cv2.destroyWindow("image")
        call_commands()
    else:
        # user dalje kategorizira stvari
        for folder in Path(path).iterdir():
            # print(folder.name)
            stevec = 0
            if folder.name == "test.txt":
                continue
            for file in folder.iterdir():
                stevec += 1
            if stevec < 20:
                razlika = 20 - stevec
                print("Potrebujemo še %d" % razlika + " slik od osebe: %s." % folder.name)


def not_person():
    print("Na sliki ni osebe.")
    root.destroy()


def automatic():
    root.destroy()
    cv2.destroyWindow("image")
    call_commands()


path_finder(pathlib.PurePath(os.getcwd()))
root = Tk()
# print(path_train_raw, path_train_aligned, path_test_raw, path_test_aligned, path_model, path_classifier_pickle)
check_number_of_images(str(path_train_raw))
root.destroy()

vse_slike = findImages.get_images()
loop = 0  # na vsak interval pogleda sliko, da ne gleda slik ene za drugo
while True:
    image = secrets.choice(vse_slike)
    pot = Path(image['path'])
    src = image['path']
    root = Tk()

    image = cv2.imread(str(image['path']))
    height, width, c = image.shape

    small = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("Image", small)

    entry = Entry(root, width=50)
    entry.pack()
    entry.focus_set()
    confirm = Button(root, text="Confirm", width=10, command=person_name)
    pass_img = Button(root, text="Pass", width=10, command=not_person)
    dovolj_mam = Button(root, text="Automatic", width=10, command=automatic)
    confirm.pack()
    pass_img.pack()
    dovolj_mam.pack()
    mainloop()

from tkinter import *
from pathlib import Path
from shutil import copy2, SameFileError
from os.path import normpath, basename

import cv2
import os
import pathlib
import secrets
import tkinter as tk
import numpy as np

# file imports
import findImages
import config
from facenet.src.align import align_dataset_mtcnn
import facenet.src.classifier as classifier
import facenet.src.encodings as encodings


# TODO: Poglej kako se dela ko je več obrazov na sliki in dodaj to še v dodajanje v te folderje za multi ljudi.


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


class AlignArguments:
    def __init__(self, path_raw_folder, path_aligned_folder):
        self.input_dir = str(path_raw_folder)
        self.output_dir = str(path_aligned_folder)
        self.image_size = 160
        self.margin = 32
        self.random_order = True
        self.detect_multiple_faces = True
        self.gpu_memory_fraction = 0.90


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
        self.batch_size = 45
        self.image_size = 160
        self.test_data_dir = str(path_aligned_folder)


class MultipleFacesFrame:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.input = Entry(self.master, width=40)
        self.input.pack()
        self.input.focus_set()
        self.more_people = tk.Button(self.frame, text=f"Next person", command=self.person_name)
        self.quit = tk.Button(self.frame, text=f"Done", command=self.close_window)
        self.more_people.pack()
        self.quit.pack()
        self.frame.pack()

    def close_window(self):
        self.master.destroy()
        OpenWindow.close_window(self.master)
        cv2.destroyWindow("Image")

    def person_name(self):
        if self.input.get() != '':
            dir_name = self.input.get().upper()
            try:
                if file_check(Path(src).stem, dir_name):
                    config.multiples.append(ImageObject(src))
                    ImageObject.append_to_folder(config.multiples[-1], dir_name)
                    print("Saved %s" % dir_name + ".")
            except SameFileError:
                print("SameFileError")
                pass


class OpenWindow:
    def __init__(self, master):
        self.master = master
        self.show_widgets(master)

    def show_widgets(self, master):
        self.frame = tk.Frame(self.master)
        self.master.title("Input text")
        self.entry = Entry(self.master, width=40)
        self.entry.pack()
        self.entry.focus_set()
        user_input = tk.Button(root, text="Confirm", width=10, command=self.person_name)
        pass_button = tk.Button(root, text="Pass", width=10, command=self.close_window)
        automatic = tk.Button(root, text="Automatic", width=10, command=self.automatic)
        self.create_window("Multiple faces", MultipleFacesFrame)
        user_input.pack()
        pass_button.pack()
        automatic.pack()
        self.frame.pack()

    def create_window(self, text, _class):
        # Button that creates a new window
        tk.Button(self.frame, text=text, width=10, command=lambda: self.new_window(_class)).pack()

    def create_button(self, text, function):
        tk.Button(self.frame, text=text, width=10, command=lambda: function).pack()

    def new_window(self, _class):
        self.win = tk.Toplevel(self.master)
        _class(self.win)

    def close_window(self):
        self.master.destroy()

    def automatic(self):
        self.master.destroy()
        cv2.destroyWindow("Image")
        call_commands()
        self.master.destroy()
        cv2.destroyWindow("Image")

    def person_name(self):
        if self.entry.get() != '':
            dir_name = self.entry.get().upper()
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
            self.check_number_of_images(main_dir)
            self.master.destroy()

    def check_number_of_images(self, path):
        # print("path = " + path)
        num_of_folders = 0

        for folder in Path(path).iterdir():
            # print(folder.name)
            if folder.name == "text.txt":
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
            if folder.name == "text.txt":
                continue
            for element in folder.iterdir():
                if element.name == "text.txt":
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
            self.master.destroy()
            cv2.destroyWindow("Image")
            call_commands()
        else:
            # user dalje kategorizira stvari
            for folder in Path(path).iterdir():
                # print(folder.name)
                stevec = 0
                if folder.name == "text.txt":
                    continue
                for file in folder.iterdir():
                    stevec += 1
                if stevec < 20:
                    razlika = 20 - stevec
                    print("Potrebujemo še %d" % razlika + " slik od osebe: %s." % folder.name)


class ImageObject:
    def __init__(self, path_to_image):
        self.path_to_image = Path(path_to_image)
        self.folders = []
        self.boundingbox = []
        self.clusterID = []
        self.embedding = []

    def append_to_folder(self, folder_name):
        self.folders.append(folder_name)

    def append_bb(self, bounding_boxes):
        self.boundingbox.append(bounding_boxes)

    def append_embedding(self, embedding):
        self.embedding.append(embedding)

    def set_cluster_id(self, clusterID):
        self.clusterID = clusterID


def folder_check(imageInstance, folder_name):
    flag = 0
    for i in range(len(imageInstance.folders)):
        if imageInstance.folders[i] == folder_name:
            flag += 1
    if flag < len(imageInstance.folders):
        ImageObject.append_to_folder(imageInstance, folder_name)


def file_check(src_name, folder_name):
    for i in range(len(config.final_multi)):
        if config.final_multi[i].path_to_image.stem == src_name:
            folder_check(config.final_multi[i], folder_name)
            return False
    return True


def group_images(results_path):
    for i in range(len(config.final_multi)):
        config.final_multi[i].folders.sort()
        config.final_multi[i].folders = remove_duplicates(config.final_multi[i].folders)
        create_folders(config.final_multi[i], results_path)


def create_folders(multi, results_path):
    group_folder = ""
    flag = True
    for folder in multi.folders:
        if flag:
            group_folder = str(folder)
            flag = False
        else:
            group_folder = group_folder + "&" + str(folder)
    group_path = os.path.join(str(results_path), str(group_folder))
    if os.path.isdir(group_path):
        copy2(str(multi.path_to_image), str(group_path))
    else:
        os.mkdir(str(group_path))
        copy2(str(multi.path_to_image), str(group_path))


def remove_duplicates(folder_list):
    return list(dict.fromkeys(folder_list))


def call_commands():
    """
    # Train Command
    print('\rResizing found images.')
    findImages.resize_images(str(path_train_raw))
    print("\rLoading Train Command")
    arguments_train_aligned = AlignArguments(path_train_raw, path_train_aligned)
    align_dataset_mtcnn.main(arguments_train_aligned)
    # print("konec 1. command")
    """
    # Test Command
    print('\rResizing found images.')
    findImages.resize_images(str(path_test_raw))
    print("\rLoading Test Command")
    arguments_train_aligned = AlignArguments(path_test_raw, path_test_aligned)
    align_dataset_mtcnn.main(arguments_train_aligned)
    # print("konec 2. command")

    # Encoding Command
    print("\rLoading Encoder and Train Command")
    arguments_classifier = ClassifyArguments(path_test_aligned, 'TRAIN')
    encodings.main(arguments_classifier)
    # print("konec Encoding command")

    """
    # Train Command
    print("\rLoading Classifier TRAIN Command")
    arguments_classifier = ClassifyArguments(path_train_aligned, 'TRAIN')
    classifier.main(arguments_classifier)
    # print("konec 3. command")
    """

    # Classify Command
    print("\rLoading Classifier CLASSIFY Command")
    arguments_classifier = ClassifyArguments(path_test_aligned, 'CLASSIFY')
    classifier.main(arguments_classifier)
    # print("konec 4. command")

    resize = 10
    for i in range(len(config.data)):
        slika = cv2.imread(str(config.data[i].path_to_image))
        for bb in range(len(config.data[i].boundingbox)):
            x = config.data[i].boundingbox[bb][0]
            y = config.data[i].boundingbox[bb][1]
            w = config.data[i].boundingbox[bb][2]
            h = config.data[i].boundingbox[bb][3]
            cv2.rectangle(slika, (x, y), (x + w, y + h), (36, 255, 12), 1)
            cv2.putText(slika, str(config.data[i].clusterID[bb]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("Image", slika)
        cv2.waitKey(0)

    """
    print("Waiting on results...")
    path_result = os.getcwd()
    path_origin = path_test_raw
    
    for folder in Path(path_origin).iterdir():
        if folder.name == "gallery":
            path_origin = path_origin / folder
    
    for folder in Path(path_result).iterdir():
        if folder.name == "results":
            path_result = path_result / folder
    # print(path_result, path_origin)
    
    for folder in Path(path_result).iterdir():
        for pic in folder.iterdir():
            original_name, ending = os.path.splitext(pic)
            name = original_name
            name = str(name[:-2])
            pic = name + ending
            for img in Path(path_origin).iterdir():
                print('\rLoading: /', end="")
                if Path(pic).stem == img.stem:
                    replace, end = os.path.splitext(pic)
                    replace = original_name + end
                    pic = replace
                    os.remove(pic)
                    print('\rLoading: -', end="")
                    copy2(path_origin / img, path_result / folder)
                    break
                print('\rLoading: \\', end="")
    
    # print("\rResults are in.")
    # print()
    # print("Collecting group images and creating folders")

    # group_images(path_result)
    """

    exit(0)


def check_number_of_images(path):
    # print("path = " + path)
    num_of_folders = 0

    for folder in Path(path).iterdir():
        # print(folder.name)
        if folder.name == "text.txt":
            continue
        num_of_folders += 1
    if num_of_folders == 0:
        return
    counter_array = [0] * num_of_folders

    counter = 0
    limit = 20
    j = 0
    i = 0
    for folder in Path(path).iterdir():
        if folder.name == "text.txt":
            continue
        for element in folder.iterdir():
            if element.name == "text.txt":
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
        cv2.destroyWindow("Image")
        call_commands()
    else:
        # user dalje kategorizira stvari
        for folder in Path(path).iterdir():
            # print(folder.name)
            stevec = 0
            if folder.name == "text.txt":
                continue
            for file in folder.iterdir():
                stevec += 1
            if stevec < 20:
                razlika = 20 - stevec
                print("Potrebujemo še %d" % razlika + " slik od osebe: %s." % folder.name)


path_finder(pathlib.PurePath(os.getcwd()))
# root = Tk()
# check_number_of_images(str(path_train_raw))
# root.destroy()
vse_slike = findImages.get_images()
call_commands()
loop = 0  # na vsak interval pogleda sliko, da ne gleda slik ene za drugo
while True:
    image = secrets.choice(vse_slike)
    pot = Path(image['path'])
    src = image['path']

    image = cv2.imread(str(image['path']))
    height, width, c = image.shape

    small = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("Image", small)

    root = Tk()
    app = OpenWindow(root)
    root.mainloop()

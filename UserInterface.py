import secrets
import shutil
from pathlib import Path
from json import JSONEncoder
from tkinter import *
from tkinter import filedialog
from shutil import copy2

import cv2
import os
import pathlib
import tkinter as tk
import json

# file imports
import findImages
import config
from facenet.src.align import align_dataset_mtcnn
import facenet.src.encodings as encodings
import facenet.src.classifier as classifier


def path_finder(path):
    global path_test_raw, path_test_aligned, path_model, path_classifier_pickle, base_images_path
    for element in Path(path).iterdir():
        try:
            if element.is_file():
                if element.name == "20170512-110547.pb":
                    path_model = Path(path) / Path(element.name)
                elif element.name == "my_classifier.pkl":
                    path_classifier_pickle = Path(path) / Path(element.name)
                else:
                    continue
            else:
                if element.name == "test_raw":
                    base_images_path = Path(path)
                    path_test_raw = Path(path) / Path(element.name)
                elif element.name == "test_aligned":
                    path_test_aligned = Path(path) / Path(element.name)
                path_finder(element)
        except WindowsError:
            pass


path_test_raw = ""              # "\\facenet\\data\\images\\test_raw\\"
path_test_aligned = ""          # "\\facenet\\data\\images\\test_aligned\\"
path_train_aligned = ""         # "\\facenet\\data\\images\\train_aligned\\"
base_images_path = ""           # "\\facenet\\data\\images\\"

path_model = ""                 # "\\facenet\\models\\20180402-114759.pb"
path_classifier_pickle = ""     # "\\facenet\\models\\my_classifier.pkl"

img_flag = True
src = ""
all_images = 0


class AlignArguments:
    def __init__(self, path_raw_folder, path_aligned_folder):
        self.input_dir = str(path_raw_folder)
        self.output_dir = str(path_aligned_folder)
        self.image_size = 160
        self.margin = 25
        self.random_order = True
        self.detect_multiple_faces = True
        self.gpu_memory_fraction = 1.0


class ClassifyArguments:
    def __init__(self, path_aligned_folder, mode):
        self.use_split_dataset = False
        self.data_dir = str(path_aligned_folder)
        self.test_data_dir = str(path_aligned_folder)
        self.mode = mode
        self.model = str(path_model)
        self.classifier_filename = str(path_classifier_pickle)
        # Vsi ti spodaj imajo nek default value znotraj funckije parse_arguments v classifier.py
        self.seed = 666
        self.min_nrof_images_per_class = 1
        self.nrof_train_images_per_class = 1
        self.batch_size = 90
        self.image_size = 160


class FinalClassifyArguments:
    def __init__(self, path_aligned_folder, mode):
        self.use_split_dataset = False
        self.data_dir = str(path_aligned_folder)
        self.test_data_dir = os.path.join(str(path_test_aligned), "gallery")
        self.mode = mode
        self.model = str(path_model)
        self.classifier_filename = str(path_classifier_pickle)
        # Vsi ti spodaj imajo nek default value znotraj funckije parse_arguments v classifier.py
        self.seed = 666
        self.min_nrof_images_per_class = 1
        self.nrof_train_images_per_class = 1
        self.batch_size = 90
        self.image_size = 160


class OpenWindow:
    def __init__(self, master):
        self.master = master
        self.show_widgets()

    def show_widgets(self):
        self.frame = tk.Frame(self.master)
        self.master.title("Input text")
        self.entry = Entry(self.master, width=40)
        self.entry.pack()
        self.entry.focus_set()
        user_input = tk.Button(root, text="Confirm", width=10, command=self.person_name)
        pass_button = tk.Button(self.master, text="Pass", width=10, command=self.close_window)
        # automatic = tk.Button(self.master, text="Automatic", width=10, command=self.automatic)
        user_input.pack()
        pass_button.pack()
        # automatic.pack()
        self.frame.pack()

    def person_name(self):
        global path_train_aligned
        if self.entry.get() != '':
            dir_name = self.entry.get().upper()
            path_train_aligned = os.path.join(base_images_path, "train_aligned")
            new_dir = os.path.join(str(path_train_aligned), Path(dir_name))
            main_dir = str(path_train_aligned)

            pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)
            try:
                copy2(str(image_path), new_dir)
                os.remove(image_path)
            except shutil.SameFileError:
                print("SameFileError")
                pass
            # Preverimo če je že dovolj slik v vsaki datoteki in tega ne rabimo več gledat pa lahko zaključimo.
            self.check_number_of_images(main_dir)
            try:
                self.master.destroy()
            except TclError:
                pass

    def create_button(self, text, function):
        tk.Button(self.frame, text=text, width=10, command=lambda: function).pack()

    def close_window(self):
        self.master.destroy()

    def automatic(self):
        self.master.destroy()
        cv2.destroyWindow("Image")
        call_commands()
        self.master.destroy()
        cv2.destroyWindow("Image")

    def check_number_of_images(self, path):
        global img_flag
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
        limit = 10
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
            img_flag = False
            # tu je treba pol klicat tiste štiri commande za align in train in classify
            print()
            print("Naslednja oseba:")
            self.master.destroy()
            cv2.destroyWindow("Image")
        else:
            # user dalje kategorizira stvari
            for folder in Path(path).iterdir():
                # print(folder.name)
                stevec = 0
                if folder.name == "text.txt":
                    continue
                for file in folder.iterdir():
                    stevec += 1
                if stevec < limit:
                    razlika = limit - stevec
                    print("\rPotrebujemo še %d" % razlika + " slik od osebe: %s." % folder.name, end="")


class ImageObject:
    def __init__(self, path_to_image):
        self.path_to_image = str(path_to_image)
        self.boundingbox = {'bbox': list(), 'path': list(), 'cluster': list(), 'embedding': list()}

    def append_bb(self, bounding_boxes):
        self.boundingbox["bbox"].append(bounding_boxes)

    def append_path(self, path):
        self.boundingbox["path"].append(path)

    def append_cluster(self, cluster):
        self.boundingbox["cluster"].append(cluster)

    def append_emb(self, embedding):
        self.boundingbox["embedding"].append(embedding)

    def reprJSON(self):
        return dict(path_to_image=self.path_to_image, boundingbox=self.boundingbox)


class ImageObjectEncoder(JSONEncoder):
    def default(self, obj):
        return obj.__dict__


def ComplexHandler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable
    else:
        raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj)))


def remove_duplicates(folder_list):
    return list(dict.fromkeys(folder_list))


def draw_bounding_boxes():
    for i in range(len(config.data)):
        slika = cv2.imread(str(config.data[i].path_to_image))
        display_flag = 0
        if len(config.data[i].boundingbox['cluster']) < 1:
            continue
        if len(config.data[i].boundingbox['bbox']) < 1 or (len(config.data[i].boundingbox['bbox']) == 1 and config.data[i].boundingbox['bbox'][0] == 99999):
            continue

        if len(config.data[i].boundingbox['cluster']) <= len(config.data[i].boundingbox['bbox']):
            for j in range(len(config.data[i].boundingbox['cluster'])):
                if config.data[i].boundingbox['cluster'][j] == 99999:
                    continue
                x = int(config.data[i].boundingbox['bbox'][j][0])
                y = int(config.data[i].boundingbox['bbox'][j][1])
                w = int(config.data[i].boundingbox['bbox'][j][2])
                h = int(config.data[i].boundingbox['bbox'][j][3])
                cv2.rectangle(slika, (x, y), (w, h), (36, 255, 12), 1)
                cv2.putText(slika, str(config.data[i].boundingbox['cluster'][j]), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                display_flag = 1
        else:
            for j in range(len(config.data[i].boundingbox['cluster'])):
                if config.data[i].boundingbox['cluster'][j] == 99999:
                    continue
                x = int(config.data[i].boundingbox['bbox'][j][0])
                y = int(config.data[i].boundingbox['bbox'][j][1])
                w = int(config.data[i].boundingbox['bbox'][j][2])
                h = int(config.data[i].boundingbox['bbox'][j][3])
                cv2.rectangle(slika, (x, y), (w, h), (36, 255, 12), 1)
                cv2.putText(slika, str(config.data[i].boundingbox['cluster'][j]), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                display_flag = 1
        if display_flag:
            cv2.imshow("Image", slika)
            cv2.waitKey(0)
    cv2.destroyWindow("Image")


def draw_bounding_boxes_final():
    for i in range(len(config.data)):
        slika = cv2.imread(str(config.data[i].path_to_image))
        display_flag = 0
        if len(config.data[i].boundingbox['cluster']) < 1:
            continue
        if len(config.data[i].boundingbox['bbox']) < 1 or (len(config.data[i].boundingbox['bbox']) == 1 and config.data[i].boundingbox['bbox'][0] == 99999):
            continue

        if len(config.data[i].boundingbox['cluster']) <= len(config.data[i].boundingbox['bbox']):
            for j in range(len(config.data[i].boundingbox['cluster'])):
                if config.data[i].boundingbox['cluster'][j] == 99999:
                    continue
                if type(config.data[i].boundingbox['cluster'][j]) == int:
                    continue

                x = int(config.data[i].boundingbox['bbox'][j][0])
                y = int(config.data[i].boundingbox['bbox'][j][1])
                w = int(config.data[i].boundingbox['bbox'][j][2])
                h = int(config.data[i].boundingbox['bbox'][j][3])
                cv2.rectangle(slika, (x, y), (w, h), (36, 255, 12), 1)
                cv2.putText(slika, str(config.data[i].boundingbox['cluster'][j]), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                display_flag = 1
        else:
            for j in range(len(config.data[i].boundingbox['cluster'])):
                if config.data[i].boundingbox['cluster'][j] == 99999:
                    continue
                if type(config.data[i].boundingbox['cluster'][j]) == int:
                    continue

                x = int(config.data[i].boundingbox['bbox'][j][0])
                y = int(config.data[i].boundingbox['bbox'][j][1])
                w = int(config.data[i].boundingbox['bbox'][j][2])
                h = int(config.data[i].boundingbox['bbox'][j][3])
                cv2.rectangle(slika, (x, y), (w, h), (36, 255, 12), 1)
                cv2.putText(slika, str(config.data[i].boundingbox['cluster'][j]), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                display_flag = 1
        if display_flag:
            cv2.imshow("Image", slika)
            cv2.waitKey(0)
    cv2.destroyWindow("Image")


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
        # tu je treba pol klicat tiste štiri commande za align in train in classify
        # root.destroy()
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
                print("\rPotrebujemo še %d" % razlika + " slik od osebe: %s." % folder.name, end="")


def delete_create():
    gallery = "gallery"
    results = "results"
    results_final = "results_final"
    aligned = "train_aligned"

    raw_gallery = os.path.join(path_test_raw, gallery)
    aliged_gallery = os.path.join(path_test_aligned, gallery)
    results_delete = os.path.join(os.getcwd(), results)
    results_final_delete = os.path.join(os.getcwd(), results_final)
    train_path = os.path.join(base_images_path, aligned)

    if os.path.isdir(results_final_delete):
        shutil.rmtree(results_final_delete)

    if os.path.isdir(results_delete):
        shutil.rmtree(results_delete)

    if os.path.isdir(raw_gallery):
        shutil.rmtree(raw_gallery)

    if os.path.isdir(aliged_gallery):
        shutil.rmtree(aliged_gallery)

    if os.path.isdir(train_path):
        shutil.rmtree(train_path)

    os.mkdir(raw_gallery)
    os.mkdir(aliged_gallery)


# Function for opening the file explorer window
def browseFiles():
        filename = filedialog.askopenfilename(initialdir=str(config.result_path), title="Select a File", filetypes=[("All files", "*.*")])


def close_window():
    window.destroy()


def file_explore():

    # Set window title
    window.title('File Explorer')

    # Set window size
    window.geometry("700x200")

    # Set window background color
    window.config(background="white")

    # Create a File Explorer label
    label_file_explorer = Label(window, text="File Explorer using Tkinter", width=100, height=4, fg="blue")

    button_explore = Button(window, text="Browse Files", command=browseFiles)

    button_exit = Button(window, text="Exit", command=close_window)

    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column=1, row=1)

    button_explore.grid(column=1, row=2)

    button_exit.grid(column=1, row=3)

    # Let the window wait for any events
    window.mainloop()


def reconfigure_delete_config():
    changed_folders = []
    for i in results_array:
        flag = False
        for j in reconfigure_array:
            if i == j:
                flag = True
        if not flag:
            changed_folders.append(i)
    i = 0
    while i in range(len(config.data)):
        data = config.data[i].boundingbox["cluster"]
        j = 0
        while j in range(len(data)):
            for value in changed_folders:
                basic = config.data[i].boundingbox
                if int(data[j]) == int(value):
                    if len(data) == 1:
                        del config.data[i]
                        i -= 1
                        break
                    del data[j]
                    del basic["embedding"][j]
                    del basic["path"][j]
                    del basic["bbox"][j]
                    j -= 1
                    break
            j += 1
        i += 1

    # Write to JSON file
    imageJSONData = json.dumps(config.data, indent=4, cls=ImageObjectEncoder)
    with open('data.json', 'w') as outfile:
        outfile.write(imageJSONData)


def reconfigure_config():
    return 0


def count_folders():
    names = []
    for folder in Path(config.result_path).iterdir():
        for file in folder.iterdir():
            if Path(file).is_dir():
                names.append(file.stem)
        names.append(folder.stem)
    return names


def rename_clusters(previous, current):
    for i in range(len(config.data)):
        for j in range(len(config.data[i].boundingbox["cluster"])):
            data = config.data[i].boundingbox["cluster"]
            if int(data[j]) == int(previous):
                config.data[i].boundingbox["cluster"][j] = int(current)

    # Write to JSON file
    imageJSONData = json.dumps(config.data, indent=4, cls=ImageObjectEncoder)
    with open('data.json', 'w') as outfile:
        outfile.write(imageJSONData)


def move_images(previous, current):
    for file in previous.iterdir():
        copy2(file, str(current))


def check_if_moved():
    for folder in Path(config.result_path).iterdir():
        for file in folder.iterdir():
            if Path(file).is_dir():
                rename_clusters(file.stem, folder.stem)
                move_images(Path(file), folder)
                shutil.rmtree(file)


def get_result_images(folder):
    folder_images = []
    for file in folder.iterdir():
        folder_images.append(file)
    return folder_images


def results_command():

    # Train Command
    print("\rLoading Classifier TRAIN Command")
    arguments_classifier = FinalClassifyArguments(path_train_aligned, 'TRAIN')
    classifier.main(arguments_classifier)
    # print("konec 3. command")

    # Classify Command
    print("\rLoading Classifier CLASSIFY Command")
    arguments_classifier = FinalClassifyArguments(path_test_aligned, 'CLASSIFY')
    classifier.main(arguments_classifier)
    # print("konec 4. command")

    # Write to JSON file
    imageJSONData = json.dumps(config.data, indent=4, cls=ImageObjectEncoder)
    with open('data.json', 'w') as outfile:
        outfile.write(imageJSONData)

    draw_bounding_boxes_final()


def call_commands():

    # Test Command
    print('\rResizing found images.')
    findImages.resize_images(str(path_test_raw))
    print("\rLoading Test Command")
    arguments_train_aligned = AlignArguments(path_test_raw, path_test_aligned)
    align_dataset_mtcnn.main(arguments_train_aligned)

    # Encoding Command
    print("\rLoading Clustering Command")
    arguments_classifier = ClassifyArguments(path_test_aligned, 'TRAIN')
    encodings.main(arguments_classifier)

    # Write to JSON file
    imageJSONData = json.dumps(config.data, indent=4, cls=ImageObjectEncoder)
    with open('data.json', 'w') as outfile:
        outfile.write(imageJSONData)

    draw_bounding_boxes()


def move_results_to_test_aligned():
    gallery = "gallery"
    path_test_gallery = os.path.join(path_test_aligned, gallery)
    try:
        shutil.rmtree(path_test_gallery)
    except FileNotFoundError:
        pass
    os.mkdir(path_test_gallery)
    for folder in Path(config.result_path).iterdir():
        for file in folder.iterdir():
            copy2(str(file), str(path_test_gallery))


path_finder(pathlib.PurePath(os.getcwd()))
delete_create()
vse_slike = findImages.get_images()
call_commands()
# Create the root window
window = Tk()
results_array = count_folders()
file_explore()
check_if_moved()
reconfigure_array = count_folders()
if len(results_array) != len(reconfigure_array):
    reconfigure_config()
    reconfigure_delete_config()
    draw_bounding_boxes()

# Anotacija slik
for folder in Path(config.result_path).iterdir():
    img_flag = True
    while img_flag:
        vse_slike = get_result_images(folder)
        image_path = secrets.choice(vse_slike)

        image = cv2.imread(str(image_path))
        height, width, c = image.shape
        cv2.imshow("Image", image)

        root = Tk()
        app = OpenWindow(root)
        root.mainloop()

move_results_to_test_aligned()
results_command()
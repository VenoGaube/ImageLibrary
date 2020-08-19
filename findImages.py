import os
from pathlib import Path
from shutil import copy2
from PIL import Image, ExifTags
from tkinter.filedialog import askdirectory

import tkinter as tk
import cv2
import exifread
import pathlib


def path_finder(path):
    global path_gallery
    for element in Path(path).iterdir():
        try:
            if element.is_file():
                continue
            else:
                if element.name == "gallery":
                    path_gallery = Path(path) / Path(element.name)
                    break
                path_finder(element)
        except WindowsError:
            pass


path_gallery = ""  # "\\facenet\\data\\images\\test_raw\\gallery"
resize = 0.3
path_finder(pathlib.PurePath(os.getcwd()))


def get_date(img_path):
    f = open(img_path, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    # Vzamemo samo tiste, ki so vezani na datum in čas
    dateTime = None
    orientation = None
    if "Image DateTime" in tags or "DateTimeOriginal" in tags:
        dateTime = str(tags["Image DateTime"])
        # Vrnemo pridobljen datum in čas
    if "Image Orientation" in tags or "Orientation" in tags:
        orientation = tags["Image Orientation"].values[0]
    return dateTime, orientation


def search_directory(rootdir, array):
    print('\rLoading: /', end="")
    try:
        if rootdir.is_file():
            # če se file konča z .jpeg ali .jpg je kategoriziran kot slika
            if rootdir.name.endswith(".jpeg") or rootdir.name.endswith(".JPEG") or rootdir.name.endswith(".jpg")\
                    or rootdir.name.endswith(".JPG"): # or rootdir.name.endswith(".png") or rootdir.name.endswith(".PNG"):
                # Pridobimo čas in datum iz meta podatkov
                image_date, orientation = get_date(rootdir)
                image = Image.open(rootdir)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                    image.save(rootdir)
                    image.close()
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                    image.save(rootdir)
                    image.close()
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                    image.save(rootdir)
                    image.close()

                # Če smo dobili nek datum in čas potem gremo v if, drugače to sliko popolnoma preskočimo
                if image_date is not None:
                    # Dodamo sliko in podatke v array
                    gallery_dir = path_gallery
                    copy2(rootdir, gallery_dir)
                    picture = {'path': Path(rootdir), 'date': image_date}
                    array.append(dict(picture))
                print('\rLoading: -', end="")

        else:
            for file in rootdir.iterdir():
                print('\rLoading: -', end="")
                # Vse datoteke ki jih nočemo gledat
                if file.name.startswith('.') or file.name == "AppData" or file.name == "desktop.ini" or file.name == "facenet":
                    continue

                # Rekurzivno gremo čez vse znotraj folderja
                search_directory(file, array)
                print('\rLoading: -', end="")
    except WindowsError:
        pass


def get_images():
    # User določi root directory
    root = tk.Tk()
    path = askdirectory(title='Select Folder')
    root.withdraw()
    pot = path

    # Tukaj bodo vse slike in njihovi podatki
    array = []

    # Sprehodimo se po vseh folderjih znotraj root direktorija
    for folder in Path(pot).iterdir():
        if folder.name != "Public" and folder.name != "Default" and folder.name != "All Users" \
                and folder.name != "desktop.ini" and folder.name != "Default User" and folder.name != "$Recycle.Bin"\
                and folder.name != "Windows":
            # Tu je idealno, da se dobi samo direktorij Users\Uporabnik, ne pa še vsi ostali neuporabni folderji
            # ("Searching " + str(folder) + " for all images.")
            print(folder)
            search_directory(folder, array)
            print('\rLoaded Images From Above Folder.')

    print("\rFound %d" % len(array) + " images.")
    root.destroy()
    return array


def resize_images(path):
    for slika in Path(path).iterdir():
        if slika.is_dir():
            resize_images(slika)
            continue
        if slika.name.endswith(".txt"):
            continue
        resized = Image.open(slika)
        shape = resized.size
        shape = list(shape)
        shape[0] = int(shape[0] * resize)
        shape[1] = int(shape[1] * resize)
        print('\rLoading: /', end="")
        resized_img = resized.resize(tuple(shape), Image.ANTIALIAS)
        print('\rLoading: -', end="")
        resized_img.save(slika)
        print('\rLoading: \\', end="")
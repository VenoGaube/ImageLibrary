import os
from pathlib import Path
from shutil import copy2
from PIL import Image, ExifTags
from tkinter.filedialog import askdirectory

import tkinter as tk
import cv2
import exifread
import pathlib
import numpy as np


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



def get_rotation(img_path):
    f = open(img_path, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    if "Image Orientation" in tags or "Orientation" in tags:
        orientation = tags["Image Orientation"].values[0]
        return orientation


def get_date(img_path):
    f = open(img_path, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    if "Image DateTime" in tags or "DateTimeOriginal" in tags:
        dateTime = str(tags["Image DateTime"])
        return dateTime


def search_directory(rootdir, array):
    print('\rLoading: /', end="")
    try:
        if rootdir.is_file():
            # če se file konča z .jpeg ali .jpg je kategoriziran kot slika
            if rootdir.name.endswith(".jpeg") or rootdir.name.endswith(".JPEG") or rootdir.name.endswith(".jpg")\
                    or rootdir.name.endswith(".JPG") or rootdir.name.endswith(".png") or rootdir.name.endswith(".PNG"):
                # Pridobimo čas in datum iz meta podatkov
                image_date = get_date(rootdir)
                # Če smo dobili nek datum in čas potem gremo v if, drugače to sliko popolnoma preskočimo
                # if image_date is not None:
                # Dodamo sliko in podatke v array
                image_name = os.path.basename(rootdir)
                gallery_image = os.path.join(path_gallery, str(image_name))
                copy2(rootdir, path_gallery)
                image = Image.open(gallery_image)
                try:
                    exif = image.info['exif']
                    image.save(gallery_image, 'JPEG', exif=exif)
                except KeyError:
                    image.save(gallery_image, 'JPEG')
                    pass
                picture = {'path': Path(gallery_image), 'date': image_date}
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
    path_finder(pathlib.PurePath(os.getcwd()))
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
        flag = False
        try:
            exif = resized.info['exif']
            flag = True
        except KeyError:
            pass
        shape = resized.size
        shape = list(shape)
        if shape[0] < 1500 and shape[1] < 1500:
            continue
        shape[0] = int(shape[0] * resize)
        shape[1] = int(shape[1] * resize)
        print('\rLoading: /', end="")
        resized_img = resized.resize(tuple(shape), Image.ANTIALIAS)
        print('\rLoading: -', end="")
        if flag:
            resized_img.save(slika, 'JPEG', exif=exif)
        else:
            resized_img.save(slika, 'JPEG')
        print('\rLoading: \\', end="")

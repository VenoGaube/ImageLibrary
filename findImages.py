import os
from pathlib import Path
from shutil import copy2

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

path_finder(pathlib.PurePath(os.getcwd()))


def get_date(img_path):
    with open(img_path, 'rb') as image:
        # Dobimo meta podatke
        exif = exifread.process_file(image)
        # Vzamemo samo tiste, ki so vezani na datum in čas
        if "Image DateTime" in exif or "DateTimeOriginal" in exif:
            dateTime = str(exif["Image DateTime"])
            # Vrnemo pridobljen datum in čas
            return dateTime


def search_directory(rootdir, array):
    print('\rLoading: /', end="")
    try:
        if rootdir.is_file():
            # če se file konča z .jpeg ali .jpg je kategoriziran kot slika
            if rootdir.name.endswith(".jpeg") or rootdir.name.endswith(".JPG"):
                # Pridobimo čas in datum iz meta podatkov
                image_date = get_date(rootdir)
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
                # Vse datoteke ki jih nočemo gledat
                if file.name.startswith(
                        '.') or file.name == "AppData" or file.name == "desktop.ini" or file.name == "facenet":
                    continue

                # Rekurzivno gremo čez vse znotraj folderja
                search_directory(file, array)
                print('\rLoading: -', end="")
    except WindowsError:
        pass


def get_images():
    # Določitev root directory-ja
    path = pathlib.PurePath(os.getcwd())
    root = path.parts[0]

    # Tukaj bodo vse slike in njihovi podatki
    array = []

    # Sprehodimo se po vseh folderjih znotraj root direktorija
    for folder in Path(root).iterdir():
        if folder.name != "Public" and folder.name != "Default" and folder.name != "All Users" \
                and folder.name != "desktop.ini" and folder.name != "Default User" and folder.name != "$Recycle.Bin":
            # Tu je idealno, da se dobi samo direktorij Users\Uporabnik, ne pa še vsi ostali neuporabni folderji
            # ("Searching " + str(folder) + " for all images.")
            print(folder)
            search_directory(folder, array)
            print('\rLoaded Images From Above Folder.')
    print("\rFound all images.")
    return array

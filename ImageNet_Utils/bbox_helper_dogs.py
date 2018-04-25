#!/usr/bin/env python
import os
from PIL import Image
import sys
import zipfile
import xml.etree.ElementTree as ET
import argparse


def scanAnnotationFolder(annotationFolderPath):
    annotationFiles = []
    for root, dirs, files in os.walk(annotationFolderPath):
        for file in files:
            annotationFiles.append(os.path.join(root, file))
    return annotationFiles


# Bounding Box Helper
class BBoxHelper:
    def __init__(self, annotation_file, image_path=None):
        self.annotation_file = annotation_file
        xmltree = ET.parse(annotation_file)
        filename = xmltree.find('filename').text
        # wnid = filename.split('_')[0]
        # image_id = filename.split('_')[1]
        wnid = os.path.basename(os.path.dirname(annotation_file)).split('-')[0]
        image_id = os.path.basename(annotation_file)
        # create a dict to save filename, wnid, image id, etc..
        # self.annotation_filename = filename
        self.annotation_filename = image_id
        self.wnid = wnid
        self.image_id = image_id
        # find bounding box
        objects = xmltree.findall('object')
        self.rects = []
        for object_iter in objects:
            bndbox = object_iter.find("bndbox")
            self.rects.append([int(it.text) for it in bndbox])

        localPath = xmltree.find('path')

        self.imgPath = None
        if localPath is not None and os.path.exists(localPath.text):
            self.imgPath = localPath.text

        if image_path is not None:
            self.imgPath = image_path

    def saveBoundBoxImage(self, imgPath=None, outputFolder=None):
        # if imgPath is not None:
        #     self.imgPath = imgPath
        #
        # if imgPath is None and self.imgPath is None:
        #     self.imgPath = self.findImagePath()

        if outputFolder == None:
            self.imgPath = self.findImagePath(imgPath)
            outputFolder = os.path.join(os.path.dirname(os.path.dirname(imgPath)),
                                        'bounding_box_imgs',
                                        os.path.basename(imgPath))

        # annotation_file_dir = os.path.dirname(os.path.realpath(self.annotation_file))
        # outputFolder = os.path.join(annotation_file_dir, savedTargetDir)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        # Get crop images
        bbs = []
        im = Image.open(self.imgPath)
        for box in self.rects:
            bbs.append(im.crop(box))
        # Save them to target dir
        count = 0
        for box in bbs:
            count = count + 1
            outPath = str(os.path.join(outputFolder, self.annotation_filename + '_box' + str(count) + '.jpg'))
            box.save(outPath)
            print('save to ' + outPath)

    def get_BoudingBoxs(self):
        return self.rects

    def getWnid(self):
        return self.wnid

    def findImagePath(self, search_folder='.'):
        filename = self.annotation_filename + str('.jpg')
        for root, dirs, files in os.walk(search_folder):
            for file in files:
                if filename == file:
                    return os.path.join(root, file)
        print(filename + ' not found')
        return None


def saveAsBoudingBoxImg(xmlfile):
    bbhelper = BBoxHelper(xmlfile)
    print(bbhelper.findImagePath())
    # Search image path according to bounding box xml, and crop it
    if shouldSaveBoundingBoxImg:
        print(bbhelper.get_BoudingBoxs())
        bbhelper.saveBoundBoxImage()


def saveAsBoudingBoxImg(xmlfile, imagePath):
    bbhelper = BBoxHelper(xmlfile)
    print(bbhelper.findImagePath(search_folder=imagePath))
    # Search image path according to bounding box xml, and crop it
    if shouldSaveBoundingBoxImg:
        print(bbhelper.get_BoudingBoxs())
        bbhelper.saveBoundBoxImage(imagePath)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Help the user to download, crop, and handle images from ImageNet')
    p.add_argument('--datadir', help='Data dir')
    p.add_argument('--bxmlpath', help='Boudingbox xml path')
    p.add_argument('--bxmldir', help='Boudingbox dir path')
    p.add_argument('--save_boundingbox', help='Search images and crop the bounding box by image paths',
                   action='store_true', default=False)
    args = p.parse_args()
    # Give bounding_box XML and show its JPEG path and bounding rects
    boundingbox_xml_file = args.bxmlpath
    boudingbox_xml_dir = args.bxmldir
    shouldSaveBoundingBoxImg = args.save_boundingbox
    data_dir = args.datadir

    if not data_dir is None:
        boudingbox_xml_dirs = os.path.join(data_dir, 'Annotation')

    if not boundingbox_xml_file is None:
        saveAsBoudingBoxImg(boundingbox_xml_file)

    if not boudingbox_xml_dir is None:
        allAnnotationFiles = scanAnnotationFolder(boudingbox_xml_dir)
        for xmlfile in allAnnotationFiles:
            saveAsBoudingBoxImg(xmlfile)

    if not boudingbox_xml_dirs is None:
        all_dogs = os.listdir(boudingbox_xml_dirs)
        for boudingbox_xml_dir in all_dogs:
            scan_folder = os.path.join(boudingbox_xml_dirs, boudingbox_xml_dir)
            allAnnotationFiles = scanAnnotationFolder(scan_folder)
            for xmlfile in allAnnotationFiles:
                imagePath = os.path.join(data_dir, 'Images', boudingbox_xml_dir)
                saveAsBoudingBoxImg(xmlfile, imagePath)

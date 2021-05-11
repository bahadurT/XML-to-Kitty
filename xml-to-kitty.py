# ----------------------------------
# Date: 11-05-2021
# Place: India
# Author: Bahadur Singh Thakur
# python main.py --dataset-path D:\datasets\Person --input-dims_width 300 --input-dims_height 300 --label_filename "person"
# ----------------------------------

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring


class xml_to_kitti():
	def __init__(self, images_dir, labels_dir, kitti_base_dir, kitti_resize_dims):
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.kitti_base_dir = kitti_base_dir
		self.kitti_resize_dims = kitti_resize_dims

		try:
			self.kitti_images =self.kitti_base_dir+"\\images"
			os.makedirs(self.kitti_images,mode=0o777)
		except:
			print("Directory Already Exists")
			self.kitti_images = os.path.join(self.kitti_base_dir,"images")


		try:
			self.kitti_labels = self.kitti_base_dir+"\\labels"
			os.makedirs(self.kitti_labels,mode = 0o777)
		except:
			print("Directory Already Exists")
			self.kitti_labels = os.path.join(self.kitti_base_dir,"labels")


	def get_image_metafile(self, image_file):
		image_name = os.path.splitext(image_file)[0]
		return os.path.join(self.labels_dir, str(image_name+'.xml'))

	def make_labels(self, image_name, category_names, bboxes):
		file_image = os.path.splitext(image_name)[0]
		img = Image.open(os.path.join(self.images_dir, image_name)).convert("RGB")
		resize_img = img.resize(self.kitti_resize_dims)
		resize_img.save(os.path.join(self.kitti_images, file_image + '.jpg'), 'JPEG')

		with open(os.path.join(self.kitti_labels, file_image + '.txt'), 'w') as label_file:
			for i in range(0, len(bboxes)):
				resized_bbox = self.resize_bbox(img=img, bbox=bboxes[i], dims=self.kitti_resize_dims)
				out_str = [category_names[i].replace(" ","")
							+ ' ' + ' '.join(['0'] * 1)
							+ ' ' + ' '.join(['0'] * 2)
							+ ' ' + ' '.join([b for b in resized_bbox])
							+ ' ' + ' '.join(['0'] * 7)
							+ '\n']
				label_file.write(out_str[0])


	def resize_bbox(self, img, bbox, dims):
		img_w, img_h = img.size
		x_min, y_min, x_max, y_max = bbox
		ratio_w, ratio_h = dims[0] / img_w, dims[1] / img_h
		new_bbox = [str(int(np.round(float(x_min) * ratio_w,2))), str(int(np.round(float(y_min) * ratio_h,2))), str(int(np.round(float(x_max) * ratio_w,2))),str(int(np.round(float(y_max) * ratio_h,2)))]
		return new_bbox


	def get_data_attributes(self,labels_dir):
		image_extensions = ['.jpeg', '.jpg', '.png']

		for image_name in os.listdir(self.images_dir):
			print(image_name)
			if image_name.endswith('.jpeg') or image_name.endswith('.jpg') or image_name.endswith('.png'):
				labels_txt = self.get_image_metafile(image_file=image_name)
				if os.path.isfile(labels_txt):
					bboxes = []
					categories= []
					xml_file = os.path.join(labels_dir,labels_txt)
					tree = ET.parse(xml_file)
					root = tree.getroot()
					file_name = root.find('object')
					node_root = Element('annotation')
					node_filename = SubElement(node_root, 'filename')
					for type_tag in root.findall('object'):
						name = type_tag.find('name').text
						xmin = int(type_tag.find('bndbox/xmin').text)
						ymin = int(type_tag.find('bndbox/ymin').text)
						xmax = int(type_tag.find('bndbox/xmax').text)
						ymax = int(type_tag.find('bndbox/ymax').text)
						box = [xmin,ymin,xmax,ymax]
						categories.append(name)
						bboxes.append(box)

					if bboxes:
						self.make_labels(image_name=image_name, category_names=categories, bboxes=bboxes)



	def test_labels(self,kitti_base_dir, file_name):
		img = Image.open(os.path.join(kitti_base_dir+'\\images\\', file_name + '.jpg'))
		img_w , img_h = img.size
		print("Image size {} X {} ".format(img_w,img_h))
		text_file = open(os.path.join(kitti_base_dir+'\\labels\\', file_name + '.txt'), 'r')

		bbox = []
		category = []
		for line in text_file:
			features = line.split()
			bbox.append([float(features[4]), float(features[5]), float(features[6]), float(features[7])])
			category.append(features[0])
		print("Bounding Box", bbox)
		print("Category:", category)
		i = 0
		outline_clr = ""
		for bb in bbox:
			draw_img = ImageDraw.Draw(img)
			shape = ((bb[0], bb[1]), (bb[2], bb[3]))
			outline_clr = "red"
			draw_img.rectangle(shape, fill=None, outline=outline_clr, width=1)
			i += 1
		img.show()

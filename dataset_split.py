import os


def checkTxt(txt_path):
	with open(txt_path, "r") as file:
		if file.empty():
			return False
		else:
			return True


if __name__ == "__main__":
	txt_dir = r'/home/hosico/DataDisk/hdd2/Dataset/data/test/training/label'
	txt_list = os.listdir(txt_dir)
	for txt_file in txt_list:
		txt_abs = os.path.join(txt_dir, txt_file)
		print(checkTxt(txt_abs), " -- ", txt_abs)

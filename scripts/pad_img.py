import os

img_path = "/clevr_dc/images/"
sc_path = "/clevr_dc/sc_images/"

img_file = sorted(os.listdir(img_path))
sc_file = sorted(os.listdir(sc_path))

zeros = '000000'


for i, img in enumerate(img_file):
    name, _ = img.split('.')
    pad_lens = 6 - len(str(name))
    full_name = zeros[:pad_lens] + name + '.png'
    os.rename(os.path.join(img_path, img), os.path.join(img_path, full_name))

for i, img in enumerate(sc_file):
    name, _ = img.split('.')
    pad_lens = 6 - len(str(name))
    full_name = zeros[:pad_lens] + name + '.png'
    os.rename(os.path.join(sc_path, img), os.path.join(sc_path, full_name))


import json

with open("/clevr_dc/captions.json", 'r') as f:
    captions = json.load(f)

dic = {}

zeros = '000000'
for i, v in captions.items():
    lens = 6-len(i.split('.')[0])
    full_name = zeros[:lens] + str(i)+'.png'
    dic[full_name] = v

with open("/clevr_dc/change_captions.json", 'w') as f:
    json.dump(dic, f)



import json
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

def main():

    model = CLIPModel.from_pretrained("/ML-A100/sshare-app/zhangsan/models/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("/ML-A100/sshare-app/zhangsan/models/clip-vit-large-patch14")
    file_path = "data1.json"
    with open(file_path, 'r') as file:
        # 使用 read() 方法读取整个文件内容
        data = json.load(file)
    count = 0
    total_org = 0.0
    total_cogvlm = 0.0
    for value in data:
        name = "./laion-high/"+value['name']+".jpg"
        org = value["org"]
        cogvlm = value["cogvlm"]


        image = Image.open(name)

        inputs = processor(text=[org, cogvlm], images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        org_score = probs[0][0].item()
        cogvlm_score = probs[0][1].item()
        value["org_score"] = org_score
        value["cogvlm_score"] = cogvlm_score
        total_org += org_score
        total_cogvlm += cogvlm_score      
        count += 1
        if count % 10 == 0 :
            file_path = "data_score.json"

            # 使用 json.dump() 函数将数据写入文件
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)    
    print("org:"+total_org/count)
    print("cogvlm:"+total_cogvlm/count)    
    file_path = "data_score.json"
    # 使用 json.dump() 函数将数据写入文件
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)             
if __name__ == "__main__":
    main()
    tr = torch.FloatTensor([[1, 2, 3]])


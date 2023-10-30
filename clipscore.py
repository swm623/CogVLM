import json
from PIL import Image
import requests
import torch
from llam2req import change_desc_with_prompt
from transformers import CLIPProcessor, CLIPModel
from wordutils import word_count_with_punctuation,cat_word_70
def main():
    file_path = "./data/data_score1.json"
    model = CLIPModel.from_pretrained("/ML-A100/sshare-app/zhangsan/models/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("/ML-A100/sshare-app/zhangsan/models/clip-vit-large-patch14")
    file_path_data = "./data/data.json"
    with open(file_path_data, 'r') as file:
        # 使用 read() 方法读取整个文件内容
        data = json.load(file)
    count = 0
    org_count = 0
    cogvlm_count = 0
    total_org = 0.0
    total_org_similarity_score = 0.0
    total_cogvlm = 0.0
    total_cogvlm_similarity_score = 0.0
    for value in data:
        name = "/home/saiwanming/workdir/data/laion-high/"+value['name']+".jpg"
        org = value["org"]
        cogvlm = value["cogvlm"]

        wc = word_count_with_punctuation(org)
        if wc >70:
            neworg = change_desc_with_prompt(org)
            value["org_new"] = neworg
        else :
            value["org_new"] = org            

        wc = word_count_with_punctuation(cogvlm)
        if wc >70:
            neworg = change_desc_with_prompt(cogvlm)
            value["cogvlm_new"] = neworg
        else :
            value["cogvlm_new"] = cogvlm   
        value["cogvlm_new"] = cat_word_70(value["cogvlm_new"])
        value["org_new"] = cat_word_70(value["org_new"])        
        org = value["org_new"]
        cogvlm = value["cogvlm_new"]

        image = Image.open(name)
        try:
            inputs = processor(text=[org, cogvlm], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            org_probabilities = probs[0][0].item()
            cogvlm_probabilities = probs[0][1].item()
            org_similarity_score = logits_per_image[0][0].item()
            cogvlm_similarity_score = logits_per_image[0][1].item()
            if org_similarity_score> cogvlm_similarity_score:
                org_count += 1
            else:
                cogvlm_count  += 1
                
        except Exception as e:
            print(f"发生异常：{e}")
            print(f"cogvlm:{cogvlm}")
            print(f"org:{org}")       
            org_probabilities = cogvlm_probabilities =  org_similarity_score = cogvlm_similarity_score =0

        value["org_similarity_score"] = org_similarity_score
        value["cogvlm_similarity_score"] = cogvlm_similarity_score
        value["org_probabilities"] = org_probabilities
        value["cogvlm_probabilities"] = cogvlm_probabilities        
        total_org += org_probabilities
        total_cogvlm += cogvlm_probabilities
        total_org_similarity_score += org_similarity_score
        total_cogvlm_similarity_score += cogvlm_similarity_score              
        count += 1
        if count % 10 == 0 :
            # 使用 json.dump() 函数将数据写入文件
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)    

        print("total_count:"+ str(count))
        print("total_org_count:"+ str(org_count))
        print("total_cogvlm_count:"+ str(cogvlm_count))    
        print("total_org_probabilities:" + str(total_org))
        print("total_cogvlm_probabilities:"+str(total_cogvlm))    
        print("total_org_probabilities/count:" + str(total_org/count))
        print("total_cogvlm_probabilities/count:"+str(total_cogvlm/count))    

        print("total_org_similarity_score:" + str(total_org_similarity_score))
        print("total_cogvlm_similarity_score:"+str(total_cogvlm_similarity_score))    
        print("total_org_probabilities/count:" + str(total_org_similarity_score/count))
        print("total_cogvlm_similarity_score/count:"+str(total_cogvlm_similarity_score/count))  

    print("total_count:"+ str(count))
    print("total_org_count:"+ str(org_count))
    print("total_cogvlm_count:"+ str(cogvlm_count))    
    print("total_org_probabilities:" + str(total_org))
    print("total_cogvlm_probabilities:"+str(total_cogvlm))    
    print("total_org_probabilities/count:" + str(total_org/count))
    print("total_cogvlm_probabilities/count:"+str(total_cogvlm/count))    

    print("total_org_similarity_score:" + str(total_org_similarity_score))
    print("total_cogvlm_similarity_score:"+str(total_cogvlm_similarity_score))    
    print("total_org_similarity_score/count:" + str(total_org_similarity_score/count))
    print("total_cogvlm_similarity_score/count:"+str(total_cogvlm_similarity_score/count))        

    # 使用 json.dump() 函数将数据写入文件
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)             
if __name__ == "__main__":
    main()
    tr = torch.FloatTensor([[1, 2, 3]])


import json
from re import A
from PIL import Image
import requests
import torch
from llam2req import change_desc_with_prompt
from transformers import CLIPProcessor, CLIPModel
from wordutils import word_count_with_punctuation,cat_word_70
def main():
    file_path = "./data/data_score1.json"
    org_avg = 0
    org_total = 0    
    orggreater_total = 0
    orggreater_count = 0
    conggreater_total = 0
    cong_total = 0    
    conggreater_count = 0
    cong_avg = 0
    with open(file_path, 'r') as file:
        # 使用 read() 方法读取整个文件内容
        data = json.load(file)
    for value in data:
        org_similarity_score = value["org_similarity_score"] 
        cogvlm_similarity_score = value["cogvlm_similarity_score"]
        if org_similarity_score > cogvlm_similarity_score:
            orggreater = org_similarity_score - cogvlm_similarity_score
            orggreater_total += orggreater
            orggreater_count += 1
            org_total += org_similarity_score
        else:
            conggreater = cogvlm_similarity_score - org_similarity_score          
            conggreater_total += conggreater
            conggreater_count += 1
            cong_total += cogvlm_similarity_score
    orgbetter  = orggreater_total / orggreater_count
    cogbetter = conggreater_total / conggreater_count
    org_avg = org_total/ orggreater_count
    cong_avg = cong_total/ conggreater_count
    print(f"orgbetter {orgbetter}")
    print(f"cogbetter {cogbetter}")
    print(f"org_avg {org_avg}")
    print(f"cong_avg {cong_avg}")    
if __name__ == "__main__":
    main()    

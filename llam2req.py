import requests
import json

url = "http://localhost:3000/v1/llama2_7b/completions"

def change_desc(prompt):
    #print(f"req prompt:{prompt}")
    reqjson = {
        "prompt": prompt,
        "n": 1,
        "best_of": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": "string",
        "stop_token_ids": [],
        "ignore_eos": False,
        "max_tokens": 500,
        "logprobs": 0,
        "skip_special_tokens": True
    }
    payload = json.dumps(reqjson)
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res_json = json.loads(response.content.decode('utf-8'))
    response_txt:str = res_json['answer'][0]['outputs'][0]['text']
    index = response_txt.find(":")
    if index != -1:
        response_txt = response_txt[index:]
    #print(f" response: {response_txt}")
    response_txt = response_txt.replace("\n","")
    return response_txt
def change_desc_with_prompt(desc):
    prompt = """<s>[INST] <<SYS>>
        To summarize the image description, keep the following points in mind:
        Provide enough contextual information for the model to understand the key features of the image.
        Use clear, concise language to describe the image and avoid lengthy and unnecessary descriptions.
        Describe the elements in the image in a logical and spatial order so that the reader can easily follow the description and understand the image.
        Keep the description concise and no longer than 70 words.
        Do not write explanations on replies
        Image Description:
        <</SYS>>
        {{desc}}
        [/INST]
    """
    prompt = prompt.replace("{{desc}}", desc)
    return change_desc(prompt)    
if __name__ == "__main__":
    desc = "In this lounge, there are green sofas and brown chairs. On the left side of a large table, there is a man wearing a white shirt and black pants sitting on a green sofa with his legs crossed. Next to him, there is a woman wearing a dark red sweater and black pants standing. There are many small tables around them with books, cups, and other items placed on top. The walls are decorated with white tiles and posters. Above the room, there are wooden beams and hanging lights. Underneath the ceiling, there are pipes and air conditioning devices. To the right of the picture, there is an orange sofa and a reddish-brown coffee table. Throughout the space, there are also many people sitting or walking."
    change_desc(desc)
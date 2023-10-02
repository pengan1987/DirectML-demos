import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

import os
from multiprocessing import cpu_count
import torch_directml
dml = torch_directml.device()

torch.set_default_device(dml)

model_path = "D:\\phi_1_5"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)


tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

while (True):
    prompt_input = input("\r\n\r\nEnter your question: ")
    prompt_input += '''
Answer:'''

    start_time = time.perf_counter()
    inputs = tokenizer(prompt_input, return_tensors="pt",
                       return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    end_time = time.perf_counter()

    print(text)
    execution_time = end_time - start_time
    # It returns time in seconds
    print("Executed in "+str(execution_time)+" seconds")

# !pip install "transformers==4.26.0"
# !pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade


# scp -r winter@192.148.247.180:/users/PAS2348/winter/Projects/Taskbot_Challenge/TacoQA_Alt/flan-t5-xl/logs/ .
# tensorboard --logdir .\logs\


from transformers import (AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,AutoTokenizer,set_seed)

model_name = "google/flan-t5-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained("/users/PAS2348/winter/Projects/Taskbot_Challenge/TacoQA_Alt/flan-t5-xxl/checkpoint-1914", device_map="auto", load_in_8bit=True, return_dict=True, max_memory={"cuda:0": "20GB", "cpu": "20GB"})

sent = "What can i cook today?"
inputs = tokenizer.encode(sent, return_tensors="pt")

generated_ids = model.generate(inputs, max_length=100)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
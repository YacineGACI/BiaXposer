import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

model_name = "bert-base-uncased"
model_name = "deepset/roberta-base-squad2"
# model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = "He bought a table."
question = "What did he buy?"

question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."

input = tokenizer(question, context)
input = {k: torch.tensor(v).unsqueeze(0) for k, v in input.items()}

print(input)

output = model(**input)

print(output)


start_logits = output.start_logits
end_logits = output.end_logits

start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits)

print(start_index, end_index)

answer_ids = input["input_ids"].squeeze()[start_index:end_index + 1].tolist()
print(answer_ids)

answer = ""
for i in answer_ids:
    answer += tokenizer.convert_ids_to_tokens(i) + " "

print(answer)

# qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
# question = "Where do I live?"
# context = "My name is Merve and I live in İstanbul."
output = qa_model(question = question, context = context)
print(output)

print(tokenizer.decode(answer_ids))
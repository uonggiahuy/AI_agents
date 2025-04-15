import lmstudio as lms

model = lms.llm("llama-3.2-1b-instruct")
result = model.respond("Hiệu trưởng của UET-VNU là ai ?")

print(result)

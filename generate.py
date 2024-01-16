import openai

# 设置API凭据
openai.api_key = 'sk-...E7Vz'

# 创建输入文本
input_text = "cat"

# 使用GPT-3模型生成句子
response = openai.Completion.create(
  engine="davinci",  # 替换为您所使用的GPT-3模型引擎
  prompt=input_text,
  max_tokens=50,  # 可根据需要调整生成句子的最大长度
  temperature=0.7,  # 控制生成句子的多样性
  n=1  # 生成一个句子
)

# 解码生成的句子
generated_sentence = response.choices[0].text.strip()

print("生成的句子：", generated_sentence)
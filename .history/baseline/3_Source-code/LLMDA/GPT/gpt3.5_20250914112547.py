import openai
import json

# 设置您的 OpenAI API 密钥
openai.api_key = ''

# 打开 JSON 文件并加载数据
with open('../data/test.json', 'r') as f:
    data = json.load(f)
data= data[:1000]

# 用于存储预测结果的列表
predictions = []

# 对每个条目进行预测
for entry in data:
    # 提取文本
    text = entry['text']
    
    # 使用 OpenAI API 发送请求进行预测
    response = openai.Completion.create(
      model="gpt-3.5-turbo-16k",  # GPT-3.5 模型的名称
      prompt=text,
      temperature=0,  # 用于控制生成文本多样性的参数
      max_tokens=1,   # 控制生成文本长度的参数
      stop=["\n"]     # 设置停止词，确保只生成一行文本
    )
    
    # 解析 API 响应并提取预测结果
    predicted_label = response.choices[0].text.strip()
    
    # 存储预测结果
    predictions.append(predicted_label)

# 将预测结果写入到文件中
with open('predict.txt', 'w') as f:
    for prediction in predictions:
        f.write(prediction + "\n")

import json
import random


def read_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_folder(folder_name):
    texts = read_file(f'{folder_name}/patches.txt')
    descriptions = read_file(f'{folder_name}/descriptions.txt')
    projects = read_file(f'{folder_name}/names.txt')
    times = read_file(f'{folder_name}/times.txt')
    labels = read_file(f'{folder_name}/types.txt')
    
    result = []
    for text, description, project, time, label in zip(texts, descriptions, projects, times, labels):
        entry = {
            "text": text,
            "name": folder_name,
            "description": description,
            "project": project,
            "time": time,
            "label": label
        }
        result.append(entry)
    
    return result

non_security_data = process_folder('non_security')
security_data = process_folder('security')


# data statistics for distribution of labels in security
from collections import defaultdict
label_distribution = defaultdict(int)
total_entries = len(security_data)
for entry in security_data:
    label = entry['label']
    label_distribution[label] += 1
# 打印统计分布以及占比
print("Label Distribution in 'security':")
for label, count in label_distribution.items():
    percentage = (count / total_entries) * 100
    print(f"{label}: {count} ({percentage:.2f}%)")





final_data = non_security_data + security_data

with open('output.json', 'w') as f:
    json.dump(final_data, f, indent=4)

# 读取output.json中的数据
with open('output.json', 'r') as f:
    data = json.load(f)

# 打乱数据，确保分割是随机的
random.shuffle(data)

# 获取数据的总数
total = len(data)

# 计算训练数据和测试数据的条目数
train_count = int(0.8 * total)
test_count = total - train_count

# 分割数据
train_data = data[:train_count]
test_data = data[train_count:]

# 将训练数据保存到train.json
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=4)

# 将测试数据保存到test.json
with open('test.json', 'w') as f:
    json.dump(test_data, f, indent=4)
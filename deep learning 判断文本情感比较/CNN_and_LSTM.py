# -*- coding: utf-8 -*-
"""
文本情绪分类完整代码（PyTorch版）
功能：训练CNN/LSTM模型对文本进行正面/中性/负面三分类
"""
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行导入
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#=================== 0. display =======================
# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 查看数据
print(train_data.head())
print(f"训练集大小: {train_data.shape}")
print(f"测试集大小: {test_data.shape}")

# 查看情感分布
print(train_data['sentiment'].value_counts())

# ==================== 1. 数据预处理 ====================
def clean_text(text):
    """文本清洗函数（核心步骤）"""
    if not isinstance(text, str):
        text = str(text)  # 确保输入为字符串
    # 移除特殊字符（保留字母和空格）
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)  # 去URL/标签
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()  # 统一小写
    # 分词与词形还原
    tokens = [WordNetLemmatizer().lemmatize(word)
              for word in text.split()
              if word not in set(stopwords.words('english'))]  # 去停用词
    return ' '.join(tokens)

# 🎯 加载数据（假设CSV包含text和sentiment列）
train_data = pd.read_csv('train.csv').dropna(subset=['text', 'sentiment'])
test_data = pd.read_csv('test.csv').dropna(subset=['text', 'sentiment'])

# 🎯 应用清洗（训练集和测试集独立处理）
train_data['cleaned_text'] = train_data['text'].apply(clean_text)
test_data['cleaned_text'] = test_data['text'].apply(clean_text)

# ==================== 2. 构建词汇表 ====================
def build_vocab(texts, max_size=10000):
    """根据训练集构建词汇表（测试集不参与）"""
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    # 按词频排序，取前max_size个词
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words[:max_size])}
    vocab["<pad>"] = 0  # 填充标记
    vocab["<unk>"] = 1  # 未知词标记
    return vocab

# 🎯 仅用训练集构建词汇表（防止数据泄露）
vocab = build_vocab(train_data['cleaned_text'])
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

# ==================== 3. 数据编码与加载 ====================
class TextDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts  # 文本数据（pandas Series）
        self.labels = labels  # 标签（numpy数组）
        self.vocab = vocab  # 词汇表字典
        self.max_len = max_len  # 统一序列长度

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 🎯 核心操作：文本→数字序列→长度标准化
        text = self.texts.iloc[idx]
        # 单词转索引（未知词用<unk>）
        sequence = [self.vocab.get(word, self.vocab["<unk>"])
                    for word in text.split()]
        # 统一长度（截断或填充）
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [self.vocab["<pad>"]] * (self.max_len - len(sequence))
        return torch.tensor(sequence), torch.tensor(self.labels[idx])


# 🎯 标签编码（正面/中性/负面 → 0/1/2）
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['sentiment'])
test_labels = label_encoder.transform(test_data['sentiment'])

# 🎯 创建Dataset和DataLoader
train_dataset = TextDataset(train_data['cleaned_text'], train_labels, vocab)
test_dataset = TextDataset(test_data['cleaned_text'], test_labels, vocab)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# ==================== 4. 模型定义 ====================
class EmotionCNN(nn.Module):
    """CNN模型结构（适合捕捉局部特征）"""
    def __init__(self, vocab_size, embed_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, 5, padding=2)  # 卷积核大小5
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 64, 3, padding=1)  # 卷积核大小3
        self.pool = nn.AdaptiveMaxPool1d(1)  # 全局最大池化层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 数据流动：嵌入层 → 卷积 → 池化 → 全连接
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # 转为[batch, embed_dim, seq_len]适应卷积
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)  # 使用定义好的池化层
        return self.fc(x)


class EmotionLSTM(nn.Module):
    """LSTM模型结构（适合序列建模）"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           batch_first=True,
                           bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.hidden_dim = hidden_dim  # 关键修复：保存为类属性

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # 输出形状: [batch, seq_len, hidden_dim*2]
        # 双向LSTM输出处理
        forward_last = lstm_out[:, -1, :self.hidden_dim]  # 正向最后时间步
        backward_first = lstm_out[:, 0, self.hidden_dim:]  # 反向第一个时间步
        x = torch.cat([forward_last, backward_first], dim=1)
        x = self.dropout(x)
        return self.fc(x)


# ==================== 5. 训练与评估 ====================
def train_and_evaluate(model, model_name, train_loader, test_loader, epochs=15):
    """训练流程（含验证和模型保存）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 🎯 训练阶段（使用训练集）
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 🎯 验证阶段（使用测试集）
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(val_acc)

        # 🎯 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')

        print(
            f'{model_name} Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title(f'{model_name} Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} Validation Accuracy')
    plt.legend()
    plt.savefig(f'{model_name}_training.png')
    plt.show()

    return best_acc


def full_evaluate(model, test_loader, label_encoder):
    """详细评估（输出分类报告）"""
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=label_encoder.classes_))


# ==================== 6. 执行训练 ====================
print("\n训练CNN模型...")
cnn_model = EmotionCNN(vocab_size)
cnn_acc = train_and_evaluate(cnn_model, "CNN", train_loader, test_loader)

print("\n训练LSTM模型...")
lstm_model = EmotionLSTM(vocab_size)
lstm_acc = train_and_evaluate(lstm_model, "LSTM", train_loader, test_loader)

# ==================== 7. 结果对比 ====================
print("\n模型性能对比:")
print(f"CNN最佳验证准确率: {cnn_acc:.4f}")
print(f"LSTM最佳验证准确率: {lstm_acc:.4f}")

print("\nCNN测试集详细评估:")
cnn_model.load_state_dict(torch.load('best_CNN_model.pth'))
full_evaluate(cnn_model, test_loader, label_encoder)

print("\nLSTM测试集详细评估:")
lstm_model.load_state_dict(torch.load('best_LSTM_model.pth'))
full_evaluate(lstm_model, test_loader, label_encoder)


# ==================== 8. 预测示例 ====================
def predict_emotion(text, model, vocab, max_len=100):
    """单条文本预测函数"""
    model.eval()
    text = clean_text(text)
    sequence = [vocab.get(word, vocab["<unk>"]) for word in text.split()]
    sequence = sequence[:max_len] + [0] * (max_len - len(sequence))
    tensor = torch.tensor(sequence).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor.to(next(model.parameters()).device))
    return label_encoder.classes_[torch.argmax(output).item()]


test_samples = [
    "I'm so happy with this product!",
    "This is neither good nor bad",
    "Terrible experience, never buy again"
]

print("\n预测示例:")
for text in test_samples:
    cnn_pred = predict_emotion(text, cnn_model, vocab)
    lstm_pred = predict_emotion(text, lstm_model, vocab)
    print(f"文本: {text[:30]}...")
    print(f"  CNN预测: {cnn_pred:<8} LSTM预测: {lstm_pred}")
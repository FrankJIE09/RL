# Reinforcement Learning: An Introduction 转换文件

## 📁 文件夹内容

- `Reinforcement_Learning_An_Introduction.txt` - PDF转换后的TXT文本文件（1.6 MB）
- `rl_book_pdf_to_txt.py` - PDF转TXT转换脚本

## 📖 书籍信息

- **书名**: Reinforcement Learning: An Introduction (Second Edition)
- **作者**: Richard S. Sutton, Andrew G. Barto
- **出版社**: The MIT Press
- **原PDF文件**: `Reinforcement Learning An Introduction (Adaptive Computation and Machine Learning series) (Sutton, Richard S., Barto, Andrew G.) (Z-Library).pdf`
- **总页数**: 548页

## 📝 转换信息

- **转换时间**: 2024-12-01
- **转换方法**: PyMuPDF (fitz)
- **输出格式**: UTF-8编码的TXT文件
- **文件大小**: 1.6 MB
- **总行数**: 31,551行

## 🔍 文件结构

TXT文件包含：
- 每页都有页码标记（Page 1, Page 2, ...）
- 完整的书籍内容（封面、目录、正文、参考文献）
- 章节标题和内容

## 💡 使用建议

### 搜索内容
```bash
# 搜索关键词
grep -n "value function" Reinforcement_Learning_An_Introduction.txt

# 搜索特定章节
grep -A 50 "Chapter 3" Reinforcement_Learning_An_Introduction.txt
```

### 重新转换
如果需要重新转换，可以运行：
```bash
python3 rl_book_pdf_to_txt.py
```

## 📚 相关文件

原PDF文件位于上级目录：
```
../Reinforcement Learning An Introduction (Adaptive Computation and Machine Learning series) (Sutton, Richard S., Barto, Andrew G.) (Z-Library).pdf
```


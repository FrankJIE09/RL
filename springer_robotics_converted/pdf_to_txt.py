#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF转TXT工具脚本
将Springer Handbook of Robotics.pdf转换为txt格式
"""

import sys
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    print("正在安装 PyMuPDF...")
    os.system("pip install PyMuPDF -q")
    import fitz

def pdf_to_txt(pdf_path, txt_path):
    """
    将PDF文件转换为TXT文件
    
    Args:
        pdf_path: PDF文件路径
        txt_path: 输出TXT文件路径
    """
    print(f"正在读取PDF文件: {pdf_path}")
    
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF总页数: {total_pages}")
        
        # 提取所有页面的文本
        text_content = []
        for page_num in range(total_pages):
            if (page_num + 1) % 100 == 0:
                print(f"正在处理第 {page_num + 1}/{total_pages} 页...")
            
            page = doc[page_num]
            text = page.get_text()
            text_content.append(f"\n\n{'='*80}\n")
            text_content.append(f"第 {page_num + 1} 页\n")
            text_content.append(f"{'='*80}\n\n")
            text_content.append(text)
        
        # 保存为TXT文件
        print(f"正在保存TXT文件: {txt_path}")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(''.join(text_content))
        
        doc.close()
        print(f"转换完成！文件已保存到: {txt_path}")
        
        # 显示文件大小
        file_size = os.path.getsize(txt_path)
        print(f"输出文件大小: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    pdf_path = os.path.join(parent_dir, "Springer Handbook of Robotics.pdf")
    txt_path = os.path.join(script_dir, "Springer_Handbook_of_Robotics.txt")
    
    # 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到PDF文件: {pdf_path}")
        sys.exit(1)
    
    # 执行转换
    pdf_to_txt(pdf_path, txt_path)


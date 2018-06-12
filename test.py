import sys
import os
import re
import numpy as np
from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load("/home/zzj/ltp_data_v3.4.0/cws.model")
dir = './tokenized_dir/'
filelist = os.listdir(dir)
abstract_avr_len= 0
article_avr_len = 0
coun = 0

# def clear_str(line):
#   emoji_pattern = re.compile(
#     u"(\ud83d[\ude00-\ude4f])|"  # emoticons
#     u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
#     u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
#     u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
#     u"(\ud83c[\udde0-\uddff])|"
#     u"[\U00010000-\U0010ffff]"  # flags (iOS)
#     "+", flags=re.UNICODE)
#
#   return emoji_pattern.sub(r'', line)
# def tokenize_stories(stories_dir,file):
#     """将weibo文件夹下对应文件分词，存储在tokenized_stories_dir/下一个个文件"""
#
#     num = 0
#     stories = []
#     stories.append(file)
#     for file in stories:
#         path = stories_dir + '/' + file
#         print('token processing:', path)
#         with open(path, 'r', encoding='utf-8')as f:
#             print('%s starting!' % (file))
#             line = f.readline()
#             while line:
#                 line = clear_str(line)
#                 coun = line.count('ishot')
#                 if coun > 0:
#                     try:
#                         line_dic = eval(line)
#                         wb_content = line_dic['wb_content']
#                         if wb_content[0] == ':': wb_content = wb_content[1:]
#
#                         content_list = re.split('[。]', wb_content)
#                         if (len(content_list) >= 2):
#                             if "的秒拍视频" in content_list[-1] or "http" in content_list[-1]: content_list.pop()
#                             if re.match(r'（[^）]*\）', content_list[-1]): content_list.pop()
#                             if re.match(r'\([^)]*\)', content_list[-1]): content_list.pop()
#                             if re.match(r'@', content_list[-1]): content_list.pop()
#
#                         wb_content = '。'.join(content_list)
#                         patterns = [r'【.+】', u'（长微博）', r'@[^\s 】。.！’”"]*', r'[^!！?？.。，,\s]*的秒拍视频', r'（[^）]{1,6}\）',
#                                     r'\([^)]{1,6}\)', r'\[.{1,6}]'
#                             , r'http://t.cn/[0-9a-zA-Z]*',u"\u200B"]
#                         for pattern in patterns[1:]:
#                             wb_content = re.sub(pattern, ' ', wb_content)
#
#                         abstract = ''
#                         st = wb_content.find('【')
#                         en = wb_content.find('】')
#                         if st != -1 and en != -1:
#                             for i in range(st + 1, en):
#                                 abstract += wb_content[i]
#
#
#                         newpath ='./' + file
#                         abstract = ' '.join(segmentor.segment(abstract))
#
#                         length = len(abstract)
#
#                         wb_content = re.sub(patterns[0], '', wb_content)
#                         length_c = len(wb_content)
#
#                         content = ' '.join(segmentor.segment(wb_content))
#                         if (length > 4 and length_c > 50 and content != '' and abstract != ''):
#                             num += 1
#                             with open(newpath, 'a', encoding='utf-8')as wn:
#                                 print(abstract, file=wn)
#                                 print(content, file=wn)
#                     except:
#                         print('error', line)
#
#                 else:
#                     pass
#                     # print('word2vec ing ')
#                     # try:
#                     #   line_dic = eval(line)
#                     #   wb_content = line_dic['wb_content']
#                     #   if wb_content[0] == ':': wb_content = wb_content[1:]
#                     #   patterns = [r'【', r'】', r'@[^\s 】。.！’”"]*', r'[^!！?？.。，,\s]*的秒拍视频', r'（[^）]{1,6}\）',
#                     #               r'\([^)]{1,6}\)', r'\[.{1,6}]'
#                     #     , r'http://t.cn/[0-9a-zA-Z]*']
#                     #   for pattern in patterns:
#                     #     wb_content = re.sub(pattern, ' ', wb_content)
#                     #   content = ' '.join(segmentor.segment(wb_content))
#                     #   with open('./trainForword2vec.txt','a+',encoding='utf-8')as fw:
#                     #     print(content,file=fw)
#                     # except:
#                     #   print('error',line)
#
#                 line = f.readline()
#             print("%s finished" % (file))
#     print('total pairs:', num)
def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            pattern = u"\u200B"
            line = re.sub(pattern, "", line)
            lines.append(line.strip())
    return lines
def fun():
  for file in filelist:
    path = dir+"/"+file
    lines = read_text_file(path)
    length = len(lines)
    for i in range(0, length - 1, 2):
        abstract = lines[i]
        abstract = abstract.strip()

        if(abstract=="" or abstract=="\n" or len(abstract)<5):
            print("abstract,",file,i,len(abstract),abstract)
            print("following art",lines[i+1])
        article = lines[i + 1].strip()
        # if(article=="" or article=="\n"or len(article)<50):
        #     print("article",article)
        if(	u"\u200B" in abstract or u"\u200B" in article):
            print("u200B")

if __name__ == '__main__':
    story_dir = './weibo'
    fun()
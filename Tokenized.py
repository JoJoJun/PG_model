"""
将爬下的源数据进行分词，按文件存储到tokenized_dir下面
格式：
<s> 摘要 </s>
一行微博内容
去掉图片链接、emoji、[赞]这样的表情、（。。报），（记者）、xxx秒拍视频等正文无关内容
只记录摘要长度>4的
"""
import sys
import os
import re
from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load("/home/zzj/ltp_data_v3.4.0/cws.model")


tokenized_stories_dir = "tokenized_dir"

# 过滤掉emoji
def clear_str(line):
  emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])|"
    u"[\U00010000-\U0010ffff]"  # flags (iOS)
    "+", flags=re.UNICODE)

  return emoji_pattern.sub(r'', line)

#stories_dir = 'weibo'  tokenized_story_dir="tokenized_dir"
def tokenize_stories(stories_dir, tokenized_stories_dir):
  """将weibo文件夹下对应文件分词，存储在tokenized_stories_dir/下一个个文件"""

  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  abstractlen=[]
  for i in range(0,100):abstractlen.append(0)
  num = 0
  for file in stories:
    path = stories_dir+'/'+file
    print('token processing:',path)
    with open(path, 'r', encoding='utf-8')as f:
      print('%s starting!' % (file))
      line = f.readline()
      while  line:
        line = clear_str(line)
        coun = line.count('ishot')
        if coun>0:
          try:
            line_dic = eval(line)
            wb_content = line_dic['wb_content']
            if wb_content[0] == ':': wb_content = wb_content[1:]

            content_list = re.split('[。]', wb_content)
            if (len(content_list) >= 2):
              if "的秒拍视频" in content_list[-1] or "http" in content_list[-1]: content_list.pop()
              if re.match(r'（[^）]*\）', content_list[-1]): content_list.pop()
              if re.match(r'\([^)]*\)', content_list[-1]): content_list.pop()
              if re.match(r'@', content_list[-1]): content_list.pop()

            wb_content = '。'.join(content_list)
            patterns = [r'【.+】', u'（长微博）', r'@[^\s 】。.！’”"]*', r'[^!！?？.。，,\s]*的秒拍视频', r'（[^）]{1,6}\）',
                        r'\([^)]{1,6}\)', r'\[.{1,6}]'
              , r'http://t.cn/[0-9a-zA-Z]*',u"\u200B"]
            for pattern in patterns[1:]:
              wb_content = re.sub(pattern, ' ', wb_content)


            abstract = ''
            st = wb_content.find('【')
            en = wb_content.find('】')
            if st != -1 and en != -1:
              for i in range(st + 1, en):
                abstract += wb_content[i]

            abstractlen[en-st-1]+=1

            newpath= os.path.join(tokenized_stories_dir, file)
            abstract = ' '.join(segmentor.segment(abstract))

            length = len(abstract)

            wb_content = re.sub(patterns[0], '', wb_content)
            length_c = len(wb_content)




            content = ' '.join(segmentor.segment(wb_content))
            if(length>4 and length_c>50 and content!='' and abstract !=''):
              num +=1
              with open(newpath,'a',encoding='utf-8')as wn:
                print(abstract,file=wn)
                print(content,file=wn)
          except:
            print('error',line)

        else:
          pass
          # print('word2vec ing ')
          # try:
          #   line_dic = eval(line)
          #   wb_content = line_dic['wb_content']
          #   if wb_content[0] == ':': wb_content = wb_content[1:]
          #   patterns = [r'【', r'】', r'@[^\s 】。.！’”"]*', r'[^!！?？.。，,\s]*的秒拍视频', r'（[^）]{1,6}\）',
          #               r'\([^)]{1,6}\)', r'\[.{1,6}]'
          #     , r'http://t.cn/[0-9a-zA-Z]*']
          #   for pattern in patterns:
          #     wb_content = re.sub(pattern, ' ', wb_content)
          #   content = ' '.join(segmentor.segment(wb_content))
          #   with open('./trainForword2vec.txt','a+',encoding='utf-8')as fw:
          #     print(content,file=fw)
          # except:
          #   print('error',line)

        line = f.readline()
      print("%s finished" % (file))
  print('total pairs:',num)
if __name__ == '__main__':
  stories_dir = 'weibo'


  if not os.path.exists(tokenized_stories_dir): os.makedirs(tokenized_stories_dir)

  tokenize_stories(stories_dir, tokenized_stories_dir)

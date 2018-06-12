'''
不包含评论处理！！！仅仅能处理一行<s>开头的摘要和一行微博的！
'''
import sys
import os
import struct
import collections
import re
from tensorflow.core.example import example_pb2
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = ['1068246700.txt','1267454277.txt','1301904252.txt','1314608344.txt','1323527941.txt','1402977920.txt',
                  '1496814565.txt','1497087080.txt','1618051664.txt',"1642088277.txt","1642512402.txt","1642585887.txt","1644114654.txt",
                  "1644119190.txt","1644358851.txt","1644948230.txt","1647688972.txt","1649159940.txt","1649173367.txt","1649597805.txt"] # 95580+77247+47250+
all_val_urls =["1641532820.txt"] # 16042
all_test_urls = ['1578839965.txt','1610362247.txt','1642471052.txt','1611935011.txt',"1641561812.txt",]  #1456+1249+ 2590+38+1598=

cnn_tokenized_stories_dir = "tokenized_dir"

finished_files_dir = "./trainWithEmbedding/finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 100000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = './trainWithEmbedding/finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)


def read_text_file(text_file):
    lines = []

    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    # url_file 为weibo下的文件list
    print("Making bin file for URLs listed  ")

    story_fnames = ['tokenized_dir/' + s for s in url_file]

    # if makevocab:
    #     vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            print('write2bin processing %s ' % s)

            article_list = []
            abstract_list = []
            with open(s, 'r', encoding='utf-8')as readingfile:
                lines = read_text_file(s)
                length = len(lines)
                for i in range(0, length - 1, 2):
                    if i + 1 < length:
                        abstract = lines[i]
                        abstract = abstract.lstrip('<s> ')
                        abstract = abstract.rstrip('</s>')
                        abstract = abstract.strip()
                        article = lines[i + 1].strip()
                        article_list.append(article)
                        abstract_list.append(abstract)
                        tf_example = example_pb2.Example()
                        f = open('check.txt',"a+",encoding='utf-8')
                        print(abstract,file=f)
                        print(article,file=f)
                        f.close()
                        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                        tf_example_str = tf_example.SerializeToString()
                        str_len = len(tf_example_str)
                        writer.write(struct.pack('q', str_len))
                        writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # article = ' '.join(article_list)
                # abstract = ' '.join(abstract_list)
                #
                # abstract = re.sub('<s>', '', abstract)
                # abstract = re.sub('</s>', '', abstract)
                # # Write the vocab to file, if applicable
                # if makevocab:
                #     art_tokens = article.split(' ')
                #     abs_tokens = abstract.split(' ')
                #     tokens = art_tokens + abs_tokens
                #     tokens = [t.strip() for t in tokens]  # strip
                #     tokens = [t for t in tokens if t != "" and t!="<s>" and t!="</s>"]  # remove empty
                #     print('updating vocab')
                #     vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # # write vocab to file
    # if makevocab:
    #     print("Writing vocab file...")
    #     with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
    #         for word, count in vocab_counter.most_common(VOCAB_SIZE):
    #             writer.write(word + ' ' + str(count) + '\n')
    #     print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)


    write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
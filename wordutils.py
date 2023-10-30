import string
import re

def word_count_with_punctuation(text):

    # 使用 translate() 方法删除标点符号
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)

    # 使用 split() 函数将文本拆分为单词
    words = text_without_punctuation.split()

    # 统计单词和标点符号的数量
    word_count = len(words)
    punctuation_count = len(text) - len(text_without_punctuation)

    return word_count+ punctuation_count

def cat_word_70(text):
    # 使用正则表达式将字符串拆分为单词、空格和标点符号
    tokens = re.findall(r'\b\w+\b|[\s.,;!?"]', text)

    # 保留前70个单词、空格和标点符号
    result = ''.join(tokens[:70])
    return result

if __name__ == "__main__":

    # 定义一个包含文本的字符串
    text = """
The image on the cover features a black font with the text "Yeni Safak" in white above and below it. A blue oval containing the words "PKK/PGY" is located above the font. A person in a red jersey and holding up both hands is on the left side of the cover, with the word "FIRSAT JETET" above them. In the upper right corner, a woman in a black jacket and gray shirt is shown with yellow hair. Below her, there is an announcement about ZİVESİ BASLAR. White lines and numbers, including 123, 45, 67, 81, 109, 112, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123
    """
    print(word_count_with_punctuation(text))
    print(cat_word_70(text))
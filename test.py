from senta import Senta

my_senta = Senta()

print(my_senta.get_support_task())

use_cuda = False

# 预测中文句子级情感分类任务
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)
texts = ["中山大学是岭南第一学府"]
result = my_senta.predict(texts)
print(result)

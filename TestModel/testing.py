# # from transformers import AutoTokenizer, AutoModel
# # from sklearn.metrics.pairwise import cosine_similarity
# # import torch
# #
# # # 用户爱好、事件名称和事件描述
# # user_hobby = "computer"
# # event_names = ["soccer pick up", "car race", "football pickup game", "basketball tournament", "AI tech meetup"]
# # event_descriptions = [
# #     "A casual soccer pick-up game for enthusiasts.",
# #     "Exciting car racing event for adrenaline junkies.",
# #     "Join us for a friendly football pickup game.",
# #     "Basketball tournament for skilled players.",
# #     "AI tech meetup to discuss the latest trends."
# # ]
# #
# # # 将用户爱好、事件名称和事件描述组合成一个列表
# # combined_texts = [user_hobby] + event_names + event_descriptions
# #
# # # 加载BERT模型和tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # model = AutoModel.from_pretrained("bert-base-uncased")
# #
# # # 处理用户输入和事件描述
# # tokenized_texts = tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True)
# #
# # # 生成文本嵌入
# # with torch.no_grad():
# #     embeddings = model(**tokenized_texts)["last_hidden_state"][:, 0, :]
# #
# # # 计算相似度
# # user_hobby_embedding = embeddings[0]
# # event_embeddings = embeddings[1:]
# #
# # # 计算与事件名称的相似度
# # name_similarity_scores = cosine_similarity(user_hobby_embedding.unsqueeze(0), event_embeddings[:len(event_names)])
# #
# # # 计算与事件描述的相似度
# # description_similarity_scores = cosine_similarity(user_hobby_embedding.unsqueeze(0), event_embeddings[len(event_names):])
# #
# # # 合并相似度得分
# # combined_similarity_scores = name_similarity_scores + description_similarity_scores
# #
# # # 排序和输出
# # sorted_events = [(event, score) for event, score in zip(event_names + event_descriptions, combined_similarity_scores[0])]
# # sorted_events = sorted(sorted_events, key=lambda x: x[1], reverse=True)
# #
# # # 打印排序后的事件
# # for event, score in sorted_events:
# #     print(f"Event: {event}, Similarity Score: {score:.4f}")
# import os
#
from semantic_text_similarity.models import WebBertSimilarity
import torch

event_ids =[1,2,3,4,5]
user_hobby = "soccer"
event_names = ["soccer pick up", "car race", "football pickup game", "basketball tournament", "AI tech discuss"]
event_descriptions = [
    "A casual soccer pick-up game for enthusiasts.",
    "Exciting car racing event for adrenaline junkies.",
    "Join us for a friendly football pickup game.",
    "Basketball tournament for skilled players.",
    "Dive into the fascinating world of artificial intelligence! This event brings together tech enthusiasts and "
    "experts for an insightful discussion on the latest AI trends, breakthroughs, and future possibilities. Explore "
    "the limitless potential of AI and its impact on various industries."
]
event_categories = ["Sport", "Sport", "Sport", "Sport", "Technology"]

combined_texts = [user_hobby] + event_names + event_descriptions + event_categories

sentence_pairs = [(user_hobby, text) for text in combined_texts]
# 加载WebBertSimilarity模型
model_path = 'C:\sjsu\CS156\WebProject\WebProject\webBert'
# Load the WebBertSimilarity model from the specified path
similarity_model = WebBertSimilarity(model_name=model_path, device='cpu')
# similarity_model = WebBertSimilarity(device='cuda')  # Use 'cpu' if you don't have a GPU

# 计算相似度
user_hobby_embedding = similarity_model.predict([(user_hobby, user_hobby)])[0]
event_embeddings = similarity_model.predict(sentence_pairs[1:])

# 计算与事件名称的相似度
name_similarity_scores = event_embeddings[:len(event_names)]

# 计算与事件描述的相似度
description_similarity_scores = event_embeddings[len(event_names):2*len(event_names)]

cate_similarity_scores = event_embeddings[2*len(event_names):]
# 合并相似度得分
combined_similarity_scores = name_similarity_scores + description_similarity_scores + cate_similarity_scores

# 排序和输出
sorted_events = [(event_id, score) for event_id, score in
                     zip(event_ids, combined_similarity_scores)]
sorted_events = sorted(sorted_events, key=lambda x: x[1], reverse=True)

# 打印排序后的事件
for event, score in sorted_events:
    print(f"Event: {event}, Similarity Score: {score:.4f}")





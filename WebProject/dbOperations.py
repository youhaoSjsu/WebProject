from datetime import datetime, timedelta

from django.db import connection
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from semantic_text_similarity.models import WebBertSimilarity

from TestModel.models import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
# from datetime import *
import random
from faker import Faker
from TestModel.serializers import EventSerializer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

fake = Faker()
import spacy

nlp = spacy.load("en_core_web_sm")


## db operation 1
def testdb(request):
    retrieved_user = User.objects.get(username='user2')
    user_profile = UserProfile.objects.create(
        user=retrieved_user,
        age=25,
        job='Software Engineer',
        gender='M',
    )

    return HttpResponse("success")


def generate_random_datetime(start_date, end_date):
    return fake.date_time_between(start_date=start_date, end_date=end_date, tzinfo=timezone.utc)


def fakeData(request):
    categories = ['Sports', 'Music', 'Technology', 'Video Games', 'Cars']
    ## create hobby category
    for category_name in categories:
        HobbyCate.objects.create(name=category_name)

    # Create 10 fake events
    for _ in range(10):
        event = Event(
            name=fake.name(),
            category=random.choice(categories),
            time=generate_random_datetime(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=365)),
            location=fake.address(),
        )
        event.save()

    return HttpResponse("fakedDate")


def imitationRating(request):
    for i in range(1, 9):
        user_id = i
        user = User.objects.get(user_id=i)
        events = makePush(user)
        for event in events:
            rating = fakeRatingBySimilarities(event.event_id, user_id)

            pr = PushRate(
                event_id=event.event_id,
                user_id=user_id,
                rate=rating,

            )
            pr.save()
    return HttpResponse("fakedData")


def fakeRating(request):
    for i in range(100):
        user_ids = np.random.choice(range(1, 9))
        event_ids = np.random.choice(range(1, 31))
        ratings = fakeRatingBySimilarities(event_ids, user_ids)

        rate_ids = i + 1

        pr = PushRate(
            event_id=event_ids,
            user_id=user_ids,
            rate=ratings,
            rate_id=rate_ids
        )
        pr.save()
        print("saved rate_id" + str(rate_ids))

    return HttpResponse("fakedData")


def fakeRatingBySimilarities(event_id, user_id):
    user_hobby_Tuple = execute_custom_sql(
        "select testmodel_hobby.name from testmodel_userhobby left join testmodel_hobby on " +
        "testmodel_userhobby.hobby_id =testmodel_hobby.hobby_id " +
        "where user_id = %s;", [user_id])
    print(user_hobby_Tuple)
    user_hobbies_token = ''
    user_hobbies_list = []
    for hobby in user_hobby_Tuple:
        user_hobbies_list.append(hobby[0])
    event = Event.objects.get(event_id=event_id)
    events = [event]
    total_point = 0.0
    poss_rate = 0
    for h in user_hobbies_list:
        event_simi_pair_list = match_similarities(h, events)
        ## only one rate
        event_simi_pair = event_simi_pair_list[0]
        total_point += event_simi_pair[1]

    ## get avg point of hobbies event similarity
    avg_point = total_point / len(user_hobbies_list)

    if avg_point > 6.00:
        hobby_simi = np.random.choice(range(5, 6))
    elif 5.0 <= avg_point < 6.00:
        hobby_simi = np.random.choice(range(4, 6))
    elif 4.00 <= avg_point < 5.00:
        hobby_simi = np.random.choice(range(4, 5))
    elif 3.50 <= avg_point < 4.00:
        hobby_simi = np.random.choice(range(3, 6))
    elif 3.00 <= avg_point < 3.50:
        hobby_simi = np.random.choice(range(3, 5))
    elif 2.50 <= avg_point < 3.00:
        hobby_simi = np.random.choice(range(2, 5))
    elif 2.00 <= avg_point < 2.5:
        hobby_simi = np.random.choice(range(2, 4))
    elif 1.50 <= avg_point < 2.00:
        hobby_simi = np.random.choice(range(2, 3))
    elif 0.50 <= avg_point < 1.5:
        hobby_simi = np.random.choice(range(1, 3))
    else:
        hobby_simi = np.random.choice(range(1, 2))
    print("similarity= " + str(avg_point) + " rate= " + str(hobby_simi))

    return hobby_simi


def execute_custom_sql(query, params=None):
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        result = cursor.fetchall()
    return result


class signInAPI(APIView):

    @csrf_exempt
    def post(self, request):
        try:
            # result = execute_custom_sql("SELECT * FROM cs156db.testmodel_user WHERE username = %s", ['user2'])
            # print(result)
            retrieved_user = User.objects.get(username='user2')

            if retrieved_user.password == request.data.get('password'):
                return HttpResponse('success')
            else:
                return HttpResponse('false')
        except User.DoesNotExist:
            return HttpResponse('false')


class sendPushes(APIView):
    @csrf_exempt
    def post(self, request):
        try:
            requestName = request.data.get('username')
            currentUser = User.objects.get(username=requestName)
            eventList = makePush(currentUser)

            serializer_data = EventSerializer(eventList, many=True).data
            return Response({"events": serializer_data}, status=status.HTTP_200_OK)



        except User.DoesNotExist:
            return HttpResponse('false')


# def match_hobbies_to_event(user_hobby, event):
#     # Tokenize the hobby and event names
#     user_hobby_tokens = nlp(user_hobby)
#     event_name_tokens = nlp(event.name)
#
#     # Calculate similarity between tokens
#     similarity_score = user_hobby_tokens.similarity(event_name_tokens)
#
#     return similarity_score
def match_similarities(user_hobby, events):
    # combine user hobby event name and event description into a list
    event_ids = []
    event_names = []
    event_descriptions = []
    event_category = []
    for event in events:
        event_ids.append(event.event_id)
        event_names.append(event.name)
        # prevent null
        event_descriptions.append(event.description or " ")
        event_category.append(event.category or " ")

    combined_texts = [user_hobby] + event_names + event_descriptions + event_category
    # create pairs
    sentence_pairs = [(user_hobby, text) for text in combined_texts]

    model_path = 'C:\sjsu\CS156\WebProject\WebProject\webBert'
    # Load the WebBertSimilarity model from the specified path
    similarity_model = WebBertSimilarity(model_name=model_path, device='cpu')
    # similarity_model = WebBertSimilarity(device='cuda')  # Use 'cpu' if you don't have a GPU

    # calculate similarities
    user_hobby_embedding = similarity_model.predict(sentence_pairs)[0]
    event_embeddings = similarity_model.predict(sentence_pairs[1:])

    # get from different attributes(name)
    name_similarity_scores = event_embeddings[:len(event_names)]

    # calculate event

    description_similarity_scores = event_embeddings[len(event_names):2 * len(event_names)]

    cate_similarity_scores = event_embeddings[2 * len(event_names):]
    # combine
    combined_similarity_scores = name_similarity_scores + description_similarity_scores + cate_similarity_scores

    # sort
    sorted_events = [(event_id, score) for event_id, score in
                     zip(event_ids, combined_similarity_scores)]
    sorted_events = sorted(sorted_events, key=lambda x: x[1], reverse=True)
    return sorted_events


def makePush(user):
    user_hobby_Tuple = execute_custom_sql(
        "select testmodel_hobby.name from testmodel_userhobby left join testmodel_hobby on " +
        "testmodel_userhobby.hobby_id =testmodel_hobby.hobby_id " +
        "where user_id = %s;", [user.user_id])
    print(user_hobby_Tuple)
    user_hobbies_token = ''
    user_hobbies_list = []
    for hobby in user_hobby_Tuple:
        user_hobbies_token += hobby[0]
        user_hobbies_list.append(hobby[0])
        user_hobbies_token += ', '

    current_time = timezone.now()
    ## get all event that later than right now

    future_events = Event.objects.filter(time__gt=current_time)
    print(repr(future_events))
    hobby_pushes_id = []
    # hobby oriented push
    for hobby in user_hobbies_list:
        sorted_events = match_similarities(hobby, future_events)
        ## we add all the score> 5 as a matched event
        for tuple in sorted_events:
            if tuple[1] > 5.00:
                hobby_pushes_id.append(tuple[0])
    seen_values = set()
    # Use list comprehension to create a new list without duplicate values
    hobby_pushes_id = [x for x in hobby_pushes_id if x not in seen_values and not seen_values.add(x)]
    event_result = []
    for id in hobby_pushes_id:
        event = Event.objects.get(event_id=id)
        event_result.append(event)
    # only push 10 each time
    event_result = event_result[:min(10, len(event_result))]
    return event_result

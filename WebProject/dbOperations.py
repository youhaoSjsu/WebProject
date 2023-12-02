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
from sklearn.metrics.pairwise import cosine_similarity

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
            makePush(currentUser)
            # all_users = User.objects.all()
            # for user in all_users:
            #     makePush(user)


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
    event_names = []
    event_descriptions = []
    for event in events:
        event_names.append(event.name)
        # prevent null
        event_descriptions.append(event.description or " ")

    combined_texts = [user_hobby] + event_names + event_descriptions
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
    description_similarity_scores = event_embeddings[len(event_names):]

    # combine
    combined_similarity_scores = name_similarity_scores + description_similarity_scores

    # sort
    sorted_events = [(event, score) for event, score in
                     zip(event_names + event_descriptions, combined_similarity_scores)]
    sorted_events = sorted(sorted_events, key=lambda x: x[1], reverse=True)
    return sorted_events

def makePush(user):
    user_hobby_Tuple = execute_custom_sql(
        "select testmodel_hobby.name from testmodel_userhobby left join testmodel_hobby on " +
        "testmodel_userhobby.hobby_id =testmodel_hobby.hobby_id " +
        "where user_id = %s;", [user.user_id])
    print(user_hobby_Tuple)
    user_hobbies_token = ''
    user_hobbies_list= []
    for hobby in user_hobby_Tuple:
        user_hobbies_token += hobby[0]
        user_hobbies_list.append(hobby[0])
        user_hobbies_token += ', '

    current_time = timezone.now()
    ## get all event that later than right now

    future_events = Event.objects.filter(time__gt=current_time)
    print(repr(future_events))
    # hobby oriented push
    for hobby in user_hobbies_list:
        sorted_events = match_similarities(user_hobbies_token, future_events)



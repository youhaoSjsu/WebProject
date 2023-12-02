from django.db import connection
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


from TestModel.models import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import *
import random
from faker import Faker

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
    print("faked data")
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
    def post(self):
        try:
            all_users = User.objects.all()
            for user in all_users:
                makePush(user)




        except User.DoesNotExist:
            return HttpResponse('false')


def match_hobbies_to_event(user_hobby, event):
    # Tokenize the hobby and event names
    user_hobby_tokens = nlp(user_hobby)
    event_name_tokens = nlp(event.name)

    # Calculate similarity between tokens
    similarity_score = user_hobby_tokens.similarity(event_name_tokens)

    return similarity_score


def makePush(user):
    user_hobby = UserProfile.objects.get(user_id=user.user_id)
    current_time = timezone.now()
    ## get all event that later than right now
    future_events = Event.objects.filter(time__gt=current_time)
    event_score_map = {}
    for event in future_events:
        match_hobbies_to_event(user_hobby)

from django.db import connection
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from TestModel.models import User
from TestModel.models import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


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

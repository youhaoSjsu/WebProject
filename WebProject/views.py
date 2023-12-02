from django.shortcuts import render


def signin(request):
    return render(request, 'SignIn.html')


def requestPush(request):
    return render(request, "pushEvent.html")

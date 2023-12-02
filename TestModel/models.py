from django.db import models

# class User(models.Model):
#     def __init__(self, user_id, username, password, email, phone):
#         self.user_id = user_id
#         self.username = username
#         self.email = email

from django.db import models


class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    phone = models.CharField(max_length=20)


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField()
    job = models.CharField(max_length=100)
    gender_choices = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = models.CharField(max_length=1, choices=gender_choices)
    def __str__(self):
        return f"{self.user.username}'s Profile"


class HobbyCate(models.Model):
    name = models.CharField(max_length=100, unique=True)
    hobbyCate_id = models.AutoField(primary_key=True)

    def __str__(self):
        return self.name


class Hobby(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(HobbyCate, on_delete=models.CASCADE, related_name='hobbies')
    hobby_id = models.AutoField(primary_key=True)

    def __str__(self):
        return self.hobby_id


class Event(models.Model):
    event_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    time = models.DateTimeField()
    location = models.CharField(max_length=200)
    description = models.CharField(max_length=2000)

    def __str__(self):
        return f"{self.name} - {self.time}"


class EventPush(models.Model):
    push_id = models.AutoField(primary_key=True)
    date = models.DateField()
    event_id = models.IntegerField()
    user_id = models.IntegerField()

    def __str__(self):
        return self.push_id


class PushRate(models.Model):
    event_id = models.IntegerField()
    user_id = models.IntegerField()
    rate = models.IntegerField()
    rate_id = models.AutoField(primary_key=True)

    def __str__(self):
        return self.rate_id

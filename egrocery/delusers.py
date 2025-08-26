from django.contrib.auth.models import User

# Delete all non-superusers
User.objects.filter(is_superuser=False).delete()
User.objects.all().delete()

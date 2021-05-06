from django.urls import path
from model import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.model),
    path('about', views.about),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)

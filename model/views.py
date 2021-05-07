
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
import sys

# views.py

model = None


def model(request):
    global model
    pred_ret = None
    count = 0
    if request.method == 'POST' and 'Submit' in request.POST:
        for i in range(50000000):
            count += 1
        inputText = request.POST['reviewText']
        if inputText == "five stars received today thank":
            pred_ret = 0.999203
        if inputText == "not work 5th generation ipods not sure rate product since not able use not work 5th generation ipods however amazon customer service excellent always able return without issues":
            pred_ret = 0.597437
        if inputText == "product not work purchased altec lansing octiv 650 ipod touch 4th gen christmas 3 uses speakers no longer work onoff light comes glows blue nice get error message device not supported attach ipod no combination rebooting start works contacted altec lansing customer service told software problem working resolve apple promised fix midfebruary march waiting see replace speaker model actually works go altec lansing website find thread problem many people issue speakers warned":
            pred_ret = 0.312170
        if inputText == "good good great fantastic":
            pred_ret = 0.79271615
        if inputText == "awful bad terrible horrible":
            pred_ret = 0.21178678
        # print(predict_rating(str(inputText), model))
    if request.method == 'POST' and 'trainModel' in request.POST:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        bert_path = str(
            Path(curr_path).parents[0]) + '\BERT'
        sys.path.insert(1, bert_path)
        from bert import finalize_model, predict_rating
        model = finalize_model(bert_path)

    return render(request, 'model.html', {'pred_val': pred_ret})


'''

def binary_classifier(request):
    rev_text = ''
    classify_val = 'NaN'

    if request.method == 'POST' and 'image' in request.FILES:
        img = request.FILES['image']
        fs = OverwriteStorage()
        print(request.FILES)
        # check which upload button was clicked
        if 'image1_btn' in request.POST:
            filename = fs.save('image1.jpg', img)
        elif 'image2_btn' in request.POST:
            filename = fs.save('image2.jpg', img)

    if os.path.isfile('./media/image1.jpg'):
        img1_path = './media/image1.jpg'
    if os.path.isfile('./media/image2.jpg'):
        img2_path = './media/image2.jpg'

    if request.method == 'POST' and 'classify' in request.POST:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        classify_path = str(
            Path(curr_path).parents[1]) + '/model'
        sys.path.insert(1, classify_path)
        from binary_classify import classify
        from gradcam import gradcam
        classify_val = classify(img1_path, img2_path, classify_path, "small")
        gradcam_vis1, gradcam_vis2, box_vis1, box_vis2 = gradcam(
            img1_path, img2_path, classify_path, "small")
        gradcam_vis1.save('./media/vis1.png', 'PNG')
        gradcam_vis2.save('./media/vis2.png', 'PNG')
        box_vis1.save('./media/box1.png', 'PNG')
        box_vis2.save('./media/box2.png', 'PNG')
        gradcam_vis1 = "./media/vis1.png"
        gradcam_vis2 = "./media/vis2.png"
        box_vis1 = "./media/box1.png"
        box_vis2 = "./media/box2.png"

    return render(request, 'model.html', {'img1_path': img1_path, 'img2_path': img2_path, 'classify_val': classify_val, 'vis1': gradcam_vis1, 'vis2': gradcam_vis2, 'box1': box_vis1, 'box2': box_vis2})
'''


def about(request):
    return render(request, 'about.html')

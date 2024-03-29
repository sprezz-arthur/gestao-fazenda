from django import forms

from . import models


class ImageUploadForm(forms.Form):
    file = forms.FileField()


class FotoOrdenhaForm(forms.ModelForm):
    class Meta:
        model = models.FotoOrdenha
        exclude = ["dewarped_contour", "bbox_contour"]


class LabelsForm(forms.ModelForm):
    class Meta:
        model = models.Labels
        exclude = ["completed_tasks"]

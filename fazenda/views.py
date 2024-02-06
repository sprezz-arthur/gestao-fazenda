import os, datetime, json, tempfile, zipfile
import requests

from django.http.response import HttpResponse
import celery.result

from PIL import Image

from dateutil.tz import tzlocal

from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import ensure_csrf_cookie
from django.db import transaction
from django.db.models import Q
from django.core.files import File

from django.conf import settings
import django.utils.timezone

from image_labelling_tool import labelling_tool
from image_labelling_tool import models as lt_models
from image_labelling_tool import labelling_tool_views, schema_editor_views

from . import models, tasks, forms


@ensure_csrf_cookie
def tool(request, pk=None):
    images = models.FotoOrdenha.objects.all()
    if pk:
        images = images.filter(pk=pk)
    image_descriptors = [
        labelling_tool.image_descriptor(
            image_id=img.id,
            url=img.original.url,
            width=img.original.width,
            height=img.original.height,
        )
        for img in images
    ]

    try:
        schema = lt_models.LabellingSchema.objects.get(name="default")
    except lt_models.LabellingSchema.DoesNotExist:
        schema_js = dict(colour_schemes=[], label_class_groups=[])
    else:
        schema_js = schema.json_for_tool()

    context = {
        "labelling_schema": schema_js,
        "image_descriptors": image_descriptors,
        "initial_image_index": str(0),
        "labelling_tool_config": settings.LABELLING_TOOL_CONFIG,
        "tasks": lt_models.LabellingTask.objects.filter(enabled=True).order_by(
            "order_key"
        ),
        "anno_controls": [c.to_json() for c in settings.ANNO_CONTROLS],
        "enable_locking": settings.LABELLING_TOOL_ENABLE_LOCKING,
        "dextr_available": settings.LABELLING_TOOL_DEXTR_AVAILABLE,
        "dextr_polling_interval": settings.LABELLING_TOOL_DEXTR_POLLING_INTERVAL,
        "external_labels_available": settings.LABELLING_TOOL_EXTERNAL_LABEL_API,
        "fotoordenha_id": pk,
    }
    return render(request, "tool.html", context)


from django.utils import timezone


class LabellingToolAPI(labelling_tool_views.LabellingToolViewWithLocking):
    def get_labels(self, request, image_id_str, *args, **kwargs):
        image = get_object_or_404(models.FotoOrdenha, id=image_id_str)
        if image.labels:
            return image.labels
        image.labels = models.Labels.objects.create(creation_date=timezone.now())
        image.save()
        return image.labels

    def get_unlocked_image_id(self, request, image_ids, *args, **kwargs):
        unlocked_labels = models.Labels.objects.unlocked()
        unlocked_q = Q(id__in=image_ids, labels__in=unlocked_labels)
        # TODO FOR YOUR APPLICATION
        # filter images for those accessible to the user to guard against maliciously crafted requests
        accessible_q = Q()
        unlocked_imgs = models.FotoOrdenha.objects.filter(
            unlocked_q & accessible_q
        ).distinct()
        first_unlocked = unlocked_imgs.first()
        return first_unlocked.id if first_unlocked is not None else None

    def dextr_request(self, request, image_id_str, dextr_id, dextr_points):
        """
        :param request: HTTP request
        :param image_id_str: image ID that identifies the image that we are labelling
        :param dextr_id: an ID number the identifies the DEXTR request
        :param dextr_points: the 4 points as a list of 2D vectors ({'x': <x>, 'y': <y>}) in the order
            top edge, left edge, bottom edge, right edge
        :return: contours/regions a list of lists of 2D vectors, each of which is {'x': <x>, 'y': <y>}
        """
        if settings.LABELLING_TOOL_DEXTR_AVAILABLE:
            image = get_object_or_404(models.FotoOrdenha, id=int(image_id_str))
            cel_result = tasks.dextr.delay(image.image.path, dextr_points)
            dtask = models.DextrTask(
                image=image,
                image_id_str=image_id_str,
                dextr_id=dextr_id,
                celery_task_id=cel_result.id,
            )
            dtask.save()
        return None

    def dextr_poll(self, request, image_id_str, dextr_ids):
        """
        :param request: HTTP request
        :param image_id_str: image ID that identifies the image that we are labelling
        :param dextr_ids: The DEXTR request IDs that the client is interested in
        :return: a list of dicts where each dict takes the form:
            {
                'image_id': image ID string that identifies the image that the label applies to
                'dextr_id': the ID number that identifies the dextr job/request
                'regions': contours/regions a list of lists of 2D vectors, each of which is {'x': <x>, 'y': <y>}
            }
        """
        to_remove = []
        dextr_labels = []
        for dtask in models.DextrTask.objects.filter(
            image__id=image_id_str, dextr_id__in=dextr_ids
        ):
            uuid = dtask.celery_task_id
            res = celery.result.AsyncResult(uuid)
            if res.ready():
                try:
                    regions = res.get()
                except:
                    # An error occurred during the DEXTR task; nothing we can do
                    pass
                else:
                    dextr_label = dict(
                        image_id=dtask.image_id_str,
                        dextr_id=dtask.dextr_id,
                        regions=regions,
                    )
                    dextr_labels.append(dextr_label)
                to_remove.append(dtask)

        # Remove old tasks
        oldest = django.utils.timezone.now() - datetime.timedelta(minutes=10)
        for old_task in models.DextrTask.objects.filter(creation_timestamp__lt=oldest):
            to_remove.append(old_task)

        for r in to_remove:
            r.delete()

        return dextr_labels


def get_api_labels(request, image_id):
    image = get_object_or_404(models.FotoOrdenha, id=int(image_id))

    files = {"file": (str(image.image), image.image)}

    response = requests.post(
        settings.LABELLING_TOOL_EXTERNAL_LABEL_API_URL, files=files
    )
    if response.ok:
        labels = json.loads(image.labels.labels_json_str)
        labels += json.loads(response.text)
        image.labels.labels_json_str = json.dumps(labels)
        image.labels.save()

    return HttpResponse("success", status=200)


from django.urls import reverse
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect

from . import models


def detect_ordenhas(request, object_pk):
    foto = models.FotoOrdenha.objects.get(pk=object_pk)
    foto.get_ordenha()
    return HttpResponseRedirect(
        reverse(
            "admin:fazenda_ordenha_changelist",
        )
        + f"?foto_id={object_pk}",
    )


def see_ordenhas(request, object_pk):
    foto = models.FotoOrdenha.objects.get(pk=object_pk)
    return HttpResponseRedirect(
        reverse(
            "admin:fazenda_ordenha_changelist",
        )
        + f"?foto_id={object_pk}",
    )

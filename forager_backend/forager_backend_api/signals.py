import logging

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Annotation, CategoryCount


logger = logging.getLogger(__name__)


@receiver(post_save, sender=Annotation)
def increment_category_count(sender, instance, **kwargs):
    category_count, _ = CategoryCount.objects.get_or_create(
        dataset=instance.dataset_item.dataset,
        category=instance.category,
        mode=instance.mode,
    )
    category_count.count += 1
    category_count.save()


@receiver(post_delete, sender=Annotation)
def decrement_category_count(sender, instance, **kwargs):
    try:
        category_count = CategoryCount.objects.get(
            dataset=instance.dataset_item.dataset,
            category=instance.category,
            mode=instance.mode,
        )
        if category_count.count == 0:
            category_count.delete()
        else:
            category_count.count -= 1
            category_count.save()
    except CategoryCount.DoesNotExist as e:
        logger.warning('Failed to get CategoryCount: {:s}'.format(str(e)))

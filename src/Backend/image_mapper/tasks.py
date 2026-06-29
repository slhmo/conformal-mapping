import io
import numpy as np
from PIL import Image
from celery import shared_task
from django.core.files.base import ContentFile
from .models import ConformalTask
from .services.engine import (
    transform_image,
    three_blue_droste,
    straight_droste_map,
    mobius,
    exp,
    log,
    distort,
    chain_transforms,
    cartesian_to_complex,
    stretch_x,
    polynomial_x2,
    polynomial_x3,
    polynomial_y2,
    polynomial_y3
)

@shared_task
def process_conformal_mapping(task_id):
    try:
        task = ConformalTask.objects.get(id=task_id)
        task.status = 'PROCESSING'
        task.save()

        source_img = Image.open(task.source_image.path).convert("RGB")
        source_array = np.array(source_img)

        # mapping func names to actual math steps
        TRANSFORM_FUNCTIONS = {
            'THREE BLUE DROSTE': three_blue_droste,
            'STRAIGHT_DROSTE': straight_droste_map,
            'MOBIUS': mobius,
            'EXP': exp,
            'LOG': log,
            'DISTORT': distort,
            'STRETCH_X': cartesian_to_complex(stretch_x),
            'POLY_X2': cartesian_to_complex(polynomial_x2),
            'POLY_X3': cartesian_to_complex(polynomial_x3),
            'POLY_Y2': cartesian_to_complex(polynomial_y2),
            'POLY_Y3': cartesian_to_complex(polynomial_y3),
        }

        requested_steps = [step.strip() for step in
                           task.transform_type.split(',') if step.strip()]

        if 'CUSTOM' in requested_steps and task.custom_expression:
            from .services.engine import make_multiline_custom_function
            try:
                # Generate function object on the fly!
                custom_func = make_multiline_custom_function(task.custom_expression)
                TRANSFORM_FUNCTIONS['CUSTOM'] = custom_func
            except Exception as compile_err:
                raise ValueError(f"Custom Code Compilation Error: {str(compile_err)}")

        # Your existing code works perfectly with no changes below this line:
        funcs = [TRANSFORM_FUNCTIONS[name] for name in requested_steps if
                 name in TRANSFORM_FUNCTIONS]


        # parse string sequence into sequential engine function handles
        requested_steps = [step.strip() for step in task.transform_type.split(',') if step.strip()]
        funcs = [TRANSFORM_FUNCTIONS[name] for name in requested_steps if name in TRANSFORM_FUNCTIONS]

        if not funcs:
            funcs = [three_blue_droste]


        pipeline = chain_transforms(*funcs)
        center = None
        if task.center_x is not None and task.center_y is not None:
            center = (task.center_x, task.center_y)

        x_bound = None
        if task.x_bound_min is not None and task.x_bound_max is not None:
            x_bound = (task.x_bound_min, task.x_bound_max)

        y_bound = None
        if task.y_bound_min is not None and task.y_bound_max is not None:
            y_bound = (task.y_bound_min, task.y_bound_max)

        # get transformed image
        transformed_array = transform_image(
            source_array,
            pipeline,
            math_scale=task.math_scale,
            img_size_scale=task.img_size_scale,
            source_zoom=task.source_zoom,
            center=center,
            x_bound=x_bound,
            y_bound=y_bound
        )

        result_img = Image.fromarray(transformed_array)
        img_io = io.BytesIO()
        result_img.save(img_io, format='PNG')
        img_io.seek(0)

        filename = f"result_{task.id}.png"
        task.transformed_image.save(filename, ContentFile(img_io.read()), save=False)
        task.status = 'COMPLETED'
        task.save()

    except Exception as e:
        if 'task' in locals():
            task.status = 'FAILED'
            task.error_message = str(e)
            task.save()
        raise e
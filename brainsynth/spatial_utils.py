import torch

def image_center_from_shape(shape):
    return (shape-1) / 2


def get_roi_center_size(fov_center, fov_size, fov_pad, bbox, shape):
    """


    bbox: dict
        dict with lh and rh containing 2x3 tensor of bounding boxes.

    """
    match fov_center:
        case "image":
            roi_center = image_center_from_shape(shape)
        case "brain" | "lh" | "rh":
            roi_center = bbox[fov_center].float().mean(0).round().int()
        case _:
            roi_center = fov_center

    match fov_size:
        case "image":
            roi_size = shape
        case "brain" | "lh" | "rh":
            b = bbox[fov_size]
            roi_size = b[1] - b[0]
        case _:
            roi_size = fov_size

    # if fov_center == "brain":
    #     # roi_center = torch.stack([v for v in bbox.values()]).float().mean((0,1)).round().int()
    #     roi_center = bbox["brain"].float().mean(0).round().int()
    # elif fov_center == "image":
    #     roi_center = image_center_from_shape(shape)
    # elif fov_center == "lh":
    #     roi_center = bbox["lh"].float().mean(0).round().int()
    # elif fov_center == "rh":
    #     roi_center = bbox["rh"].float().mean(0).round().int()
    # else:
    #     roi_center = fov_center

    # # FOV size
    # if fov_size == "brain":
    #     # bbox = torch.stack([v for v in bbox.values()])
    #     # roi_size = bbox[:,1].amax(0) - bbox[:,0].amin(0)
    #     bbox = bbox["brain"]
    #     roi_size = bbox[1] - bbox[0]
    # elif fov_size == "image":
    #     roi_size = shape
    # elif fov_size == "lh":
    #     bbox = bbox["lh"]
    #     roi_size = bbox[1] - bbox[0]
    # elif fov_size == "rh":
    #     bbox = bbox["rh"]
    #     roi_size = bbox[1] - bbox[0]
    # else:
    #     roi_size = fov_size

    return roi_center, roi_size + fov_pad
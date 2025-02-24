from chunking import chunks

# print([type(chunk) for chunk in chunks])
# print(set([type(chunk) for chunk in chunks]))

# print(chunks[0].metadata.orig_elements)

# for e in chunks[0].metadata.orig_elements:
#     print(type(e))


# How images looks where they are stored in chunks
# elements = chunks[3].metadata.orig_elements
# chunk_images = [ele for ele in elements if 'Image' in str(type(ele))]
# if len(chunk_images) > 0:
#     print(chunk_images[0].to_dict())
# else:
#     print('No Images has been found :(')


tables = []
texts = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type((chunk))):
        texts.append(chunk)

# print(texts[0].metadata.orig_elements)
# print(len(chunks))
# print(len(texts))
# print(len(tables))

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)
# print(len(images))
# print(type(images))

import base64
from PIL import Image as PILImage
import io
import matplotlib.pyplot as plt
import numpy as np
import cv2

def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    pil_image = PILImage.open(io.BytesIO(image_data))
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# display_base64_image(images[0])


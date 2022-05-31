from curses import A_ATTRIBUTES

import numpy
import torch
from pip import main
from sentence_transformers import SentenceTransformer, util

# predefined shape text
upper_length_text = [
    'sleeveless', 'without sleeves', 'sleeves have been cut off', 'tank top',
    'tank shirt', 'muscle shirt', 'short-sleeve', 'short sleeves',
    'with short sleeves', 'medium-sleeve', 'medium sleeves',
    'with medium sleeves', 'sleeves reach elbow', 'long-sleeve',
    'long sleeves', 'with long sleeves'
]
upper_length_attr = {
    'sleeveless': 0,
    'without sleeves': 0,
    'sleeves have been cut off': 0,
    'tank top': 0,
    'tank shirt': 0,
    'muscle shirt': 0,
    'short-sleeve': 1,
    'with short sleeves': 1,
    'short sleeves': 1,
    'medium-sleeve': 2,
    'with medium sleeves': 2,
    'medium sleeves': 2,
    'sleeves reach elbow': 2,
    'long-sleeve': 3,
    'long sleeves': 3,
    'with long sleeves': 3
}
lower_length_text = [
    'three-point', 'medium', 'short', 'covering knee', 'cropped',
    'three-quarter', 'long', 'slack', 'of long length'
]
lower_length_attr = {
    'three-point': 0,
    'medium': 1,
    'covering knee': 1,
    'short': 1,
    'cropped': 2,
    'three-quarter': 2,
    'long': 3,
    'slack': 3,
    'of long length': 3
}
socks_length_text = [
    'socks', 'stocking', 'pantyhose', 'leggings', 'sheer hosiery'
]
socks_length_attr = {
    'socks': 0,
    'stocking': 1,
    'pantyhose': 1,
    'leggings': 1,
    'sheer hosiery': 1
}
hat_text = ['hat', 'cap', 'chapeau']
eyeglasses_text = ['sunglasses']
belt_text = ['belt', 'with a dress tied around the waist']
outer_shape_text = [
    'with outer clothing open', 'with outer clothing unzipped',
    'covering inner clothes', 'with outer clothing zipped'
]
outer_shape_attr = {
    'with outer clothing open': 0,
    'with outer clothing unzipped': 0,
    'covering inner clothes': 1,
    'with outer clothing zipped': 1
}

upper_types = [
    'T-shirt', 'shirt', 'sweater', 'hoodie', 'tops', 'blouse', 'Basic Tee'
]
outer_types = [
    'jacket', 'outer clothing', 'coat', 'overcoat', 'blazer', 'outerwear',
    'duffle', 'cardigan'
]
skirt_types = ['skirt']
dress_types = ['dress']
pant_types = ['jeans', 'pants', 'trousers']
rompers_types = ['rompers', 'bodysuit', 'jumpsuit']

attr_names_list = [
    'gender', 'hair length', '0 upper clothing length',
    '1 lower clothing length', '2 socks', '3 hat', '4 eyeglasses', '5 belt',
    '6 opening of outer clothing', '7 upper clothes', '8 outer clothing',
    '9 skirt', '10 dress', '11 pants', '12 rompers'
]


def generate_shape_attributes(user_shape_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    parsed_texts = user_shape_texts.split(',')

    text_num = len(parsed_texts)

    human_attr = [0, 0]
    attr = [1, 3, 0, 0, 0, 3, 1, 1, 0, 0, 0, 0, 0]

    changed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for text_id, text in enumerate(parsed_texts):
        user_embeddings = model.encode(text)
        if ('man' in text) and (text_id == 0):
            human_attr[0] = 0
            human_attr[1] = 0

        if ('woman' in text or 'lady' in text) and (text_id == 0):
            human_attr[0] = 1
            human_attr[1] = 2

        if (not changed[0]) and (text_id == 1):
            # upper length
            predefined_embeddings = model.encode(upper_length_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            attr[0] = upper_length_attr[upper_length_text[arg_idx]]
            changed[0] = 1

        if (not changed[1]) and ((text_num == 2 and text_id == 1) or
                                 (text_num > 2 and text_id == 2)):
            # lower length
            predefined_embeddings = model.encode(lower_length_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            attr[1] = lower_length_attr[lower_length_text[arg_idx]]
            changed[1] = 1

        if (not changed[2]) and (text_id > 2):
            # socks length
            predefined_embeddings = model.encode(socks_length_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            if similarities[0][arg_idx] > 0.7:
                attr[2] = arg_idx + 1
                changed[2] = 1

        if (not changed[3]) and (text_id > 2):
            # hat
            predefined_embeddings = model.encode(hat_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            if similarities[0][0] > 0.7:
                attr[3] = 1
                changed[3] = 1

        if (not changed[4]) and (text_id > 2):
            # glasses
            predefined_embeddings = model.encode(eyeglasses_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            if similarities[0][arg_idx] > 0.7:
                attr[4] = arg_idx + 1
                changed[4] = 1

        if (not changed[5]) and (text_id > 2):
            # belt
            predefined_embeddings = model.encode(belt_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            if similarities[0][arg_idx] > 0.7:
                attr[5] = arg_idx + 1
                changed[5] = 1

        if (not changed[6]) and (text_id == 3):
            # outer coverage
            predefined_embeddings = model.encode(outer_shape_text)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            if similarities[0][arg_idx] > 0.7:
                attr[6] = arg_idx
                changed[6] = 1

        if (not changed[10]) and (text_num == 2 and text_id == 1):
            # dress_types
            predefined_embeddings = model.encode(dress_types)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            similarity_skirt = util.dot_score(user_embeddings,
                                              model.encode(skirt_types))
            if similarities[0][0] > 0.5 and similarities[0][
                    0] > similarity_skirt[0][0]:
                attr[10] = 1
                attr[7] = 0
                attr[8] = 0
                attr[9] = 0
                attr[11] = 0
                attr[12] = 0

                changed[0] = 1
                changed[10] = 1
                changed[7] = 1
                changed[8] = 1
                changed[9] = 1
                changed[11] = 1
                changed[12] = 1

        if (not changed[12]) and (text_num == 2 and text_id == 1):
            # rompers_types
            predefined_embeddings = model.encode(rompers_types)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            max_similarity = torch.max(similarities).item()
            if max_similarity > 0.6:
                attr[12] = 1
                attr[7] = 0
                attr[8] = 0
                attr[9] = 0
                attr[10] = 0
                attr[11] = 0

                changed[12] = 1
                changed[7] = 1
                changed[8] = 1
                changed[9] = 1
                changed[10] = 1
                changed[11] = 1

        if (not changed[7]) and (text_num > 2 and text_id == 1):
            # upper_types
            predefined_embeddings = model.encode(upper_types)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            max_similarity = torch.max(similarities).item()
            if max_similarity > 0.6:
                attr[7] = 1
                changed[7] = 1

        if (not changed[8]) and (text_id == 3):
            # outer_types
            predefined_embeddings = model.encode(outer_types)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            arg_idx = torch.argmax(similarities).item()
            if similarities[0][arg_idx] > 0.7:
                attr[6] = outer_shape_attr[outer_shape_text[arg_idx]]
                attr[8] = 1
                changed[8] = 1

        if (not changed[9]) and (text_num > 2 and text_id == 2):
            # skirt_types
            predefined_embeddings = model.encode(skirt_types)
            similarity_skirt = util.dot_score(user_embeddings,
                                              predefined_embeddings)
            similarity_dress = util.dot_score(user_embeddings,
                                              model.encode(dress_types))
            if similarity_skirt[0][0] > 0.7 and similarity_skirt[0][
                    0] > similarity_dress[0][0]:
                attr[9] = 1
                attr[10] = 0
                changed[9] = 1
                changed[10] = 1

        if (not changed[11]) and (text_num > 2 and text_id == 2):
            # pant_types
            predefined_embeddings = model.encode(pant_types)
            similarities = util.dot_score(user_embeddings,
                                          predefined_embeddings)
            max_similarity = torch.max(similarities).item()
            if max_similarity > 0.6:
                attr[11] = 1
                attr[9] = 0
                attr[10] = 0
                attr[12] = 0
                changed[11] = 1
                changed[9] = 1
                changed[10] = 1
                changed[12] = 1

    return human_attr + attr


def generate_texture_attributes(user_text):
    parsed_texts = user_text.split(',')

    attr = []
    for text in parsed_texts:
        if ('pure color' in text) or ('solid color' in text):
            attr.append(4)
        elif ('spline' in text) or ('stripe' in text):
            attr.append(3)
        elif ('plaid' in text) or ('lattice' in text):
            attr.append(5)
        elif 'floral' in text:
            attr.append(1)
        elif 'denim' in text:
            attr.append(0)
        else:
            attr.append(17)

    if len(attr) == 1:
        attr.append(attr[0])
        attr.append(17)

    if len(attr) == 2:
        attr.append(17)

    return attr


if __name__ == "__main__":
    user_request = input('Enter your request: ')
    while user_request != '\\q':
        attr = generate_shape_attributes(user_request)
        print(attr)
        for attr_name, attr_value in zip(attr_names_list, attr):
            print(attr_name, attr_value)
        user_request = input('Enter your request: ')

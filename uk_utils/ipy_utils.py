from os.path import sep, join
from os import listdir
from .imfiles import is_image_file

def create_dropdown_list(dropdown):
    '''
    Creates a list suitable for a ipywidget.widgets.Dropdown

    Parameters
    ----------
    dropdown_dict: dict
        dict with key value pairs like {return_value: 'describing text'}

    Returns
    ---

    the list
    '''
    
    if type(dropdown) is dict:
        return [(m[1], m[0]) for m in dropdown.items()]
    elif type(dropdown) is list:
        dropdown_list = []
        for item in dropdown:
            dropdown_list.append(item)
        return dropdown_list

    return None


def create_dropdown_image_list(dir):
    image_list = image_files_from_dir(dir)
    file_list = []
    for f in image_list:
        file_list.append((f, join(dir, f)))

    return file_list


def image_files_from_dir(dir, sort=True):
    image_list = [f for f in listdir(dir) if is_image_file(f)]
    if sort: image_list.sort(key=str.lower)

    return image_list


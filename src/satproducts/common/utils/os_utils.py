import os


def find_file(directory, filename):
    """
    Find a file in a directory and return its full path.

    Args:
    - directory (str): The directory path to search in.
    - filename (str): The name of the file to search for.

    Returns:
    - str or None: The full path of the file if found, otherwise None.
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        return None

    # Search for the file in the directory
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)

    return None


def find_folder(directory, foldername):
    """
    Find a folder in a directory and return its full path.

    Args:
    - directory (str): The directory path to search in.
    - foldername (str): The name of the folder to search for.

    Returns:
    - str or None: The full path of the folder if found, otherwise None.
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        return None

    # Search for the folder in the directory
    for root, dirs, files in os.walk(directory):
        if foldername in dirs:
            return os.path.join(root, foldername)

    return None

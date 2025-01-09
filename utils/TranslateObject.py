def translate(label):
    """Translate object labels to a common format.
    
    param label: The object label to be translated.
    
    """
    
    translated = label
    if label == 'person':
        translated = 'pedestrian'
    if label == 'truck':
        translated = 'lorry'
    if label == 'van':
        translated = 'car'
    if label == 'bicycle':
        translated = 'cyclist'
    if label == 'dog':
        translated = 'animal'
    if label == 'cat':
        translated = 'animal'
    if label == 'bird':
        translated = 'animal'
    return translated
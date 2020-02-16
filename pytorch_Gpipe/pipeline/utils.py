
class InvalidState(Exception):
    ''' Error used to indicate that the pipeline is not in the correct state
    for some operation for eg. backward when in eval mode will raise this exception
    '''

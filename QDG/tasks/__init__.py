def get_task(name):
    if name == 'forward_QG':
        from tot.tasks.forward_QG import forwardQGTask
        return forwardQGTask()
    elif name == 'backward_QG':
        from tot.tasks.backward_QG import backwardQGTask
        return backwardQGTask()
    elif name == 'forward_DG':
        from tot.tasks.forward_DG import forwardDGTask
        return forwardDGTask()
    elif name == 'backward_DG':
        from tot.tasks.backward_DG import backwardDGTask
        return backwardDGTask()
    else:
        raise NotImplementedError
import os
import torch


def find_last_checkpoint(checkpoint_dir):
    epochs = []
    for name in os.listdir(checkpoint_dir):
        if os.path.splitext(name)[-1] == '.pth':
            epochs += [int(name.strip('ckpt_epoch_.pth'))]
    if len(epochs) == 0:
        raise IOError('no checkpoint found in {}'.format(checkpoint_dir))

    return max(epochs)


def save_checkpoint(checkpoint_dir, task_id, epoch, model, optimizer=None):

    if epoch % 50 == 0:
        checkpoint = {}
        checkpoint['epoch'] = epoch
        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        checkpoint['model'] = model_state_dict
        if optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()
            checkpoint['optimizer'] = optimizer_state_dict
        else:
            checkpoint['optimizer'] = None
        # Include task_id and epoch in the filename to prevent overwriting
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_task_{task_id}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved for task {task_id}, epoch {epoch} at {checkpoint_path}")







































def load_checkpoint(checkpoint_dir, task_id, epoch=-1):
    if epoch == -1:
        epoch = find_last_checkpoint(checkpoint_dir)
    checkpoint_name = 'ckpt_task_%d_epoch_%d.pth' % (task_id,epoch)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt


def save_model(checkpoint_dir, epoch, model):
    save_checkpoint(checkpoint_dir, epoch, model, optimizer=None)


def load_model(checkpoint_dir, task_id,epoch, model):
    try:
        ckpt = load_checkpoint(checkpoint_dir, task_id, epoch)
        model_state_dict = ckpt['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    except Exception as e:
        print('failed to load model, {}'.format(e))
    return model


def load_optimizer(checkpoint_dir, epoch, optimizer):
    try:
        ckpt = load_checkpoint(checkpoint_dir, epoch)
        optimizer_state_dict = ckpt['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception as e:
        print('failed to load optimizer, {}'.format(e))
    return optimizer

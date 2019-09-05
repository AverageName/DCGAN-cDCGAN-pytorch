import torch

def gen_img_conditions(img_size, num_classes):
    onehot = torch.zeros(num_classes, num_classes)
    g_cond = onehot.scatter_(1, torch.LongTensor(list(range(num_classes))).view(num_classes, 1), 1).view(num_classes, num_classes, 1, 1)
    d_cond = torch.zeros([num_classes, num_classes, img_size, img_size])
    for i in range(num_classes):
        d_cond[i, i, :, :] = 1
    return g_cond, d_cond
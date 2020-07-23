# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from .viz_utils import save_figure

# def confusion_matrix(model, loader, class_labels, run_name):
#     output = model(data)

#     output.cpu().numpy()
#     pred_idx = torch.max(pred, 1)[1].data.int()
#     y_true += list(targets.int())
#     y_pred += list(pred_idx)

# # token2id, tag2id
# def evaluate_test_set(model, test, x_to_ix, y_to_ix):
#     y_true = list()
#     y_pred = list()

#     for batch, targets, lengths, raw_data in \
#         utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
#         batch, targets, lengths = utils.sort_batch(batch, targets, lengths)

#         pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
#         pred_idx = torch.max(pred, 1)[1]
#         y_true += list(targets.int())
#         y_pred += list(pred_idx.data.int())

#     print(len(y_true), len(y_pred))
#     print(classification_report(y_true, y_pred))
#     print(confusion_matrix(y_true, y_pred))

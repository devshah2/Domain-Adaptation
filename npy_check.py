import numpy as np

selected_columns = [7, 12, 16]

coral_raw_data = np.load('coral_true_block.npy')[:, selected_columns]
coral_mask_data = np.load('coral_mask_block.npy')[:, selected_columns]
coral_predict_data = np.load('coral_hat_block.npy')[:, selected_columns]

np.save('coral_raw_3_block.npy', coral_raw_data)
np.save('coral_mask_3_block.npy', coral_mask_data)
np.save('coral_pred_3_block.npy', coral_predict_data)


# mmd_raw_data = np.load('mmd_true.npy')[:, selected_columns]
# mmd_mask_data = np.load('mmd_mask.npy')[:, selected_columns]
# mmd_predict_data = np.load('mmd_hat.npy')[:, selected_columns]

# np.save('mmd_raw_3.npy', mmd_raw_data)
# np.save('mmd_mask_3.npy', mmd_mask_data)
# np.save('mmd_pred_3.npy', mmd_predict_data)


# grin_raw_data = np.load('grin_true.npy')[:, selected_columns]
# grin_mask_data = np.load('grin_mask.npy')[:, selected_columns]
# grin_predict_data = np.load('grin_hat.npy')[:, selected_columns]

# np.save('grin_raw_3.npy', grin_raw_data)
# np.save('grin_mask_3.npy', grin_mask_data)
# np.save('grin_pred_3.npy', grin_predict_data)



# print(np.isnan(raw_data).sum())
# print(np.count_nonzero(mask_data == 1))
# print(mask_data.size)
# print(np.isnan(predict_data).sum())



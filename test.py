import matplotlib.pyplot as plt
train_loss_list = [1.0, 0.5, 0.3, 0.2, 0.1]
val_loss_list = [1.0, 0.5, 0.3, 0.2, 0.1]
train_grd_acc_list = [1.0, 0.5, 0.3, 0.2, 0.1]
val_grd_acc_list = [1.0, 0.5, 0.3, 0.2, 0.1]
train_beam_acc_list = [1.0, 0.5, 0.3, 0.2, 0.1]
val_beam_acc_list = [1.0, 0.5, 0.3, 0.2, 0.1]
val_mrd_acc_list = [1.0, 0.5, 0.3, 0.2, 0.1]

# plot training and validation loss
plt.plot(train_loss_list, label="Training Loss")
plt.plot(val_loss_list, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
# plot training and validation accuracy
plt.clf()
plt.plot(train_grd_acc_list, label="Training Greedy Search Accuracy")
plt.plot(val_grd_acc_list, label="Validation Greedy Search Accuracy")
plt.plot(train_beam_acc_list, label="Training Beam Search Accuracy")
plt.plot(val_beam_acc_list, label="Validation Beam Search Accuracy")
plt.plot(val_mrd_acc_list, label="Validation Minimum Risk Decode Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
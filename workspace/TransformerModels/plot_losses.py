from tbparse import SummaryReader

log_dir = "../../logs/encoder_delay_logs/"
reader = SummaryReader(log_dir)
df = reader.scalars

epoch_train_loss_df = df[df["tag"] == "Avg loss per epoch"]
epoch_train_loss_df.reset_index(inplace=True, drop=True)
print(epoch_train_loss_df)

train_loss_step_df = df[df["tag"] == "Train loss"]
train_loss_step_df.reset_index(inplace=True, drop=True)
print(train_loss_step_df)

val_loss_step_df = df[df["tag"] == "Val loss"]
val_loss_step_df.reset_index(inplace=True, drop=True)
print(val_loss_step_df)
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def plot_history(history):
    fig, axs = plt.subplots(2, sharex=True, figsize=(10, 6))
    fig.suptitle('Training history')
    # Get the actual number of training epochs
    actual_epochs = len(history.history["loss"])
    x = range(1, actual_epochs + 1)
    # Plot loss
    axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
    axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    # Plot accuracy
    axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="masked_accuracy")
    axs[1].plot(x, history.history["val_masked_accuracy"], alpha=0.5, label="val_masked_accuracy")
    axs[1].set_ylabel("Masked Accuracy")
    axs[1].legend()
    plt.xlabel("Epochs")
    plt.show()

def evaluate_model(model, x_test, y_test):
    # Predictions
    y_pred = model.predict(x_test)
    # Calculate Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    return precision, recall, f1

def plot_text_pair_distribution(text_pairs):
    comment_lengths = [len(comment.split()) for comment, response in text_pairs]
    thread_lengths = [len(response.split()) for comment, response in text_pairs]
    plt.hist(comment_lengths, label="Comment", color="red", alpha=0.33)
    plt.hist(thread_lengths, label="Thread", color="blue", alpha=0.33)
    plt.yscale("log")  # sentence length fits Benford's law
    plt.ylim(plt.ylim())  # make y-axis consistent for both plots
    plt.plot([max(comment_lengths), max(comment_lengths)], plt.ylim(), color="red")
    plt.plot([max(thread_lengths), max(thread_lengths)], plt.ylim(), color="blue")
    plt.legend()
    plt.title("Examples count vs Token length")
    plt.show()

def count_tokens_and_lengths(text_pairs):
    comment_tokens, response_comment_tokens = set(), set()
    comment_maxlen, response_maxlen = 0, 0
    for comment, response in text_pairs:
        comment_tok, response_tok = comment.split(), response.split()
        comment_maxlen = max(comment_maxlen, len(comment_tok))
        response_maxlen = max(response_maxlen, len(response_tok))
        comment_tokens.update(comment_tok)
        response_comment_tokens.update(response_tok)
    return comment_tokens, response_comment_tokens, comment_maxlen, response_maxlen
        
    print(f"Total Comment tokens: {len(comment_tokens)}")
    print(f"Total Response tokens: {len(response_comment_tokens)}")
    print(f"Max Comment length: {comment_maxlen}")
    print(f"Max Response length: {response_maxlen}")
    print(f"{len(text_pairs)} total pairs")

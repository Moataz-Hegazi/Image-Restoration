import matplotlib.pyplot as plt

def show_predictions(color, gray, predictions):

    fig, axes = plt.subplots(5,3,figsize=(15,15))

    for i in range(5):
        axes[0, 0].set_title('Grayscale Image')
        axes[i,0].imshow(gray[i].squeeze(), cmap="gray")
        axes[i,0].axis("off")

        axes[0, 1].set_title('Color Image')
        axes[i,1].imshow(color[i].permute(1,2,0))
        axes[i,1].axis("off")

        axes[0, 2].set_title('Predicted Image')
        axes[i,2].imshow(predictions[i].permute(1,2,0))
        axes[i,2].axis("off")

    plt.show()
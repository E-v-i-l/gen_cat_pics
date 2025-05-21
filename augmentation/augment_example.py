import cv2
import matplotlib.pyplot as plt
from augment import load_and_transform

def load_image(img_path):
    image = cv2.imread(img_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def augment_n_times(image, num_of_examples=10):
    aug_list = []
    for _ in range(num_of_examples):
        augmented = load_and_transform(image)
        aug_list.append(augmented)
    return aug_list

def show_augmentations(original, augmented_list):
    images = [original] + augmented_list
    titles = ['Original'] + [f'Aug {i+1}' for i in range(len(augmented_list))]
    cols = 6
    rows = (len(images) + cols - 1) // cols

    plt.figure(figsize=(16, 6))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if(__name__ == '__main__'):
    img_path = "./data/cats/1.jpeg"
    original = load_image(img_path)
    augmented_images = augment_n_times(img_path, num_of_examples=11)
    show_augmentations(original, augmented_images)
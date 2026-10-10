import numpy as np
from tensorflow.keras.datasets import cifar10
from skimage.feature import hog
from sklearn.decomposition import PCA
import joblib

def extract_hog_features(images):
    features = []
    for image in images:
        hog_feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(3, 3),
            channel_axis=-1
        )
        features.append(hog_feature)
    return np.array(features, dtype=np.float32)

def extract_spatial_color_features(images):
    features = []
    for image in images:
        image_features = []
        for row in range(4):
            for col in range(4):
                region = image[
                    row * 8:(row + 1) * 8,
                    col * 8:(col + 1) * 8
                ]
                for channel in range(3):
                    channel_data = region[:, :, channel]
                    image_features.append(np.mean(channel_data))
                    image_features.append(np.std(channel_data))
                    image_features.append(np.min(channel_data))
                    image_features.append(np.max(channel_data))
        features.append(image_features)
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype(np.float32) / 255.0

    print("Extracting HOG features...")
    x_train_hog = extract_hog_features(x_train)

    print("Extracting spatial color features...")
    x_train_color = extract_spatial_color_features(x_train)

    print("Combining features...")
    x_train_combined = np.concatenate([x_train_hog, x_train_color], axis=1)

    print("Fitting PCA...")
    pca = PCA(n_components=0.90, random_state=42)
    pca.fit(x_train_combined)

    print("Saving PCA model...")
    joblib.dump(pca, 'models/pca_model.joblib')
    print("PCA model saved successfully.")

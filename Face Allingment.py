# Imports for all libraries
import numpy as np
import cv2 as cv
import sklearn, subprocess, random, time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Global variables for model performance tuning and showing data
VISUALISE = True
CALCULATE_OPTIMAL = False
GREY_SCALE = False
SAVEPREDICTIONS = True
OPTIMAL_COMPONENTS = 256
TEST_SIZE = 0.2
RANDOM_STATE = random.randint(10, 30)
ORIGINAL_IMAGE_SIZE = (256, 256)
TRAIN_IMAGE_SIZE = (128, 128) # All data recordings in report set for a 128 x 128 train image size, (64, 64) for quicker but less accurate

##############################################################################################

def confirm_checksum(filename, true_checksum):
    
    checksum = subprocess.check_output(['shasum', filename]).decode('utf-8')
    assert checksum.split(' ')[0] == true_checksum, 'Checksum does not match for ' + filename + ' redownload the data.'

# Checksum verification for data files
confirm_checksum('training_images.npz', 'cf2a926d2165322adcd19d2e88b2eb1cd200ea5c')
confirm_checksum('examples.npz', '0fadc9226e4c0efb4479c5c1bf79491d75828ad3')
confirm_checksum('test_images.npz', 'c7b4b297c7e5c5009706f893ad1fb77a2aa73f95')

# Load data
print("Loading image data...")
data = np.load('training_images.npz', allow_pickle=True)
images = data['images']
pts = data['points']

test_data = np.load('test_images.npz', allow_pickle=True)
test_images = test_data['images']

example_data = np.load('examples.npz', allow_pickle=True)
example_images = example_data['images']

# Save points as CSV
def save_as_csv(points, location='.'):
    print("saving Predictions.csv data...")
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:]) == 44 * 2, 'wrong number of points provided. There should be 44 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

# Loss function
def loss_function(predicted_pts, actual_pts):
    
    assert predicted_pts.shape == actual_pts.shape, "Shape mismatch between predicted and actual points."
    distances = np.sqrt(np.sum((predicted_pts - actual_pts) ** 2, axis=-1))
    mean_loss = np.mean(distances)

    return mean_loss * (ORIGINAL_IMAGE_SIZE[0] / TRAIN_IMAGE_SIZE[0]) # UPDATED so it compensates for the image size reduction

##############################################################################################


def show_random_images_with_points(images, pts): # Shows 3 random images and plots the given points on them
    
    num_images = len(images)

    # Select 3 random indices
    random_indices = np.random.choice(num_images, size=3, replace=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    
    for i, ax in enumerate(axes.flat):
        idx = random_indices[i]
        img = images[idx]

        pt = pts[idx]
        if GREY_SCALE:
            img = mcolors.rgb_to_hsv(img)[:, :, 2] # show it as grey scale
        ax.imshow(img)
        ax.scatter(pt[:, 0], pt[:, 1], c='r', s=10)
        ax.set_title(f"Image {idx+1}")
        ax.axis('off')
    
    plt.show()

def visualize_best_worst_test_images(X_test, y_test, model): # Shows the models best and worst predictions
    
    # Compute losses for all test images
    losses = []
    for i in range(len(X_test)):
        test_image_pca = X_test[i]
        actual_keypoints = y_test[i]
        predicted_keypoints = model.predict(test_image_pca.reshape(1, -1))
        predicted_keypoints = predicted_keypoints.reshape(-1, 2)
        loss = loss_function(predicted_keypoints, actual_keypoints)
        losses.append(loss)

    # Find indices of best and worst predictions
    best_idx = np.argmin(losses)
    worst_idx = np.argmax(losses)

    # Visualize best and worst predictions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Best prediction
    test_image_sift = X_test[best_idx]
    actual_keypoints = y_test[best_idx]
    predicted_keypoints = model.predict(test_image_sift.reshape(1, -1))
    predicted_keypoints = predicted_keypoints.reshape(-1, 2)

    original_image = X_test_images[best_idx]
    original_image_resized = cv.resize(original_image, TRAIN_IMAGE_SIZE)
    
    axes[0].imshow(original_image_resized, cmap='gray')
    axes[0].scatter(actual_keypoints[:, 0], actual_keypoints[:, 1], c='g', marker='o', label='Actual Keypoints')
    axes[0].scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='r', marker='x', label='Predicted Keypoints')
    axes[0].set_title(f'Best Prediction (Loss: {losses[best_idx]:.4f})')
    axes[0].legend()

    # Worst prediction
    test_image = X_test[worst_idx]
    actual_keypoints = y_test[worst_idx]
    predicted_keypoints = model.predict(test_image_sift.reshape(1, -1))
    predicted_keypoints = predicted_keypoints.reshape(-1, 2)

    original_image = X_test_images[worst_idx]
    original_image_resized = cv.resize(original_image, TRAIN_IMAGE_SIZE)
    axes[1].imshow(original_image_resized, cmap='gray')
    axes[1].scatter(actual_keypoints[:, 0], actual_keypoints[:, 1], c='g', marker='o', label='Actual Keypoints')
    axes[1].scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='r', marker='x', label='Predicted Keypoints')
    axes[1].set_title(f'Worst Prediction (Loss: {losses[worst_idx]:.4f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return losses[best_idx], losses[worst_idx]


##############################################################################################

    
def find_optimal_pca_components(X_train): # Finds the optimal amount of PCA components to represents the majority of variance 
    
    print("Calculating optimal PCA components...")

    total_variance = []
    possible_components = [0, 16] + list(range(32, 256, 16)) # all number of components to test 

    for n_components in possible_components:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        explained_variance = sum(pca.explained_variance_ratio_) # get the sum of the variances of all components
        total_variance.append(explained_variance)
        print("Variance with ", n_components, " is ", explained_variance) 

    # Find the minimum number of components that explain at least 90% variance
    variance_threshold = 0.9
    optimal_idx = np.argmax(np.array(total_variance) >= variance_threshold)
    optimal_components = possible_components[optimal_idx]
    optimal_variance = total_variance[optimal_idx]

    # Plot the elbow curve
    if VISUALISE:
        plt.figure(figsize=(8, 6))
        plt.plot(possible_components, total_variance, marker='o')
        plt.xlabel('Number of PCA components')
        plt.ylabel('Explained variance ratio')
        plt.title('Elbow plot for PCA components')

        # Draw a red line at the chosen number of components
        plt.axvline(x=optimal_components, color='r', linestyle='--', label=f'Optimal components: {optimal_components}')
        plt.legend()
        plt.show()

    print(f"Optimal number of PCA components: {optimal_components} (Explained variance ratio: {optimal_variance:.4f})")

    return optimal_components


def preprocess_images(images, pts, img_size=TRAIN_IMAGE_SIZE): # Preprocesses the images to reduce processing time and increase accuracy (remove complexity)
    
    # If keypoints are not provided then just convert the image
    if pts is None:
        resized_images = [cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), img_size) if GREY_SCALE else cv.resize(img, img_size) for img in images] # Convert to grey scale
        normalized_images = np.array(resized_images) / 255.0 # Normalise pixel values
        return normalized_images, None

    else:
        print("Preprocessing images...")
        resized_images = [cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), img_size) if GREY_SCALE else cv.resize(img, img_size) for img in images]
        normalized_images = np.array(resized_images) / 255.0
        # Resize keypoints proportionally
        resized_pts = [kp * [new_w / w, new_h / h] for img, kp, (h, w), (new_h, new_w) in zip(images, pts, [img.shape[:2] for img in images], [img_size] * len(images))]
        points = np.array(resized_pts)
        # Return normalized images and resized keypoints
        return normalized_images, points


def predict_image_points(images, model, original_size = ORIGINAL_IMAGE_SIZE, train_size = TRAIN_IMAGE_SIZE): # predict image points from a image using the trained model
    
    predicted_keypoints_list = []
    processed_images, _ = preprocess_images(images, None) # pre-processing
    flat_images = processed_images.reshape(processed_images.shape[0], -1) # flattening
    
    for image in flat_images:
        # Apply PCA transformation to test data
        flat_image_pca = pca.transform(image.reshape(1, -1))
        
        # Make predictions
        predicted_keypoints = model.predict(flat_image_pca)
        predicted_keypoints = predicted_keypoints.reshape(-1, 2)
        
        # Calculate scaling factor
        scaling_factor = np.array(original_size) / np.array(train_size)
        
        # Upscale keypoints to original size
        predicted_keypoints *= scaling_factor
        
        predicted_keypoints_list.append(predicted_keypoints)

    return np.array(predicted_keypoints_list)


def modify_eye_lip_colour(images, points): # Modifys the eye and lip colour of a random example image using the models predicted points

    print("Modifying eye and lip colours...")

    # Randomly select an image and its corresponding points
    idx = np.random.randint(0, len(images))
    img = images[idx]
    pts = points[idx]

    # Indices corresponding to the eyes and lips in the points data
    left_eye_indices = list(range(20, 26))
    right_eye_indices = list(range(26, 32))
    mouth_indices = list(range(32, 44)) + [32]  # Closing the loop for mouth

    # Extract eye and lip points
    left_eye_pts = pts[left_eye_indices]
    right_eye_pts = pts[right_eye_indices]
    mouth_pts = pts[mouth_indices]

    plt.figure(num="Example image with Lip/Eye colour change")
    plt.imshow(img)
    plt.axis('off')

    # Draw filled polygons for left eye
    left_eye_patch = Polygon(left_eye_pts, closed=True, edgecolor=None, facecolor=np.random.rand(3), alpha=0.5)
    plt.gca().add_patch(left_eye_patch)

    # Draw filled polygons for right eye
    right_eye_patch = Polygon(right_eye_pts, closed=True, edgecolor=None, facecolor=np.random.rand(3), alpha=0.5)
    plt.gca().add_patch(right_eye_patch)

    # Draw filled polygons for mouth
    mouth_patch = Polygon(mouth_pts, closed=True, edgecolor=None, facecolor=np.random.rand(3), alpha=0.5)

    plt.gca().add_patch(mouth_patch)
    plt.show()

# Train linear regression model

def linearRegression(X_train, X_test, y_train, y_test): # the linear regression model, proved to be quite effective for the task even though the task favours more complex models
    
    X_train_flattened = X_train.reshape(X_train.shape[0], -1) # Flatten the data for the linear model
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # Define parameter grid for GridSearchCV
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}  # Regularization parameter values to search over

    # Create Ridge regression model
    ridge_model = Ridge()
    grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

    # Perform grid search to find the best hyperparameters
    grid_search.fit(X_train_flattened, y_train.reshape(y_train.shape[0], -1))

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on test set
    y_test_pred_flattened = best_model.predict(X_test_flattened)

    num_points = y_test.shape[1]
    y_test_pred = y_test_pred_flattened.reshape(y_test_pred_flattened.shape[0], num_points, 2)
    model_loss = loss_function(y_test_pred, y_test_labels)


    # Compute validation loss
    val_losses = -grid_search.cv_results_['mean_test_score']  # Get validation loss for each alpha value

    if VISUALISE:
        alphas = [param['alpha'] for param in grid_search.cv_results_['params']]
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, val_losses, marker='o')
        plt.title('Validation Loss vs Alpha Value')
        plt.xlabel('Alpha')
        plt.ylabel('Validation Loss')
        plt.xscale('log')
        plt.show()

    return best_model, model_loss


if __name__ == "__main__": 

    # Preprocess data
    processed_images, points = preprocess_images(images, pts)
    if VISUALISE:
        show_random_images_with_points(processed_images, points)

    # Split data into train and test data
    X_train_images, X_test_images, y_train_labels, y_test_labels = train_test_split(processed_images, points, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Flatten the images
    X_train_images_flat = X_train_images.reshape(X_train_images.shape[0], -1)
    X_test_images_flat = X_test_images.reshape(X_test_images.shape[0], -1)

    # Find optimal amount of components for PCA
    if CALCULATE_OPTIMAL:
        optimal_components = find_optimal_pca_components(X_train_images_flat)
    else:
        optimal_components = OPTIMAL_COMPONENTS
        print("Optimal amount of components: ", optimal_components)
    pca = PCA(n_components=optimal_components)

    # Fit the data into the amount of components
    pca_fit_start = time.time()
    X_train_images_pca = pca.fit_transform(X_train_images_flat)
    X_test_images_pca = pca.transform(X_test_images_flat)
    pca_fit_end = time.time()
    print("PCA fitting time: ", pca_fit_end - pca_fit_start)

    # Train linear regression model without augmentation
    start_time = time.time()
    print("Training Linear Regression model...")
    linear_model, model_loss = linearRegression(X_train_images_pca, X_test_images_pca, y_train_labels, y_test_labels)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Loss with Linear Regression: {model_loss}")
    print(f"Time taken for execution and training: {execution_time} seconds")

    # Analysis and evaluate the model
    if VISUALISE:
        best, worst = visualize_best_worst_test_images(X_test_images_pca, y_test_labels, linear_model)

    # Part two, Eye and lip coulour modification with example images
    example_points = predict_image_points(example_images, linear_model)
    if VISUALISE:
        show_random_images_with_points(example_images, example_points)
    modify_eye_lip_colour(example_images, example_points)

    # Save the results of test data as a csv file
    test_image_predictions = predict_image_points(test_images, linear_model)
    if SAVEPREDICTIONS:
        save_as_csv(test_image_predictions)
